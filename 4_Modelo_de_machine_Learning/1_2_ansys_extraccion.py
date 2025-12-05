import numpy as np
import pandas as pd
from math import pi
from ansys.mapdl.reader import read_binary
import os
import glob
from pathlib import Path
import threading
import sys
import platform
import warnings
import gc
import signal

# Suprimir warnings de NumPy
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ===== MANEJO DE SEÑALES =====
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operación excedió el tiempo límite")

# ===== VALIDACIÓN =====
def validar_archivo_rst(rst_file, timeout_segundos=10):
    """Valida que el archivo RST exista y sea legible."""
    if not os.path.exists(rst_file):
        return False, "Archivo no existe"
    
    if os.path.getsize(rst_file) == 0:
        return False, "Archivo vacío"
    
    tamanio_mb = os.path.getsize(rst_file) / 1e6
    
    # Clasificar por tamaño - más restrictivo
    if tamanio_mb > 700:  # > 700MB se considera muy grande
        return False, f"Archivo muy grande ({tamanio_mb:.0f}MB) - se omite"
    elif tamanio_mb > 500:
        return "GRANDE", f"Archivo grande ({tamanio_mb:.0f}MB)"
    
    # Validación rápida sin leer todo el archivo
    result_holder = {"value": None, "error": None}
    
    def read_with_timeout():
        try:
            result_holder["value"] = read_binary(rst_file)
        except Exception as e:
            result_holder["error"] = str(e)[:50]
    
    thread = threading.Thread(target=read_with_timeout, daemon=True)
    thread.start()
    thread.join(timeout=timeout_segundos)
    
    if thread.is_alive():
        return False, f"Timeout ({timeout_segundos}s)"
    
    if result_holder["error"]:
        return False, result_holder["error"]
    
    if result_holder["value"] is None:
        return False, "No se pudo leer"
    
    result = result_holder["value"]
    
    try:
        if result.nsets == 0:
            return False, "Sin result sets"
    except:
        return False, "Archivo corrupto"
    
    return True, "OK"


def buscar_archivo_rst_sys1(carpeta_referencia, referencia):
    """Busca archivo .rst SOLO en SYS-1."""
    variantes_simulacion = ["3_SIMULACION", "3_SIMULACIÓN"]
    
    for carpeta_sim_nombre in variantes_simulacion:
        carpeta_sim = Path(carpeta_referencia) / carpeta_sim_nombre
        
        if not carpeta_sim.exists():
            continue
        
        carpetas_files = list(carpeta_sim.glob("*_files"))
        
        for carpeta_files in carpetas_files:
            rst_path = carpeta_files / "dp0" / "SYS-1" / "MECH" / "file.rst"
            
            if rst_path.exists():
                return str(rst_path)
    
    return None


# ===== CONFIGURACIÓN =====
class ConfiguracionResorte:
    def __init__(self, d, Dm, Na, Nt, Lf, G=80e3, tipo="AUTO"):
        self.d = d
        self.Dm = Dm
        self.Na = Na
        self.Nt = Nt
        self.Lf = Lf
        self.G = G
        self.tipo = tipo
        self.C = Dm / d
        self.p = Lf / Na
        self.K_wahl = (4*self.C - 1)/(4*self.C - 4) + 0.615/self.C
        self.factor_correccion_esfuerzo = 5.82


def detectar_eje(nodes):
    """Detecta el eje principal del resorte."""
    spans = nodes.max(axis=0) - nodes.min(axis=0)
    return {0: "X", 1: "Y", 2: "Z"}[np.argmax(spans)]


def reconstruir_geometria(nodes, verbose=False):
    """Reconstruye parámetros geométricos del resorte."""
    X, Y, Z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    axis = detectar_eje(nodes)

    if axis == "X":
        A = X; R1, R2 = Y, Z
    elif axis == "Y":
        A = Y; R1, R2 = X, Z
    else:
        A = Z; R1, R2 = X, Y

    Lf = float(np.percentile(A, 99) - np.percentile(A, 1))

    r = np.sqrt(R1**2 + R2**2)
    mask = (A > np.percentile(A, 5)) & (A < np.percentile(A, 95))
    r_filt = r[mask]

    r_inner = np.percentile(r_filt, 1)
    r_outer = np.percentile(r_filt, 99)

    d = float(r_outer - r_inner)
    Dm = float(r_outer + r_inner)

    A_c = A[mask]
    theta = np.arctan2(R2[mask], R1[mask])
    theta_u = np.unwrap(theta)

    A_norm = A_c - A_c.mean()
    m = np.sum(A_norm * theta_u) / np.sum(A_norm ** 2)
    total_angle = m * (A.max() - A.min())
    Na = abs(total_angle / (2 * pi))

    p = Lf / Na if Na > 0 else float("nan")
    Nt_estimado = Na + 2.0

    return {
        "axis": axis,
        "Lf": Lf,
        "Dm": Dm,
        "d": d,
        "Na": Na,
        "Nt": Nt_estimado,
        "p": p,
    }


def obtener_fuerza_global(result, step, axis="Z"):
    """Extrae fuerza de reacción."""
    try:
        rf_data = result.nodal_reaction_forces(step)
        
        if isinstance(rf_data, (list, tuple)) and len(rf_data) == 3:
            force_values = np.array(rf_data[0])
            dofs = np.array(rf_data[2])
            
            axis_to_dof = {"X": 1, "Y": 2, "Z": 3}
            target_dof = axis_to_dof[axis]
            
            mask_axial = (dofs == target_dof)
            forces_axial = force_values[mask_axial]
            
            if len(forces_axial) > 0:
                F_total = float(abs(forces_axial.sum()))
                return F_total
        
        return None
    except Exception:
        return None


def obtener_seqv_max(result, step, config):
    """Extrae esfuerzo de von Mises."""
    try:
        stress_tuple = result.nodal_stress(step)
        
        if not isinstance(stress_tuple, (list, tuple)) or len(stress_tuple) != 2:
            return None
        
        stress_data = np.array(stress_tuple[1])
        
        if stress_data.shape[1] < 6:
            return None
        
        Sx = stress_data[:, 0]
        Sy = stress_data[:, 1]
        Sz = stress_data[:, 2]
        Sxy = stress_data[:, 3]
        Syz = stress_data[:, 4]
        Szx = stress_data[:, 5]
        
        seqv = np.sqrt(
            0.5 * (
                (Sx - Sy)**2 + 
                (Sy - Sz)**2 + 
                (Sz - Sx)**2 + 
                6 * (Sxy**2 + Syz**2 + Szx**2)
            )
        )
        
        seqv_valid = seqv[np.isfinite(seqv) & (seqv > 0)]
        
        if len(seqv_valid) == 0:
            return None
        
        seqv_p999 = float(np.percentile(seqv_valid, 99.9))
        seqv_max = float(np.max(seqv_valid))
        
        if seqv_max > 3.0 * seqv_p999:
            seqv_raw_mpa = seqv_p999
        else:
            seqv_raw_mpa = seqv_max
        
        seqv_corregido_mpa = seqv_raw_mpa * config.factor_correccion_esfuerzo
        seqv_final_pa = seqv_corregido_mpa * 1e6
        
        return seqv_final_pa
    except Exception:
        return None


def obtener_deflexion_mejorada(result, step, axis, Lf):
    """Extrae deflexión."""
    try:
        nodes = result.grid.points
        axis_idx = {"X": 0, "Y": 1, "Z": 2}[axis]
        
        disp_tuple = result.nodal_displacement(step)
        
        if not isinstance(disp_tuple, (list, tuple)) or len(disp_tuple) != 2:
            return None, None
        
        disp_data = np.array(disp_tuple[1])
        
        if disp_data.shape[1] < 3:
            return None, None
        
        coords_axial = nodes[:, axis_idx]
        disp_axial_raw = disp_data[:, axis_idx]
        
        threshold_sup = np.percentile(coords_axial, 98)
        threshold_inf = np.percentile(coords_axial, 2)
        
        mask_sup = coords_axial >= threshold_sup
        mask_inf = coords_axial <= threshold_inf
        
        disp_sup_raw = disp_axial_raw[mask_sup]
        disp_inf_raw = disp_axial_raw[mask_inf]
        
        disp_sup_raw = disp_sup_raw[np.isfinite(disp_sup_raw)]
        disp_inf_raw = disp_inf_raw[np.isfinite(disp_inf_raw)]
        
        if len(disp_sup_raw) == 0 or len(disp_inf_raw) == 0:
            return None, None
        
        mean_sup_raw = np.mean(disp_sup_raw)
        mean_inf_raw = np.mean(disp_inf_raw)
        
        delta_raw = abs(mean_inf_raw - mean_sup_raw)
        
        if delta_raw > Lf:
            factor = 1000.0
        elif delta_raw > 0.01:
            factor = 1.0
        else:
            factor = 0.001
        
        disp_sup = disp_sup_raw / factor
        disp_inf = disp_inf_raw / factor
        
        p95_sup = np.percentile(np.abs(disp_sup), 95)
        p95_inf = np.percentile(np.abs(disp_inf), 95)
        
        if p95_sup < p95_inf:
            y = p95_inf
        else:
            y = p95_sup
        
        L = Lf - y
        
        if L < 0:
            return None, None
        
        return y, L
    except Exception:
        return None, None


def procesar_referencia_con_timeout(rst_file, referencia, G=80e3, timeout=120):
    """Procesa una referencia con timeout para evitar cuelgues."""
    result_holder = {"df": None, "error": None, "completed": False}
    
    def worker():
        try:
            result = read_binary(rst_file)
            nodes = result.grid.points
            geom = reconstruir_geometria(nodes, verbose=False)
            
            config = ConfiguracionResorte(
                d=geom['d'], Dm=geom['Dm'], Na=geom['Na'], 
                Nt=geom['Nt'], Lf=geom['Lf'], G=G, tipo="AUTO"
            )
            
            dataset = []
            max_steps = min(result.nsets, 50)  # Limitar a 50 steps máximo para archivos grandes
            
            for step in range(max_steps):
                try:
                    y, L = obtener_deflexion_mejorada(result, step, geom['axis'], config.Lf)
                    F = obtener_fuerza_global(result, step, axis=geom['axis'])
                    Seqv_max = obtener_seqv_max(result, step, config)

                    fila = {
                        "Referencia": referencia,
                        "d": config.d, "Dm": config.Dm, "C": config.C,
                        "Na": config.Na, "Nt": config.Nt, "p": config.p,
                        "Lf": config.Lf, "L": L, "y": y,
                        "F": F, "Seqv_max": Seqv_max, "step": step
                    }
                    dataset.append(fila)
                    
                    if (step + 1) % 10 == 0:
                        gc.collect()
                except Exception as e:
                    continue

            df = pd.DataFrame(dataset)
            df_clean = df.dropna(subset=['L', 'y', 'F', 'Seqv_max']).copy()
            
            if len(df_clean) > 0:
                df_clean['k'] = df_clean['F'] / df_clean['y']
            
            result_holder["df"] = df_clean
            result_holder["completed"] = True
            
        except Exception as e:
            result_holder["error"] = str(e)
    
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        print(f"   ⚠️ TIMEOUT después de {timeout}s - Archivo demasiado complejo")
        return pd.DataFrame(), "timeout"
    
    if result_holder["error"]:
        print(f"   ❌ Error: {result_holder['error'][:50]}")
        return pd.DataFrame(), "error"
    
    if result_holder["completed"] and result_holder["df"] is not None:
        print(f"   ✅ {len(result_holder['df'])} filas extraídas y limpiadas.")
        return result_holder["df"], "success"
    
    return pd.DataFrame(), "unknown"


def automatizar_extraccion(directorio_raiz, output_file="dataset_final_maestro.csv", G=80e3):
    """Extracción con guardado progresivo y manejo robusto de timeouts."""
    
    if not os.path.isdir(directorio_raiz):
        print(f"❌ ERROR: El directorio raíz '{directorio_raiz}' no existe.")
        return None

    patron_referencias = os.path.join(directorio_raiz, '*')
    carpetas_referencia = [p for p in glob.glob(patron_referencias) 
                          if os.path.isdir(p) and os.path.basename(p) not in 
                          ['venv', '__pycache__', '.git', 'PyAnsys', '.venv']]
    
    print(f"🔎 Se encontraron {len(carpetas_referencia)} carpetas.\n")
    print(f"🖥️  Sistema: {platform.system()}")
    print(f"📂 Directorio: {os.getcwd()}")
    print(f"📂 Salida: {os.path.abspath(output_file)}\n")
    
    print("📋 PROCESANDO REFERENCIAS (SOLO SYS-1):")
    print("=" * 100)
    
    dataframes_maestros = []
    referencias_procesadas = []
    referencias_saltadas = []
    
    for i, full_path in enumerate(carpetas_referencia, 1):
        referencia = os.path.basename(full_path)
        
        try:
            rst_file = buscar_archivo_rst_sys1(full_path, referencia)
            
            if rst_file is None:
                print(f"[{i:3d}] ⚠️  {referencia:50s} → Sin RST en SYS-1")
                referencias_saltadas.append((referencia, "Sin RST"))
                sys.stdout.flush()
                continue
            
            # Validar
            tamanio_mb = os.path.getsize(rst_file) / 1e6
            valido = validar_archivo_rst(rst_file, timeout_segundos=15)
            
            # Saltar archivos muy grandes
            if tamanio_mb > 700:
                print(f"[{i:3d}] ⚠️  {referencia:50s} → Muy grande ({tamanio_mb:.0f}MB) - SALTADO")
                referencias_saltadas.append((referencia, f"Muy grande ({tamanio_mb:.0f}MB)"))
                sys.stdout.flush()
                continue
            
            if valido == "GRANDE" or (valido[0] == True and tamanio_mb > 400):
                print(f"[{i:3d}] 📦 {referencia:50s} → Grande ({tamanio_mb:.0f}MB)")
                timeout = 180  # 3 minutos para archivos grandes
            elif valido[0] == False:
                print(f"[{i:3d}] ❌ {referencia:50s} → {valido[1]}")
                referencias_saltadas.append((referencia, valido[1]))
                sys.stdout.flush()
                continue
            else:
                print(f"[{i:3d}] ✅ {referencia:50s}")
                timeout = 60  # 1 minuto para archivos normales
            
            print(f"   ⚙️  Procesando: {referencia}")
            sys.stdout.flush()
            
            # Procesar con timeout
            df_referencia, status = procesar_referencia_con_timeout(
                rst_file, referencia=referencia, G=G, timeout=timeout
            )
            
            if status == "success" and not df_referencia.empty:
                dataframes_maestros.append(df_referencia)
                referencias_procesadas.append(referencia)
                
                # GUARDAR PROGRESO CADA 5 REFERENCIAS
                if len(dataframes_maestros) % 5 == 0:
                    df_temp = pd.concat(dataframes_maestros, ignore_index=True)
                    backup_name = f"backup_{len(dataframes_maestros)}refs.csv"
                    df_temp.to_csv(backup_name, index=False)
                    print(f"   💾 Backup guardado: {backup_name}")
                    
                # GUARDAR CSV PARCIAL (SOBRESCRIBIR)
                df_parcial = pd.concat(dataframes_maestros, ignore_index=True)
                df_parcial.to_csv(output_file, index=False)
                
            elif status == "timeout":
                referencias_saltadas.append((referencia, "Timeout"))
            else:
                referencias_saltadas.append((referencia, "Error procesamiento"))
            
            gc.collect()
            sys.stdout.flush()
        
        except KeyboardInterrupt:
            print("\n\n⚠️ INTERRUMPIDO POR USUARIO")
            print("💾 Guardando progreso...")
            break
        except Exception as e:
            print(f"[{i:3d}] ❌ {referencia:50s} → {str(e)[:50]}")
            referencias_saltadas.append((referencia, str(e)[:40]))
            sys.stdout.flush()

    # ===== RESUMEN =====
    print("\n" + "=" * 100)
    print("📊 RESUMEN:")
    print("=" * 100)
    print(f"   Total carpetas:        {len(carpetas_referencia)}")
    print(f"   ✅ Procesadas:          {len(referencias_procesadas)}")
    print(f"   ⚠️  Saltadas/Errores:    {len(referencias_saltadas)}")

    # ===== GUARDAR FINAL =====
    if dataframes_maestros:
        df_final = pd.concat(dataframes_maestros, ignore_index=True)
        df_final.to_csv(output_file, index=False)
        
        print(f"\n🎉 DATASET GUARDADO:")
        print(f"   📁 {os.path.abspath(output_file)}")
        print(f"   📊 Filas: {len(df_final)}")
        print(f"   🏷️  Referencias: {df_final['Referencia'].nunique()}")
        
        return df_final
    else:
        print("\n⚠️ No se extrajeron datos.")
        return None


if __name__ == "__main__":
    print("=" * 100)
    print("🚀 EXTRACCIÓN DE DATOS - VERSIÓN ROBUSTA")
    print("   📍 Solo SYS-1")
    print("   📍 Timeouts configurables")
    print("   📍 Guardado progresivo cada 5 referencias")
    print("=" * 100 + "\n")
    
    df_resultado = automatizar_extraccion(
        directorio_raiz="./",
        output_file="dataset_maestro_resortes.csv"
    )
    
    print(f"\n✅ Proceso completado.")