import os
import glob
import pandas as pd
import numpy as np

# Importamos las herramientas de PyAnsys para el postprocesamiento (DPF)
from ansys.dpf.core import operators as ops 
from ansys.dpf import core as dpf





def get_data_from_rst(rst_path):
    """
    Función que lee un archivo de resultados de ANSYS (.rst) y extrae
    tiempos, desplazamientos, esfuerzos y fuerzas de reacción.
    Se definen los operadores de resultado DENTRO del bucle para resolver
    el NameError y los problemas de conexión de tiempo.
    """
    print(f"  -> Procesando archivo: {rst_path}")

    # 1. Conexión a la Data Source
    try:
        data_source = dpf.DataSources(rst_path)
    except Exception as e:
        print(f"  [ERROR] No se pudo cargar el archivo DPF: {e}")
        return None

    # 2. DEFINICIÓN DE OPERADORES INICIALES (MALLA Y TIEMPOS)
    # a) OBTENER EL OBJETO DE MALLA (USANDO 'mesh_provider')
    try:
        mesh_op = ops.mesh.mesh_provider(data_sources=data_source)
        mesh = mesh_op.outputs.mesh() 
        print(f"    Malla: Nodos={mesh.nodes.n_nodes}, Elementos={mesh.elements.n_elements}")
    except Exception as e:
        print(f"  [ERROR] No se pudo cargar la malla: {e}")
        return None

    # b) OBTENER PASOS DE TIEMPO (USANDO SOPORTE DE TIEMPO PARA TODOS LOS PASOS)
    try:
        # Usamos el operador de PyAnsys completo (solucionando el AttributeError)
        time_op = dpf.operators.result.time_freq_steps(data_sources=data_source)
        
        # Accedemos a la lista de pasos de tiempo/subpasos
        all_time_steps = time_op.outputs.time_steps.array.astype(np.float64)
        
        print(f"    Encontrados {len(all_time_steps)} subpasos (resultados) para iterar.")

    except Exception as e:
        # Si falla (simulación estática o error de sintaxis), asumimos el paso 1
        print(f"    [AVISO] Falló la extracción de subpasos ({e}). Asumiendo solo el paso final.")
        all_time_steps = np.array([1.0], dtype=np.float64) 

    if not all_time_steps.size:
        all_time_steps = np.array([1.0], dtype=np.float64)

    
    # 3. ITERACIÓN Y EXTRACCIÓN DE DATOS
    extracted_data = []
    
    # Extraemos el nombre del proyecto de la ruta
    project_path_parts = rst_path.split(os.sep)
    try:
        project_name = project_path_parts[project_path_parts.index('3_SIMULACION') - 1]
    except ValueError:
        project_name = os.path.basename(os.path.dirname(os.path.dirname(rst_path)))


    for time_step in all_time_steps:
        
        # --- DEFINICIONES (MOVIDAS AL INICIO DEL BUCLE) ---
        
        # OBTENEMOS EL VALOR FLOTANTE NATIVO DE PYTHON
        current_time_value = float(time_step)
        # Reemplazamos el valor del tiempo por el índice de subpaso
        # Esto le dice a PyAnsys "dame el resultado del subpaso número 5, 6, 7..."
        substep_index = int(time_step) # Tomamos el paso (que debe ser 1, 2, 3...) 
        
        # 1. Desplazamiento (Definición y Conexión)
        displacement_op = ops.result.displacement(data_sources=data_source, time_scoping=[current_time_value])
        displacement_norm_op = ops.math.norm(displacement_op)
        
        # 2. Esfuerzos (Definición y Conexión)
        stress_op = ops.result.stress(data_sources=data_source, requested_location=dpf.locations.nodal, time_scoping=[current_time_value])
        
              
        # --- OBTENER LOS CONTENEDORES DE RESULTADOS (DEBEN IR AQUÍ) ---

        disp_fields = displacement_norm_op.outputs[0]  # Define el contenedor
        stress_fields = stress_op.outputs[0] # Define el contenedor

        # Desplazamiento (Máximo)
        disp_data = np.asarray(disp_fields.get_data())
        max_displacement = disp_data.max()

        # Esfuerzo (Máximo)
        stress_data = np.asarray(stress_fields.get_data())
        max_von_mises = stress_data.max()

       # --- CÁLCULO DE FUERZA DE REACCIÓN (BLOQUE SEGURO) ---
        try:
            # Revertimos a support_reaction (Nomenclatura que debe funcionar)
            reaction_force_op = dpf.operators.result.support_reaction(data_sources=data_source, time_scoping=[current_time_value])
            rf_fields = reaction_force_op.outputs[0] # Acceso al Field
            
            # ... (Lógica de cálculo de fuerza) ...
            final_reaction_value = np.linalg.norm(total_force_vector)
            
        except Exception as e:
            # Si hay un error, la fuerza será 0.0
            print(f"    [AVISO] Falló la extracción de fuerza: {e}. Asumiendo 0.0")
            final_reaction_value = 0.0    

        # Guardar la fila de datos
        extracted_data.append({
            'Proyecto': project_name, 
            'Tiempo': current_time_value, # Usamos current_time_value para consistencia
            'Max_Desplazamiento': max_displacement,
            'Max_Von_Mises': max_von_mises,
            'Total_Reaction_Force_Norm': final_reaction_value, # USAR LA VARIABLE FINAL Y SEGURA
            'RST_Source': rst_path 
        })

    return pd.DataFrame(extracted_data)



# ----------------------------------------------------------------------
# FUNCIÓN DE NAVEGACIÓN (LÓGICA DE BÚSQUEDA AJUSTADA)
# ----------------------------------------------------------------------

def process_all_projects(root_directory, output_filename="ansys_extracted_data.csv"):
    """
    Busca archivos .rst dentro de la estructura de carpetas:
    ROOT_DIR / [Proyecto X] / 3_SIMULACION / ... / file.rst
    """
    print(f"Iniciando la búsqueda de archivos .rst en: {root_directory}")

        # Este patrón busca recursivamente ('**') la carpeta '3_SIMULACION', y luego
    # busca cualquier archivo .rst dentro de sus subcarpetas ('**/*.rst').
    # Esto es más seguro que el intento anterior.
    rst_pattern = os.path.join(root_directory, '**', '3_SIMULACION', '**', '*.rst')
    rst_files = glob.glob(rst_pattern, recursive=True)
    if not rst_files:
        print("---")
        print(" No se encontró ningún archivo .rst dentro de la estructura. ¡Verifica la ruta y el nombre '3_SIMULACION'!")
        print(f"Patrón de búsqueda usado: {rst_pattern}")
        return

    print(f"--- Archivos .rst encontrados: {len(rst_files)}")

    all_data_frames = []

    for rst_path in rst_files:
        df = get_data_from_rst(rst_path)
        if df is not None:
            all_data_frames.append(df)
     # Manejo y guardado del DataFrame final (Solución a NameError y IndentationError)
    final_df = pd.DataFrame() # Inicialización segura

    if all_data_frames:
        final_df = pd.concat(all_data_frames, ignore_index=True)
        
        # EL BLOQUE DE ABAJO ESTÁ CORRECTAMENTE INDENTADO (MOVIDO A LA DERECHA)
        
        # Guardar el archivo CSV con delimitadores compatibles (¡IMPORTANTE!)
        final_df.to_csv(output_filename, index=False, sep=';', decimal='.', encoding='utf-8')

        print("\n" + "="*50)
        print(f"¡Éxito! Datos exportados a '{output_filename}'")
        print(f"El dataset final tiene {len(final_df)} filas.")
        print("="*50)
    else:
        print("\n No se pudieron extraer datos de ningún archivo para el CSV.")

# ----------------------------------------------------------------------
#                      VARIABLES A AJUSTAR
# ----------------------------------------------------------------------

# **AJUSTAMOS ESTA RUTA:** Debe ser la carpeta que contiene las carpetas de Proyecto (DAIHATSU_TERIOS_2006_2018_TRA)
# Basado en tu ruta: E:\Trabajo_de_grado\2_Proyecto\3_Dataset_FEA_de_resortes_IMAL\Recolección\DAIHATSU_TERIOS_2006_2018_TRA\...
PROJECTS_ROOT_PATH = r"E:\Trabajo_de_grado\2_Proyecto\3_Dataset_FEA_de_resortes_IMAL\Recolección\DAIHATSU_TERIOS_2006_2018_TRA"

# Nombre del archivo de salida
OUTPUT_FILE = "dataset_para_ia.csv"

# Llamamos a la función principal para comenzar la ejecución
if __name__ == "__main__":
    process_all_projects(PROJECTS_ROOT_PATH, OUTPUT_FILE)