import os
import glob
import pandas as pd
import numpy as np

# Importamos las herramientas de PyAnsys
from ansys.dpf import core as dpf
from ansys.dpf.core import fields_factory
from ansys.dpf.core import locations
from ansys.dpf.core import operators as ops

field_with_classic_api = dpf.Field()
field_with_classic_api.location = locations.nodal
field_with_factory = fields_factory.create_scalar_field(10)

def get_data_from_rst(rst_path):
    """
    Extrae Desplazamiento y Esfuerzo para TODOS los 24 subpasos de carga.
    """
    print(f"  -> Procesando archivo: {rst_path}")

    try:
        data_source = dpf.DataSources(rst_path)
    except Exception as e:
        print(f"  [ERROR] No se pudo cargar el archivo DPF: {e}")
        return None

    # OBTENER MALLA Y TIEMPOS
    try:
        mesh_op = ops.mesh.mesh_provider(data_sources=data_source)
        mesh = mesh_op.outputs.mesh() 
        print(f"    Malla: Nodos={mesh.nodes.n_nodes}, Elementos={mesh.elements.n_elements}")
    except Exception as e:
        print(f"  [ERROR] No se pudo cargar la malla: {e}")
        return None

    # --- OBTENER TODOS LOS 24 SUBPASOS (SOLUCIÓN DE CONTINGENCIA) ---
    try:
        # Forzamos la lista de subpasos [1.0, 2.0, ..., 24.0]
        num_substeps = 24 
        all_time_steps = np.arange(1.0, float(num_substeps) + 1.0).astype(np.float64)
        print(f"    Forzando {len(all_time_steps)} subpasos de carga para iterar.")
    except Exception as e:
        all_time_steps = np.array([1.0], dtype=np.float64) 
    
    if not all_time_steps.size:
        all_time_steps = np.array([1.0], dtype=np.float64)

    
    # Extraemos el nombre del proyecto
    project_path_parts = rst_path.split(os.sep)
    try:
        project_name = project_path_parts[project_path_parts.index('3_SIMULACION') - 1]
    except ValueError:
        project_name = os.path.basename(os.path.dirname(os.path.dirname(rst_path)))

    
    # 2. ITERACIÓN Y EXTRACCIÓN DE DATOS POR PASO DE CARGA
    extracted_data = []

    for time_step in all_time_steps:
        
        # Valor del paso de carga (float)
        current_time_value = float(time_step) 
        
        # --- A. DESPLAZAMIENTO ---
        displacement_op = ops.result.displacement(data_sources=data_source, time_scoping=[current_time_value])
        displacement_norm_op = ops.math.norm(displacement_op)
        disp_fields = displacement_norm_op.outputs[0] 
        
        # --- B. ESFUERZOS ---
        stress_op = ops.result.stress(data_sources=data_source, 
                                      requested_location=dpf.locations.nodal, 
                                      time_scoping=[current_time_value])
        stress_fields = stress_op.outputs[0]
        
        
        # 3. CÁLCULO FINAL Y GUARDADO
        
        # Máximo Desplazamiento
        disp_data = disp_fields.get_data()
        max_displacement = disp_data.max()

        # Máximo Esfuerzo
        stress_data = stress_fields.get_data()
        max_von_mises = stress_data.max()
        
        # LA FUERZA SE EXCLUYE POR INCOMPATIBILIDAD
        
        # Guardar la fila de datos
        extracted_data.append({
            'Proyecto': project_name, 
            'Paso_Carga': current_time_value, 
            'Max_Desplazamiento': max_displacement,
            'Max_Von_Mises': max_von_mises,
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