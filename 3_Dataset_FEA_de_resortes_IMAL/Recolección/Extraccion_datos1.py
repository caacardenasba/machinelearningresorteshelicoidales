import os
import glob
import pandas as pd
import numpy as np

# Importamos las herramientas de PyAnsys para el postprocesamiento (DPF)
from ansys.dpf import core as dpf
from ansys.dpf.core import operators as ops



def get_data_from_rst(rst_path):
    """
    Función que lee un archivo de resultados de ANSYS (.rst) y extrae
    tiempos, desplazamientos, esfuerzos y fuerzas de reacción.
    Utiliza la nomenclatura de operadores compatible con la versión DPF.
    """
    print(f"  -> Procesando archivo: {rst_path}")

    # 1. Conexión a la Data Source
    try:
        data_source = dpf.DataSources(rst_path)
    except Exception as e:
        print(f"  [ERROR] No se pudo cargar el archivo DPF: {e}")
        return None

    # 2. Definición de los Operadores de Extracción

        # a) OBTENER EL OBJETO DE MALLA (USANDO 'mesh_provider')
    try:
        # 1. Definimos el operador
        mesh_op = ops.mesh.mesh_provider(data_sources=data_source)
        # 2. Obtenemos el objeto mesh
        mesh = mesh_op.outputs.mesh() 
        print(f"    Malla: Nodos={mesh.nodes.n_nodes}, Elementos={mesh.elements.n_elements}")
    except Exception as e:
        print(f"  [ERROR] No se pudo cargar la malla: {e}")
        return None

    # b) OBTENER PASOS DE TIEMPO (CORREGIDO: usando 'time_steps')
    try:
        time_op = ops.result.time_steps(data_sources=data_source)
        all_time_steps = time_op.outputs.time_steps.array
    except AttributeError:
        # Fallback si 'time_steps' no funciona, asume un análisis estático de un solo paso (paso 1)
        all_time_steps = np.array([1])
        print("    Advertencia: No se encontraron pasos de tiempo. Asumiendo análisis estático (paso 1).")
    
    if not all_time_steps.size:
        all_time_steps = np.array([1]) # Asegura al menos el paso 1 para estático


    for time_step in all_time_steps:
        
        # Conectar el paso de tiempo a los operadores que lo necesitan
        # 1. Conexión al Desplazamiento (el operador de RESULTADO)
        displacement_op.inputs.time_scoping.connect([time_step])
        
        # 2. Conexión a Esfuerzo
        stress_op.inputs.time_scoping.connect([time_step])
        
        # 3. Conexión a Fuerza
        reaction_force_op.inputs.time_scoping.connect([time_step])


        # --- Obtener los resultados ---

        # Desplazamiento (Máximo Valor en todos los nodos)
        # Aquí pedimos la salida del operador de la NORMA
        disp_fields = displacement_norm_op.outputs.fields_container()
        # ... (resto del cálculo de max_displacement)

# d) Operador para obtener los Esfuerzos (Asumimos Von Mises o lo calculamos si es un tensor)
    # Volvemos a la definición más simple y antigua, que devuelve el campo de esfuerzos.
    stress_op = ops.result.stress(data_sources=data_source,
                                  requested_location=dpf.locations.nodal) 
    
    # IMPORTANTE: Si 'stress_op' devuelve un campo tensorial (6 componentes), 
    # la siguiente parte del script (alrededor de la línea 90) que usa 'np.max' 
    # podría necesitar ajustarse si no devuelve Von Mises por defecto.
    # Por ahora, dejaremos el resto del script intacto y probaremos.

    # e) Operador para obtener la Fuerza de Reacción (Reaction Force) - ¡SOLUCIÓN FINAL!
    # Obtenemos el campo de fuerzas sin el operador de suma, y lo sumaremos con NumPy más tarde.
    reaction_force_op = ops.result.nodal_force(data_sources=data_source) 
    # Ya no necesitamos sum_reaction_force_op


    # 3. Iteración y Extracción de Datos
    extracted_data = []
    
    # Extraemos el nombre del proyecto de la ruta
    # Esto asume que el nombre del proyecto es la carpeta inmediatamente superior a 3_SIMULACION
    project_path_parts = rst_path.split(os.sep)
    try:
        # Busca el nombre del proyecto que está antes de '3_SIMULACION' en la ruta.
        project_name = project_path_parts[project_path_parts.index('3_SIMULACION') - 1]
    except ValueError:
        project_name = os.path.basename(os.path.dirname(os.path.dirname(rst_path)))


    for time_step in all_time_steps:
        # Establecer el paso de tiempo para los operadores
        
        # 1. Conectar tiempo al operador de Desplazamiento BASE (el que tiene tiempo)
        displacement_op.inputs.time_scoping.connect([time_step])
        
        # 2. Conectar tiempo a los otros operadores de Resultado
        stress_op.inputs.time_scoping.connect([time_step])
        reaction_force_op.inputs.time_scoping.connect([time_step]) # Agregamos la fuerza


        # Obtener los resultados (se ejecutan en este punto)

        # Desplazamiento (Máximo Valor en todos los nodos)
        disp_fields = displacement_norm_op.outputs.fields_container()
        max_displacement = np.max([field.data.max() for field in disp_fields])

        # Esfuerzo (Máximo Esfuerzo de Von Mises en todos los nodos)
        stress_fields = stress_op.outputs.fields_container()
        max_von_mises = np.max([field.data.max() for field in stress_fields])

        # Fuerza de Reacción (Magnitud total de la suma de fuerzas)
        # 1. Obtenemos el contenedor de campos del operador de fuerza.
        rf_fields = reaction_force_op.outputs.fields_container()
        
        # 2. Sumamos las fuerzas de todos los nodos en el primer campo (paso de tiempo)
        # Esto suma Fx, Fy, Fz en todos los nodos para obtener la fuerza de reacción TOTAL.
        total_force_vector = np.sum(rf_fields[0].data, axis=0) 
        
        # 3. Calculamos la magnitud (Norma) de ese vector total.
        total_reaction_force_norm = np.linalg.norm(total_force_vector)


        # Guardar la fila de datos
        extracted_data.append({
            'Proyecto': project_name, 
            'Tiempo': time_step,
            'Max_Desplazamiento': max_displacement,
            'Max_Von_Mises': max_von_mises,
            'Total_Reaction_Force_Norm': total_reaction_force_norm,
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
            
    # Combina todos los DataFrames
    if all_data_frames:
        final_df = pd.concat(all_data_frames, ignore_index=True)
        
        # Guarda el DataFrame final en un archivo CSV
        final_df.to_csv(output_filename, index=False)
        print("\n" + "="*50)
        print(f" ¡Éxito! Datos exportados a '{output_filename}'")
        print(f"El dataset final tiene {len(final_df)} filas.")
        print("="*50)
    else:
        print("\n No se pudieron extraer datos de ningún archivo. Revisar errores anteriores.")

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