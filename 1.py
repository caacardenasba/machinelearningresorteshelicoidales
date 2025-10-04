import numpy as np
import pandas as pd  # Necesario para crear el DataFrame y exportar
import os          # Necesario para manejar rutas de archivos

from ansys.dpf import core as dpf
from ansys.dpf.core import examples

# 1. Definición de la ruta de tu archivo (EXISTENTE)
filename = r"E:\Documentos Carlos\Profesional\Dependiente\7. IMAL\3. PROYECTOS\HELICOIDALES\HYUNDAI_STA_FE_01H067GT_TRA\3_SIMULACIÓN\HYUNDAI_STA_FE_01H067GT_TRA_files\dp0\SYS\MECH\file.rst"

# 2. Carga del Modelo y Ejecución del Operador (EXISTENTE)
model = dpf.Model(filename)

# 2. SE DEFINE EL OPERADOR 'disp_op'
disp_op = model.results.displacement() 

# 1. Obtener todos los resultados disponibles (los 15 pasos de tiempo)
fields = disp_op.outputs.fields_container()

# Lista para almacenar los resultados combinados
all_results = []

# 2. Iterar sobre los resultados disponibles (Pasos de 1 a 15)
# Requerimos una lista de los pasos de tiempo disponibles (Pasos 1, 2, 3...)
# Si tienes 15 iteraciones, la lista va de 1 a 15.
num_pasos = len(fields) # Obtiene el número total de campos disponibles (debería ser 15)

for paso_tiempo in range(1, num_pasos + 1):
    
    # SOLICITUD DE CAMPO ESPECÍFICO
    # Pedimos al modelo el desplazamiento SÓLO para el paso de tiempo actual.
    # Usamos la sintaxis de lista para asegurar que PyAnsys reconozca el argumento time_scoping.
    disp_result_for_step = model.results.displacement.fields_container[paso_tiempo - 1]
    
    # 3. Extraer los datos crudos y las IDs para el Field actual
    datos_desplazamiento = disp_result_for_step.data          
    nodos_ids = disp_result_for_step.scoping.ids
    
    # 4. Crear el DataFrame temporal para este paso de tiempo
    df_temp = pd.DataFrame(datos_desplazamiento, columns=['Despl_X', 'Despl_Y', 'Despl_Z'])
    
    # 5. Añadir metadatos cruciales: Node_ID y Paso_Tiempo
    df_temp['Node_ID'] = nodos_ids
    df_temp['Paso_Tiempo'] = paso_tiempo 
    
    # 6. Agregar el DataFrame temporal a la lista de resultados
    all_results.append(df_temp)

# 7. Combinar todos los resultados en un único DataFrame final
final_df = pd.concat(all_results, ignore_index=True)

# La tabla final (final_df) tendrá (15 iteraciones * N nodos) filas.

# 8. Exportar a CSV
output_directory = r"C:\Users\cacb2\Documents\pruebas" 
output_file = os.path.join(output_directory, "desplazamiento_nodal_transitorio.csv")
os.makedirs(output_directory, exist_ok=True)
final_df.to_csv(output_file, index=False, sep=';', decimal='.')

print(f"\n Datos de desplazamiento de las 15 iteraciones exportados con éxito a: {output_file}")