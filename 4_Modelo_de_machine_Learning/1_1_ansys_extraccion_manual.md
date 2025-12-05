# 📊 Documentación Técnica: Script de Extracción de Datos FEA de Ansys

## 🎯 Propósito General

Este script automatiza la extracción masiva de datos de simulaciones de elementos finitos (FEA) de resortes helicoidales desde archivos binarios `.rst` de Ansys MAPDL. Su objetivo es construir un dataset estructurado para análisis de machine learning que correlacione parámetros geométricos con respuestas mecánicas.

---

## 🏗️ Arquitectura del Sistema

### **Flujo de Trabajo Principal**

```
Directorio Raíz
    ↓
Búsqueda de Carpetas de Referencias
    ↓
Localización de archivos .rst (SYS-1 únicamente)
    ↓
Validación de archivos (tamaño, integridad, timeout)
    ↓
Lectura y Procesamiento con Timeout
    ↓
Extracción de datos por step de simulación:
    - Geometría reconstruida
    - Deflexión
    - Fuerza de reacción
    - Esfuerzo von Mises
    ↓
Limpieza y guardado progresivo
    ↓
Dataset CSV consolidado
```

---

## 📦 Componentes Principales

### **1. Gestión de Seguridad y Timeouts**

#### `TimeoutError` (Clase de Excepción)
```python
class TimeoutError(Exception)
```
- Excepción personalizada para manejar operaciones que exceden límites de tiempo

#### `timeout_handler(signum, frame)`
- **Propósito**: Handler de señales POSIX para interrupciones por timeout
- **Limitación**: Solo funciona en sistemas Unix/Linux
- **Nota**: Actualmente declarado pero no implementado en el flujo principal

---

### **2. Validación de Archivos**

#### `validar_archivo_rst(rst_file, timeout_segundos=10)`

**Propósito**: Validación multi-nivel de archivos RST antes del procesamiento completo

**Niveles de Validación**:

1. **Existencia física**
   ```python
   if not os.path.exists(rst_file): return False, "Archivo no existe"
   ```

2. **Tamaño del archivo**
   - `> 700 MB` → Rechazado (muy grande)
   - `> 500 MB` → Marcado como "GRANDE"
   - Previene consumo excesivo de memoria

3. **Lectura en thread aislado**
   ```python
   thread = threading.Thread(target=read_with_timeout, daemon=True)
   thread.start()
   thread.join(timeout=timeout_segundos)
   ```
   - Evita bloqueos indefinidos
   - Thread daemon permite terminación limpia

4. **Validación estructural**
   ```python
   if result.nsets == 0: return False, "Sin result sets"
   ```
   - Verifica presencia de steps de simulación

**Retorno**: 
- `(True, "OK")` → Válido
- `("GRANDE", mensaje)` → Válido pero pesado
- `(False, razón)` → Inválido

---

### **3. Búsqueda de Archivos**

#### `buscar_archivo_rst_sys1(carpeta_referencia, referencia)`

**Propósito**: Localización específica de archivos `.rst` en estructura de Ansys Workbench

**Ruta esperada**:
```
carpeta_referencia/
└── 3_SIMULACION (o 3_SIMULACIÓN)
    └── *_files/
        └── dp0/
            └── SYS-1/
                └── MECH/
                    └── file.rst  ← Objetivo
```

**Características**:
- Maneja variantes de encoding (`SIMULACION` vs `SIMULACIÓN`)
- Solo busca en sistema `SYS-1` (filtro deliberado)
- Retorna `None` si no encuentra el archivo

---

### **4. Configuración de Resortes**

#### `ConfiguracionResorte` (Clase)

**Parámetros de entrada**:

| Parámetro | Descripción | Unidad |
|-----------|-------------|---------|
| `d` | Diámetro del alambre | mm |
| `Dm` | Diámetro medio del resorte | mm |
| `Na` | Número de espiras activas | - |
| `Nt` | Número total de espiras | - |
| `Lf` | Longitud libre | mm |
| `G` | Módulo de corte | MPa |

**Propiedades calculadas**:

1. **Índice de resorte (C)**:
   ```python
   C = Dm / d
   ```
   - Relación diámetro medio / diámetro alambre
   - Típicamente 4 < C < 12

2. **Paso (p)**:
   ```python
   p = Lf / Na
   ```
   - Distancia axial entre espiras

3. **Factor de Wahl (K_wahl)**:
   ```python
   K_wahl = (4*C - 1)/(4*C - 4) + 0.615/C
   ```
   - Corrección de concentración de esfuerzos
   - Considera curvatura y esfuerzo cortante directo

4. **Factor de corrección de esfuerzo**:
   ```python
   factor_correccion_esfuerzo = 5.82
   ```
   - **Crítico**: Ajusta resultados FEA a teoría de resortes
   - Calibrado empíricamente (valor a validar)

---

### **5. Reconstrucción Geométrica**

#### `detectar_eje(nodes)`

**Propósito**: Identificar el eje longitudinal del resorte

**Algoritmo**:
```python
spans = nodes.max(axis=0) - nodes.min(axis=0)
return {0: "X", 1: "Y", 2: "Z"}[np.argmax(spans)]
```
- Calcula el rango en cada dirección
- Selecciona la dirección de mayor extensión

---

#### `reconstruir_geometria(nodes, verbose=False)`

**Propósito**: Ingeniería inversa de parámetros geométricos desde la malla FEA

**Proceso detallado**:

1. **Separación de coordenadas**:
   ```python
   if axis == "X":
       A = X; R1, R2 = Y, Z  # A = axial, R1/R2 = radiales
   ```

2. **Longitud libre (Lf)**:
   ```python
   Lf = np.percentile(A, 99) - np.percentile(A, 1)
   ```
   - Usa percentiles para robustez ante outliers
   - Evita nodos extremos de malla

3. **Radio interno/externo**:
   ```python
   r = np.sqrt(R1**2 + R2**2)  # Radio desde eje central
   r_inner = np.percentile(r_filt, 1)
   r_outer = np.percentile(r_filt, 99)
   ```
   - Filtra región central del resorte (5%-95%)
   - Elimina efectos de extremos

4. **Diámetros**:
   ```python
   d = r_outer - r_inner      # Diámetro del alambre
   Dm = r_outer + r_inner     # Diámetro medio
   ```

5. **Número de espiras activas (Na)** - **Método sofisticado**:
   ```python
   theta = np.arctan2(R2, R1)           # Ángulo polar
   theta_u = np.unwrap(theta)            # Desenvuelve discontinuidades
   A_norm = A_c - A_c.mean()             # Centra coordenadas
   m = np.sum(A_norm * theta_u) / np.sum(A_norm ** 2)  # Regresión lineal ponderada
   total_angle = m * (A.max() - A.min())
   Na = abs(total_angle / (2 * pi))
   ```
   
   **Interpretación**:
   - `m`: Pendiente de la hélice (rad/mm)
   - `total_angle`: Rotación total en radianes
   - `Na`: Número de vueltas completas (2π)

6. **Paso y espiras totales**:
   ```python
   p = Lf / Na
   Nt_estimado = Na + 2.0  # Convención: 2 espiras inactivas
   ```

**Retorno**: Diccionario con 7 parámetros geométricos

---

### **6. Extracción de Resultados FEA**

#### `obtener_fuerza_global(result, step, axis="Z")`

**Propósito**: Sumar fuerzas de reacción en dirección axial

**Proceso**:
```python
rf_data = result.nodal_reaction_forces(step)
# rf_data = (force_values, nodes, dofs)

axis_to_dof = {"X": 1, "Y": 2, "Z": 3}
mask_axial = (dofs == target_dof)
F_total = abs(forces_axial.sum())
```

**Aspectos clave**:
- DOF mapping: X→1, Y→2, Z→3 (convención Ansys)
- Suma algebraica (considera signos)
- Valor absoluto final (magnitud de compresión/extensión)

---

#### `obtener_seqv_max(result, step, config)`

**Propósito**: Calcular esfuerzo equivalente de von Mises

**Algoritmo completo**:

1. **Extracción de tensor de esfuerzos**:
   ```python
   stress_data = np.array(stress_tuple[1])
   # Columnas: [Sx, Sy, Sz, Sxy, Syz, Szx]
   ```

2. **Cálculo de von Mises**:
   ```python
   seqv = np.sqrt(
       0.5 * (
           (Sx - Sy)**2 + (Sy - Sz)**2 + (Sz - Sx)**2 + 
           6 * (Sxy**2 + Syz**2 + Szx**2)
       )
   )
   ```
   - Fórmula estándar para estado 3D de esfuerzos

3. **Filtrado de outliers**:
   ```python
   seqv_p999 = np.percentile(seqv_valid, 99.9)
   seqv_max = np.max(seqv_valid)
   
   if seqv_max > 3.0 * seqv_p999:
       seqv_raw_mpa = seqv_p999  # Descarta outlier extremo
   else:
       seqv_raw_mpa = seqv_max
   ```
   - Protección contra singularidades numéricas

4. **Corrección de esfuerzo**:
   ```python
   seqv_corregido_mpa = seqv_raw_mpa * 5.82
   seqv_final_pa = seqv_corregido_mpa * 1e6  # MPa → Pa
   ```
   - Factor 5.82: **Ajuste crítico a validar**
   - Compensa diferencias entre FEA y teoría analítica

---

#### `obtener_deflexion_mejorada(result, step, axis, Lf)`

**Propósito**: Calcular deflexión del resorte con detección automática de unidades

**Proceso complejo**:

1. **Extracción de desplazamientos**:
   ```python
   disp_data = result.nodal_displacement(step)
   disp_axial_raw = disp_data[:, axis_idx]
   ```

2. **Identificación de extremos**:
   ```python
   threshold_sup = np.percentile(coords_axial, 98)
   threshold_inf = np.percentile(coords_axial, 2)
   
   mask_sup = coords_axial >= threshold_sup  # Cara superior
   mask_inf = coords_axial <= threshold_inf  # Cara inferior
   ```

3. **Detección de unidades** - **Sistema inteligente**:
   ```python
   delta_raw = abs(mean_inf_raw - mean_sup_raw)
   
   if delta_raw > Lf:
       factor = 1000.0    # Datos en μm → convertir a mm
   elif delta_raw > 0.01:
       factor = 1.0       # Datos en mm (correcto)
   else:
       factor = 0.001     # Datos en m → convertir a mm
   ```
   - **Heurística basada en coherencia física**
   - Compara deflexión con longitud libre

4. **Cálculo de deflexión robusta**:
   ```python
   p95_sup = np.percentile(np.abs(disp_sup), 95)
   p95_inf = np.percentile(np.abs(disp_inf), 95)
   
   y = max(p95_sup, p95_inf)  # Deflexión máxima
   L = Lf - y                  # Longitud comprimida
   ```
   - Usa percentiles 95 para robustez
   - Considera la cara con mayor desplazamiento

**Retorno**: `(y, L)` → (deflexión, longitud comprimida)

---

### **7. Procesamiento con Timeout**

#### `procesar_referencia_con_timeout(rst_file, referencia, G=80e3, timeout=120)`

**Propósito**: Procesar archivo completo con protección contra cuelgues

**Arquitectura de threading**:

```python
result_holder = {"df": None, "error": None, "completed": False}

def worker():
    # Procesamiento completo aquí
    result_holder["df"] = df_clean
    result_holder["completed"] = True

thread = threading.Thread(target=worker, daemon=True)
thread.start()
thread.join(timeout=timeout)

if thread.is_alive():
    return pd.DataFrame(), "timeout"
```

**Flujo interno del worker**:

1. **Lectura del archivo**:
   ```python
   result = read_binary(rst_file)
   nodes = result.grid.points
   ```

2. **Reconstrucción geométrica**:
   ```python
   geom = reconstruir_geometria(nodes)
   config = ConfiguracionResorte(...)
   ```

3. **Iteración por steps**:
   ```python
   max_steps = min(result.nsets, 50)  # Límite para archivos grandes
   
   for step in range(max_steps):
       y, L = obtener_deflexion_mejorada(...)
       F = obtener_fuerza_global(...)
       Seqv_max = obtener_seqv_max(...)
       
       fila = {
           "Referencia": referencia,
           "d": config.d, "Dm": config.Dm, "C": config.C,
           "Na": config.Na, "Nt": config.Nt, "p": config.p,
           "Lf": config.Lf, "L": L, "y": y,
           "F": F, "Seqv_max": Seqv_max, "step": step
       }
       dataset.append(fila)
   ```

4. **Limpieza de datos**:
   ```python
   df_clean = df.dropna(subset=['L', 'y', 'F', 'Seqv_max'])
   df_clean['k'] = df_clean['F'] / df_clean['y']  # Rigidez
   ```

5. **Gestión de memoria**:
   ```python
   if (step + 1) % 10 == 0:
       gc.collect()  # Liberación periódica
   ```

**Retornos posibles**:
- `(df, "success")`: Datos extraídos correctamente
- `(df_vacío, "timeout")`: Excedió tiempo límite
- `(df_vacío, "error")`: Error durante procesamiento

---

### **8. Automatización Masiva**

#### `automatizar_extraccion(directorio_raiz, output_file, G=80e3)`

**Propósito**: Orquestar extracción de múltiples referencias con guardado progresivo

**Fases de ejecución**:

#### **Fase 1: Descubrimiento**
```python
carpetas_referencia = [p for p in glob.glob(patron_referencias) 
                      if os.path.isdir(p) and os.path.basename(p) not in 
                      ['venv', '__pycache__', '.git', 'PyAnsys', '.venv']]
```
- Excluye carpetas del sistema
- Cada carpeta = una referencia de resorte

#### **Fase 2: Validación y Clasificación**
```python
tamanio_mb = os.path.getsize(rst_file) / 1e6

if tamanio_mb > 700:
    # SALTAR - muy grande
    referencias_saltadas.append(...)
elif tamanio_mb > 400:
    timeout = 180  # 3 minutos
else:
    timeout = 60   # 1 minuto
```

**Estrategia de timeouts escalonados**:
- Archivos pequeños (<400 MB): 60s
- Archivos grandes (400-700 MB): 180s
- Archivos muy grandes (>700 MB): Saltados

#### **Fase 3: Procesamiento con Checkpoints**

```python
if status == "success" and not df_referencia.empty:
    dataframes_maestros.append(df_referencia)
    
    # CHECKPOINT cada 5 referencias
    if len(dataframes_maestros) % 5 == 0:
        df_temp = pd.concat(dataframes_maestros, ignore_index=True)
        backup_name = f"backup_{len(dataframes_maestros)}refs.csv"
        df_temp.to_csv(backup_name, index=False)
    
    # GUARDADO CONTINUO (sobrescribe)
    df_parcial = pd.concat(dataframes_maestros, ignore_index=True)
    df_parcial.to_csv(output_file, index=False)
```

**Beneficios**:
- Protección contra interrupciones (Ctrl+C)
- Recuperación desde backups
- Monitoreo de progreso en tiempo real

#### **Fase 4: Resumen Final**

```python
print(f"   Total carpetas:        {len(carpetas_referencia)}")
print(f"   ✅ Procesadas:          {len(referencias_procesadas)}")
print(f"   ⚠️  Saltadas/Errores:    {len(referencias_saltadas)}")
```

---

## 📊 Estructura del Dataset Final

### **Columnas del CSV**

| Columna | Tipo | Descripción | Unidad |
|---------|------|-------------|--------|
| `Referencia` | str | Identificador único del resorte | - |
| `d` | float | Diámetro del alambre | mm |
| `Dm` | float | Diámetro medio del resorte | mm |
| `C` | float | Índice de resorte (Dm/d) | - |
| `Na` | float | Espiras activas | - |
| `Nt` | float | Espiras totales | - |
| `p` | float | Paso del resorte | mm |
| `Lf` | float | Longitud libre | mm |
| `L` | float | Longitud comprimida | mm |
| `y` | float | Deflexión | mm |
| `F` | float | Fuerza aplicada | N |
| `Seqv_max` | float | Esfuerzo von Mises máximo | Pa |
| `k` | float | Rigidez (calculada: F/y) | N/mm |
| `step` | int | Índice del paso de carga | - |

### **Relaciones clave para ML**

**Variables independientes (features)**:
```
X = [d, Dm, C, Na, Nt, p, Lf, L, y]
```

**Variables dependientes (targets)**:
```
y₁ = F (fuerza)
y₂ = Seqv_max (esfuerzo)
y₃ = k (rigidez)
```

**Posibles tareas de ML**:
1. **Regresión**: Predecir `F` dado `y` y geometría
2. **Clasificación**: Predecir falla (Seqv > límite)
3. **Optimización**: Maximizar `k` minimizando `Seqv`

---

## ⚙️ Consideraciones Técnicas

### **Robustez y Manejo de Errores**

1. **Triple nivel de protección**:
   - Validación previa (tamaño, estructura)
   - Threading con timeout
   - Try-except en cada función crítica

2. **Gestión de memoria**:
   ```python
   gc.collect()  # Cada 10 steps
   thread = threading.Thread(..., daemon=True)  # Auto-limpieza
   ```

3. **Supresión de warnings**:
   ```python
   warnings.filterwarnings('ignore', category=RuntimeWarning)
   ```
   - Evita ruido en logs de operaciones masivas

### **Limitaciones Conocidas**

1. **Factor de corrección hardcodeado** (5.82):
   - Requiere calibración experimental
   - Puede no ser universal para todos los materiales

2. **Solo SYS-1**:
   - Ignora otros sistemas de Ansys Workbench
   - Decisión de diseño para simplificar

3. **Límite de 50 steps**:
   ```python
   max_steps = min(result.nsets, 50)
   ```
   - Previene procesamiento excesivo en archivos grandes

4. **Timeouts fijos**:
   - Pueden ser insuficientes en hardware lento
   - Podrían ser excesivos en hardware potente

### **Dependencias Críticas**

```python
import numpy as np              # Operaciones numéricas
import pandas as pd             # Manipulación de datos
from ansys.mapdl.reader import read_binary  # Parser de archivos RST
```

**Versiones recomendadas**:
- `ansys-mapdl-reader >= 0.52`
- `numpy >= 1.20`
- `pandas >= 1.3`

---

## 🚀 Uso Recomendado

### **Ejecución básica**:
```python
df = automatizar_extraccion(
    directorio_raiz="./mis_simulaciones",
    output_file="resortes_dataset.csv",
    G=80e3  # MPa - módulo de corte típico para acero
)
```

### **Estructura de carpetas esperada**:
```
directorio_raiz/
├── Resorte_001/
│   └── 3_SIMULACION/
│       └── Resorte_001_files/
│           └── dp0/SYS-1/MECH/file.rst
├── Resorte_002/
│   └── 3_SIMULACIÓN/  ← Maneja tilde
│       └── ...
└── Resorte_N/
```

### **Monitoreo en tiempo real**:
```
[  1] ✅ Resorte_001                           
   ⚙️  Procesando: Resorte_001
   ✅ 45 filas extraídas y limpiadas.
   💾 Backup guardado: backup_5refs.csv
```

---

## 🔬 Aplicaciones en Machine Learning

### **1. Modelos predictivos**
```python
# Ejemplo conceptual
from sklearn.ensemble import RandomForestRegressor

X = df[['d', 'Dm', 'Na', 'y']].values
y = df['F'].values

model = RandomForestRegressor()
model.fit(X, y)
```

### **2. Análisis de sensibilidad**
- Identificar qué parámetros geométricos más afectan a `Seqv`
- Optimizar diseños para minimizar peso manteniendo rigidez

### **3. Validación de modelos analíticos**
- Comparar predicciones FEA vs fórmulas clásicas (Wahl, etc.)
- Cuantificar errores en rangos operativos

---

## 📈 Diagrama de Flujo Completo

```
┌─────────────────────────────────────────────────────────────┐
│                    INICIO DEL SCRIPT                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Escaneo de carpetas en directorio_raiz                     │
│  - Excluye: venv, __pycache__, .git                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
           ┌─────────────────────────────┐
           │  Para cada carpeta          │
           └─────────────┬───────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  buscar_archivo_rst_sys1()                                  │
│  - Busca: 3_SIMULACION/*_files/dp0/SYS-1/MECH/file.rst     │
└────────────────────────┬────────────────────────────────────┘
                         │
                    ¿Encontrado?
                    /         \
                 NO/           \SI
                  /             \
                 ▼               ▼
        [Saltar carpeta]  ┌──────────────────────┐
                          │ validar_archivo_rst()│
                          └──────────┬───────────┘
                                     │
                              ¿Válido? (tamaño < 700MB)
                              /              \
                           NO/                \SI
                            /                  \
                           ▼                    ▼
                  [Saltar archivo]    ┌───────────────────────┐
                                      │ Asignar timeout:      │
                                      │ - <400MB: 60s         │
                                      │ - 400-700MB: 180s     │
                                      └──────────┬────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────┐
│  procesar_referencia_con_timeout()                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Thread Worker:                                        │  │
│  │ 1. read_binary(rst_file)                              │  │
│  │ 2. reconstruir_geometria(nodes)                       │  │
│  │    - detectar_eje()                                   │  │
│  │    - Calcular: Lf, d, Dm, Na, Nt, p                  │  │
│  │ 3. Crear ConfiguracionResorte                         │  │
│  │ 4. Para cada step (max 50):                           │  │
│  │    - obtener_deflexion_mejorada()                     │  │
│  │    - obtener_fuerza_global()                          │  │
│  │    - obtener_seqv_max()                               │  │
│  │    - Agregar fila al dataset                          │  │
│  │ 5. Limpiar NaN y calcular k = F/y                     │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────┘
                         │
                    ¿Timeout?
                    /         \
                 SI/           \NO
                  /             \
                 ▼               ▼
      [Retornar DataFrame   [Retornar DataFrame
       vacío, "timeout"]     con datos, "success"]
                 │               │
                 └───────┬───────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Agregación de resultados                                   │
│  - dataframes_maestros.append(df_referencia)                │
│  - Cada 5 refs: backup_Nrefs.csv                            │
│  - Sobrescribir: dataset_maestro_resortes.csv               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              ¿Más carpetas?
              /            \
           SI/              \NO
            /                \
           ▼                  ▼
    [Siguiente carpeta]  ┌───────────────────────┐
                         │ Consolidar dataset    │
                         │ final y mostrar       │
                         │ estadísticas          │
                         └───────────────────────┘
                                   │
                                   ▼
                             [FIN DEL SCRIPT]
```

---

## 🔍 Análisis Detallado de Algoritmos Clave

### **Reconstrucción de Espiras (Na)**

El algoritmo más sofisticado del script es la detección del número de espiras activas:

```python
# Paso 1: Coordenadas polares
theta = np.arctan2(R2, R1)  # Ángulo en [-π, π]

# Paso 2: Desenvolver discontinuidades 2π
theta_u = np.unwrap(theta)  # Convierte -π→π en -π→3π→5π...

# Paso 3: Regresión lineal ponderada
A_norm = A_c - A_c.mean()  # Centra en 0
m = np.sum(A_norm * theta_u) / np.sum(A_norm ** 2)

# m es la pendiente [rad/mm] de la hélice:
# Para resorte con 5 espiras en 100mm:
# m = (5 * 2π) / 100 = 0.314 rad/mm

# Paso 4: Extrapolar a longitud total
total_angle = m * (A.max() - A.min())  # Radianes totales
Na = abs(total_angle / (2 * pi))  # Convertir a vueltas completas
```

**Ventajas de este método**:
- ✅ No requiere conocer de antemano la orientación del resorte
- ✅ Robusto ante irregularidades en la malla
- ✅ Funciona con resortes de cualquier paso
- ✅ No depende de conteo manual de nodos

**Ejemplo numérico**:
```
Resorte con:
- 5 espiras activas
- Longitud: 100 mm
- Orientación: Eje Z

Proceso:
1. theta va de 0 a 10π (5 vueltas × 2π)
2. theta_u = [0, 0.1π, 0.2π, ..., 10π]
3. m ≈ 0.314 rad/mm
4. total_angle ≈ 31.4 rad
5. Na = 31.4 / (2π) ≈ 5.0 ✓
```

---

### **Detección Automática de Unidades**

El sistema inteligente de detección de unidades en `obtener_deflexion_mejorada()`:

```python
delta_raw = abs(mean_inf_raw - mean_sup_raw)

if delta_raw > Lf:
    factor = 1000.0    # μm → mm
elif delta_raw > 0.01:
    factor = 1.0       # Ya está en mm
else:
    factor = 0.001     # m → mm
```

**Lógica de decisión**:

| Condición | Interpretación | Acción |
|-----------|----------------|---------|
| `delta > Lf` | Deflexión > longitud libre (imposible físicamente) | Dividir entre 1000 |
| `0.01 < delta < Lf` | Rango razonable para mm | No ajustar |
| `delta < 0.01` | Valores muy pequeños (probablemente metros) | Multiplicar por 1000 |

**Ejemplo**:
```
Resorte: Lf = 50 mm, comprimido 10 mm

Caso A - Datos en μm:
  delta_raw = 10000 μm
  delta_raw > Lf → factor = 1000
  y = 10000 / 1000 = 10 mm ✓

Caso B - Datos en mm:
  delta_raw = 10 mm
  0.01 < delta_raw < Lf → factor = 1.0
  y = 10 mm ✓

Caso C - Datos en m:
  delta_raw = 0.01 m
  delta_raw < 0.01 → factor = 0.001
  y = 0.01 / 0.001 = 10 mm ✓
```

---

## 🧮 Fórmulas y Teoría de Resortes

### **Ecuaciones fundamentales utilizadas**

1. **Factor de Wahl (K_wahl)**:
   ```
   K = (4C - 1)/(4C - 4) + 0.615/C
   
   Donde:
   - C = Dm/d (índice de resorte)
   - Dm = diámetro medio
   - d = diámetro del alambre
   ```
   
   **Propósito**: Corregir la distribución no uniforme de esfuerzos en la sección del alambre debido a:
   - Esfuerzo cortante directo
   - Curvatura de la espira

2. **Rigidez teórica del resorte**:
   ```
   k = (G × d⁴) / (8 × Dm³ × Na)
   
   Donde:
   - G = módulo de corte [MPa]
   - d = diámetro del alambre [mm]
   - Dm = diámetro medio [mm]
   - Na = número de espiras activas
   ```

3. **Esfuerzo cortante máximo (teoría)**:
   ```
   τ_max = K × (8 × F × Dm) / (π × d³)
   
   Donde:
   - K = factor de Wahl
   - F = fuerza aplicada [N]
   ```

4. **Deflexión bajo carga**:
   ```
   y = (8 × F × Dm³ × Na) / (G × d⁴)
   ```

---

## 🎓 Calibración del Factor de Corrección (5.82)

### **Problema identificado**

Los resultados de esfuerzo de von Mises (σ_eq) del FEA son diferentes al esfuerzo cortante teórico (τ) debido a:

1. **Diferencia conceptual**:
   - von Mises: Criterio de fluencia multiaxial (√3×τ_oct)
   - Teoría: Esfuerzo cortante puro en la superficie

2. **Estado de esfuerzos real**:
   - El resorte tiene esfuerzos normales y cortantes combinados
   - La teoría simplificada solo considera cortante

### **Hipótesis del factor 5.82**

```python
seqv_corregido = seqv_FEA × 5.82
```

**Posibles justificaciones**:

1. **Relación von Mises - Cortante**:
   ```
   Para cortante puro: σ_eq = √3 × τ ≈ 1.732 × τ
   ```
   - Pero 5.82 ≈ 3.36 × 1.732, sugiere factor adicional

2. **Concentración de esfuerzos**:
   - La malla FEA captura gradientes locales
   - El percentil 99.9 subestima el pico real
   - Factor compensatorio ≈ 3-4×

3. **Conversión de unidades oculta**:
   - Si FEA retorna en unidades diferentes (GPa vs MPa)

### **⚠️ RECOMENDACIÓN CRÍTICA**

Este factor debe ser **validado experimentalmente**:

```python
# Procedimiento de calibración recomendado:

# 1. Seleccionar N resortes de referencia
resortes_test = ["REF_001", "REF_025", "REF_050"]

# 2. Obtener σ_eq_FEA del script
sigma_fea = df[df['Referencia'].isin(resortes_test)]['Seqv_max']

# 3. Calcular τ_teorico con fórmulas clásicas
tau_teorico = K_wahl * (8 * F * Dm) / (pi * d**3)

# 4. Determinar factor óptimo
factor_calibrado = tau_teorico / (sigma_fea / 1e6)

# 5. Validar en resortes diferentes
# Si factor_calibrado ≈ constante → usar promedio
# Si varía → depende de geometría (C, Na, etc.)
```

---

## 📊 Análisis Estadístico del Dataset

### **Métricas de calidad esperadas**

Una vez generado el dataset, validar:

1. **Consistencia geométrica**:
   ```python
   # Verificar índice de resorte
   df['C_calculado'] = df['Dm'] / df['d']
   assert np.allclose(df['C'], df['C_calculado'], rtol=0.01)
   
   # Rango típico: 4 < C < 12
   assert df['C'].between(3, 15).all()
   ```

2. **Consistencia física**:
   ```python
   # Rigidez positiva
   assert (df['k'] > 0).all()
   
   # Deflexión menor que longitud libre
   assert (df['y'] < df['Lf']).all()
   
   # Fuerza y deflexión proporcionales
   correlation = df.groupby('Referencia').apply(
       lambda x: x['F'].corr(x['y'])
   )
   assert (correlation > 0.95).mean() > 0.9  # 90% con R² > 0.95
   ```

3. **Cobertura de espacio de diseño**:
   ```python
   print("Rango de diámetros de alambre:")
   print(f"  d: [{df['d'].min():.2f}, {df['d'].max():.2f}] mm")
   
   print("Rango de diámetros medios:")
   print(f"  Dm: [{df['Dm'].min():.2f}, {df['Dm'].max():.2f}] mm")
   
   print("Rango de espiras activas:")
   print(f"  Na: [{df['Na'].min():.1f}, {df['Na'].max():.1f}]")
   ```

### **Gráficos de diagnóstico recomendados**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Curva Fuerza-Deflexión por referencia
fig, ax = plt.subplots(figsize=(12, 6))
for ref in df['Referencia'].unique()[:10]:  # Primeras 10
    data = df[df['Referencia'] == ref]
    ax.plot(data['y'], data['F'], marker='o', label=ref)
ax.set_xlabel('Deflexión y [mm]')
ax.set_ylabel('Fuerza F [N]')
ax.set_title('Curvas Características de Resortes')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)

# 2. Distribución de rigideces
fig, ax = plt.subplots(figsize=(10, 6))
df.groupby('Referencia')['k'].mean().hist(bins=50, ax=ax)
ax.set_xlabel('Rigidez k [N/mm]')
ax.set_ylabel('Frecuencia')
ax.set_title('Distribución de Rigideces en el Dataset')
ax.grid(True)

# 3. Mapa de calor: Geometría vs Rigidez
pivot = df.pivot_table(values='k', index='d', columns='Dm', aggfunc='mean')
sns.heatmap(pivot, cmap='viridis', cbar_kws={'label': 'k [N/mm]'})
plt.title('Rigidez en función de d y Dm')

# 4. Validación de linealidad Fuerza-Deflexión
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, ref in enumerate(df['Referencia'].unique()[:6]):
    ax = axes[i//3, i%3]
    data = df[df['Referencia'] == ref]
    ax.scatter(data['y'], data['F'], alpha=0.6)
    
    # Regresión lineal
    z = np.polyfit(data['y'], data['F'], 1)
    p = np.poly1d(z)
    ax.plot(data['y'], p(data['y']), "r--", linewidth=2)
    
    # R²
    residuals = data['F'] - p(data['y'])
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data['F'] - data['F'].mean())**2)
    r2 = 1 - (ss_res / ss_tot)
    
    ax.set_title(f'{ref}\nR² = {r2:.4f}')
    ax.set_xlabel('y [mm]')
    ax.set_ylabel('F [N]')
    ax.grid(True)
plt.tight_layout()
```

---

## 🚨 Casos de Error y Soluciones

### **Error 1: "Sin RST en SYS-1"**

**Causa**: El script solo busca en la ruta específica `SYS-1`

**Soluciones**:
```python
# Opción A: Modificar buscar_archivo_rst_sys1() para buscar en todos los sistemas
def buscar_archivo_rst_cualquier_sistema(carpeta_referencia):
    patron = "**/MECH/file.rst"
    archivos = list(Path(carpeta_referencia).glob(patron))
    return str(archivos[0]) if archivos else None

# Opción B: Especificar sistema manualmente
def buscar_archivo_rst_custom(carpeta_referencia, sistema="SYS-1"):
    # ... modificar línea:
    rst_path = carpeta_files / "dp0" / sistema / "MECH" / "file.rst"
```

---

### **Error 2: "Timeout después de Ns"**

**Causa**: Archivo demasiado complejo o hardware lento

**Soluciones**:
```python
# 1. Aumentar timeout globalmente
df = automatizar_extraccion(
    directorio_raiz="./",
    output_file="dataset.csv",
    timeout_grande=300,  # Añadir este parámetro
)

# 2. Reducir max_steps para archivos grandes
max_steps = min(result.nsets, 20)  # De 50 a 20

# 3. Procesar en modo reducido
def procesamiento_rapido(rst_file):
    # Solo extraer cada 5º step
    for step in range(0, result.nsets, 5):
        # ...
```

---

### **Error 3: "Archivo corrupto"**

**Causa**: Simulación no finalizó correctamente

**Diagnóstico**:
```python
# Verificar manualmente
from ansys.mapdl.reader import read_binary

result = read_binary("ruta/file.rst")
print(f"Número de sets: {result.nsets}")
print(f"Número de nodos: {result.grid.n_points}")
print(f"Tipos de resultados disponibles: {result.available_results}")

# Si result.nsets = 0 → archivo incompleto
# Si result.grid.n_points < 100 → malla muy tosca
```

---

### **Error 4: "Valores NaN en dataset"**

**Causa**: Alguna función de extracción falló para ciertos steps

**Análisis**:
```python
# Identificar qué variable causa NaN
df_problemas = df[df.isna().any(axis=1)]
print(df_problemas[['Referencia', 'step', 'L', 'y', 'F', 'Seqv_max']])

# Causas comunes:
# - L o y = NaN → Problema en obtener_deflexion_mejorada()
# - F = NaN → No hay fuerzas de reacción en ese step
# - Seqv_max = NaN → Problemas con tensor de esfuerzos

# Solución temporal: Rellenar con interpolación
df['F'].interpolate(method='linear', inplace=True)
```

---

## 🔧 Optimizaciones Avanzadas

### **1. Paralelización con Multiprocessing**

```python
from multiprocessing import Pool, cpu_count

def procesar_carpeta_wrapper(carpeta_full):
    """Wrapper para paralelización"""
    referencia = os.path.basename(carpeta_full)
    rst_file = buscar_archivo_rst_sys1(carpeta_full, referencia)
    
    if rst_file is None:
        return None
    
    df, status = procesar_referencia_con_timeout(rst_file, referencia)
    return df if status == "success" else None

def automatizar_extraccion_paralelo(directorio_raiz, n_workers=4):
    carpetas = glob.glob(os.path.join(directorio_raiz, '*'))
    carpetas = [c for c in carpetas if os.path.isdir(c)]
    
    with Pool(n_workers) as pool:
        resultados = pool.map(procesar_carpeta_wrapper, carpetas)
    
    dfs_validos = [df for df in resultados if df is not None]
    return pd.concat(dfs_validos, ignore_index=True)

# Uso
df = automatizar_extraccion_paralelo("./", n_workers=cpu_count()-1)
```

**Ganancia esperada**: 3-4× más rápido en CPUs modernos (8+ cores)

---

### **2. Caché de Geometrías**

```python
import pickle

def reconstruir_geometria_cached(nodes, cache_file="geom_cache.pkl"):
    """Guarda geometrías para no recalcular"""
    
    # Hash único basado en los nodos
    nodes_hash = hash(nodes.tobytes())
    
    # Intentar cargar desde caché
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        
        if nodes_hash in cache:
            return cache[nodes_hash]
    else:
        cache = {}
    
    # Calcular y guardar
    geom = reconstruir_geometria(nodes)
    cache[nodes_hash] = geom
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    
    return geom
```

---

### **3. Logging Estructurado**

```python
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'extraccion_{datetime.now():%Y%m%d_%H%M%S}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Uso en el código
logger.info(f"Iniciando procesamiento de {referencia}")
logger.warning(f"Archivo grande detectado: {tamanio_mb:.0f}MB")
logger.error(f"Error en {referencia}: {str(e)}")
```

---

## 📚 Referencias Bibliográficas

### **Teoría de Resortes**

1. **Wahl, A.M. (1963)**. *Mechanical Springs*. McGraw-Hill.
   - Fuente original del Factor de Wahl

2. **Shigley, J.E. & Mischke, C.R. (2001)**. *Mechanical Engineering Design*. 6th Ed.
   - Capítulo 10: Diseño de resortes helicoidales

3. **SAE Spring Committee (2008)**. *SAE HS-1582: Manual on Design and Application of Helical and Spiral Springs*.
   - Estándares industriales

### **Elementos Finitos**

4. **Budynas-Nisbett (2015)**. *Shigley's Mechanical Engineering Design*. 10th Ed.
   - Validación FEA vs teoría analítica

5. **Ansys Inc. (2023)**. *MAPDL Theory Reference*.
   - Formulación de elementos y criterios de convergencia

### **Machine Learning en Ingeniería**

6. **Schmidt, J. et al. (2019)**. "Recent advances and applications of machine learning in solid-state materials science". *npj Computational Materials*, 5(1), 83.

7. **Agrawal, A. & Choudhary, A. (2016)**. "Perspective: Materials informatics and big data: Realization of the 'fourth paradigm'". *APL Materials*, 4(5), 053208.

---

## 🎯 Checklist de Validación del Dataset

Antes de usar el dataset para ML, verificar:

- [ ] **Completitud**: Todas las referencias procesadas exitosamente
- [ ] **Consistencia geométrica**: C = Dm/d se cumple
- [ ] **Consistencia física**: k > 0, y < Lf, F > 0 cuando y > 0
- [ ] **Linealidad F-y**: R² > 0.95 para cada referencia
- [ ] **Sin outliers extremos**: Seqv dentro de 3σ del promedio
- [ ] **Rango de diseño**: Cobertura adecuada del espacio (d, Dm, Na)
- [ ] **Balance de clases**: Si se usa clasificación, clases equilibradas
- [ ] **Metadatos**: Documentar unidades, convenciones de signos
- [ ] **Trazabilidad**: Poder vincular cada fila a su archivo RST original
- [ ] **Reproducibilidad**: Ejecutar script 2 veces da mismo resultado

---

## 📝 Plantilla de Reporte de Dataset

```markdown
# Dataset: Resortes Helicoidales - FEA

## Información General
- **Fecha de generación**: YYYY-MM-DD
- **Script versión**: 1.0
- **Módulo de corte (G)**: 80,000 MPa (acero)
- **Total de referencias**: N
- **Total de muestras (filas)**: M
- **Referencias exitosas**: X
- **Referencias omitidas**: Y

## Estadísticas Descriptivas

### Parámetros Geométricos
| Variable | Min | Max | Media | Std | Unidad |
|----------|-----|-----|-------|-----|--------|
| d        | ... | ... | ...   | ... | mm     |
| Dm       | ... | ... | ...   | ... | mm     |
| C        | ... | ... | ...   | ... | -      |
| Na       | ... | ... | ...   | ... | -      |
| Lf       | ... | ... | ...   | ... | mm     |

### Variables de Respuesta
| Variable | Min | Max | Media | Std | Unidad |
|----------|-----|-----|-------|-----|--------|
| F        | ... | ... | ...   | ... | N      |
| y        | ... | ... | ...   | ... | mm     |
| k        | ... | ... | ...   | ... | N/mm   |
| Seqv_max | ... | ... | ...   | ... | MPa    |

## Correlaciones Principales
- F vs y: r = ...
- k vs C: r = ...
- Seqv vs F: r = ...

## Limitaciones Conocidas
1. Factor de corrección (5.82) no validado experimentalmente
2. Sólo sistema SYS-1 de Ansys Workbench
3. Máximo 50 steps por referencia

## Archivos Generados
- `dataset_maestro_resortes.csv` (principal)
- `backup_Nrefs.csv` (checkpoints)
- `extraccion_YYYYMMDD_HHMMSS.log`
```

---

## ✅ Conclusión Final

Este script representa un **sistema de producción robusto** para:

1. ✅ **Automatización completa** del proceso ETL desde simulaciones FEA
2. ✅ **Tolerancia a fallos** mediante validación, timeouts y checkpoints
3. ✅ **Escalabilidad** para procesar cientos de referencias
4. ✅ **Trazabilidad** con logs y estructura clara
5. ✅ **Preparación ML** con dataset limpio y estructurado

### **Próximos pasos recomendados**:

1. **Validación experimental** del factor 5.82
2. **Implementar paralelización** para >100 referencias
3. **Añadir visualización automática** de curvas F-y
4. **Extender a otros tipos de resortes** (compresión, torsión)
5. **Integrar con pipeline ML** (train/test split automático)

---

**Autor del Script**: [Tu nombre]  
**Fecha de Documentación**: Noviembre 2025  
**Versión**: 1.0  
**Licencia**: [Especificar]

---

## 📧 Soporte y Contribuciones

Para reportar issues, mejoras o preguntas:
- Email: tu_email@dominio.com
- GitHub: tu_usuario/repo
- Documentación adicional: [URL]

---

*Documentación generada con Claude (Anthropic)*