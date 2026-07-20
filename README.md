# P30 — Análisis cinemático lateral de marcha con DeepLabCut

Pipeline reproducible en Python para el análisis de marcha murina en vista lateral a partir de archivos de seguimiento generados por **DeepLabCut**. El flujo incluye preprocesamiento, detección de ciclos, goniometría, variables temporales, selección de los 10 ciclos con mayor máximo angular por articulación, análisis grupal y organización final de resultados.

## Flujo de trabajo

```text
DeepLabCut (.h5 / .csv)
        ↓
01. Preprocesamiento y detección de ciclos
        ↓
02. Goniometría lateral + selección Top 10
        ↓
03. Variables temporales y toe clearance
        ↓
04. Validación y consolidación
        ↓
05. Gráficos goniométricos Top 10
        ↓
06. Análisis grupal WT vs SOD1
        ↓
07. Organización final de resultados P30
```

## Scripts

### `01_preprocesamiento_y_ciclos.py`
Lee los archivos de DeepLabCut, filtra coordenadas según `likelihood`, interpola gaps cortos, suaviza las señales y detecta eventos repetitivos de marcha mediante un enfoque multiseñal.

Los ciclos se construyen únicamente entre eventos consecutivos detectados. No se insertan ciclos sintéticos.

Principales salidas:

- coordenadas limpias;
- eventos detectados;
- ciclos de marcha;
- ciclos normalizados;
- señales de detección;
- candidatos rechazados;
- gráfico de control de ciclos.

Ejemplo:

```bash
python 01_preprocesamiento_y_ciclos.py archivo_DLC.h5 --outdir salida_01
```

---

### `02_goniometria_lateral_por_ciclos_TOP10.py`
Calcula la goniometría lateral usando los ciclos definidos por el script 01.

Articulaciones analizadas:

- cadera: `crest-hip-knee`;
- rodilla: `hip-knee-ankle`;
- tobillo: `knee-ankle-foot`;
- pie: `ankle-foot-toe`.

Conserva la **goniometría completa de todos los ciclos válidos** y genera adicionalmente una selección **Top 10 independiente por articulación**, correspondiente a los 10 ciclos con mayor ángulo máximo para cada articulación.

Principales salidas:

- ángulos por frame;
- perfiles angulares normalizados por ciclo;
- rangos angulares por ciclo;
- resúmenes de rango angular;
- tablas y perfiles Top 10;
- gráficos de control.

Ejemplo:

```bash
python 02_goniometria_lateral_por_ciclos_TOP10.py \
    salida_01/archivo_clean_coords.csv \
    --cycles salida_01/archivo_gait_cycles.csv \
    --outdir salida_02 \
    --top-n 10
```

---

### `03_variables_temporales_y_toe_clearance.py`
Calcula variables temporales a partir de los ciclos validados:

- duración de zancada;
- toe-off estimado;
- duración de apoyo;
- duración de oscilación;
- porcentaje de apoyo;
- porcentaje de oscilación;
- toe clearance.

Las variables dependientes de toe-off utilizan su propio control de calidad. Un ciclo puede seguir siendo válido para goniometría aunque no sea aceptado para el análisis temporal.

> **Nota:** el toe clearance se expresa en píxeles mientras no exista una calibración espacial en mm/píxel.

Ejemplo:

```bash
python 03_variables_temporales_y_toe_clearance.py \
    salida_01/archivo_clean_coords.csv \
    --cycles salida_01/archivo_gait_cycles.csv \
    --fps 60 \
    --outdir salida_03
```

---

### `04_validacion_estadistica_y_excel.py`
Integra las salidas de goniometría y variables temporales, conserva las banderas de control de calidad y genera un libro Excel de auditoría y resumen.

Principios principales:

- no excluye animales automáticamente;
- no reemplaza datos reales;
- conserva valores originales y controles de calidad;
- considera al **animal como unidad estadística principal**, no al ciclo.

Ejemplo:

```bash
python 04_validacion_estadistica_y_excel.py \
    --input-dir resultados_pipeline \
    --out validacion_estadistica_dlc.xlsx
```

---

### `05_graficos_goniometria_top10.py`
Genera gráficos goniométricos individuales y grupales utilizando la selección Top 10.

Para cada articulación, la selección se realiza de forma independiente:

- 10 ciclos con mayor máximo de cadera;
- 10 ciclos con mayor máximo de rodilla;
- 10 ciclos con mayor máximo de tobillo;
- 10 ciclos con mayor máximo de pie.

Para los gráficos grupales, primero se obtiene el perfil promedio de cada animal y luego se calcula el promedio del grupo. De esta forma, un animal con mayor número de ciclos no recibe un peso desproporcionado.

Ejemplo:

```bash
python 05_graficos_goniometria_top10.py \
    --input-dir resultados_pipeline \
    --groups-csv groups.csv \
    --outdir resultados_05_goniometria \
    --top-n 10
```

El archivo de grupos debe tener el formato:

```csv
animal_id,group
856,SOD1
857,SOD1
860,WT
```

Se incluye `groups_template.csv` como plantilla.

---

### `06_analisis_grupal_WT_vs_SOD1.py`
Consolida los resultados por genotipo y genera la comparación grupal entre **WT** y **SOD1**.

Incluye:

- resúmenes por animal;
- perfiles goniométricos grupales;
- comparación WT vs SOD1;
- variables temporales;
- control de calidad;
- estadística exploratoria a nivel de animal.

Este script no utiliza la metodología estadística del código histórico `groupLateral.py`: no genera datos aleatorios ni reemplaza mediciones reales antes del análisis.

---

### `07_organizar_resultados_P30.py`
Organiza automáticamente las salidas técnicas del pipeline en una estructura final legible y ordenada.

Estructura esperada:

```text
P30/
├── Individual/
│   ├── 856/
│   │   ├── Ciclos/
│   │   ├── Goniometría/
│   │   │   ├── Completa/
│   │   │   └── Top 10/
│   │   ├── Variables temporales/
│   │   └── Resumen/
│   └── ...
│
└── Grupal comparativo/
    ├── General/
    ├── WT/
    ├── SOD1/
    └── WT vs SOD1/
```

## Dependencias

Python 3.10 o superior recomendado.

Paquetes principales:

```bash
pip install numpy pandas scipy matplotlib openpyxl tables
```

`tables` es necesario para leer archivos `.h5` con `pandas.read_hdf()`.

## Datos de entrada

Los archivos de DeepLabCut deben contener los siguientes puntos anatómicos:

```text
crest
hip
knee
ankle
foot
toe
```

con coordenadas:

```text
x
y
likelihood
```

## Parámetros principales del análisis P30

- frecuencia de adquisición: **60 Hz**;
- umbral mínimo de likelihood: **0.70**;
- suavizado principal: **5 frames**;
- normalización del ciclo: **101 puntos (0–100 %)**;
- selección goniométrica: **Top 10 ciclos por articulación según ángulo máximo**.

Los parámetros pueden modificarse mediante los argumentos de cada script cuando corresponda.

## Consideraciones metodológicas

### Goniometría Top 10

La selección Top 10 es adicional a la goniometría completa. Los datos originales de todos los ciclos válidos se conservan para trazabilidad.

### Variables temporales

El inicio de ciclo y el toe-off son estimaciones cinemáticas derivadas del seguimiento por video. No equivalen a una medición directa mediante plataforma de fuerza.

### Unidad experimental

En los análisis grupales, el animal debe considerarse la unidad experimental principal. Los múltiples ciclos de un mismo animal son medidas repetidas y no réplicas biológicas independientes.

### Estadística

El pipeline no genera datos sintéticos para realizar pruebas estadísticas. Las comparaciones deben realizarse sobre los datos observados y respetando la estructura experimental.

## Cohorte P30 utilizada durante el desarrollo

En el conjunto utilizado para validar este pipeline:

- **SOD1:** 856, 857, 859, 862, 864, 865
- **WT:** 860, 861, 863, 867

La asignación de grupos debe mantenerse en un archivo externo (`groups.csv`) cuando el pipeline se reutilice con nuevas cohortes.

## Reproducibilidad

Se recomienda conservar siempre:

1. los archivos H5 originales de DeepLabCut;
2. los parámetros utilizados en cada ejecución;
3. las salidas completas de cada etapa;
4. las tablas de ciclos seleccionados para Top 10;
5. la asignación animal-grupo utilizada en el análisis grupal.

Esto permite reconstruir el análisis desde los datos originales hasta las figuras y tablas finales.
