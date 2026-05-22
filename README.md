# Predicción de Mortalidad por Insuficiencia Cardíaca

## Proyecto Final - Inteligencia Artificial y Estadística
**Universidad de Sevilla · Curso 2025-26**

Las muertes por enfermedades al corazón lideran el puesto de mayor causa de mortalidad en este último año.
Ser capaz de adelantarnos a estas dolencias puede llegar a ser el factor decisivo entre la vida y la muerte.
Es por ello que se ha ideado esta app, que es capaz de expresar de manera numérica el riesgo de afección cardíaca según los valores clínicos de la persona.

Para poder conseguir este objetivo, se ha partido de una serie de datos de 299 pacientes con distintas analíticas y la variable booleana que indicaba si falleció o no por un ataque al corazón.
Con estos datos, se han tratado una serie de modelos para analizar cual predice mejor estos casos, por último, el mejor modelo se ha implementado en un predictor donde el usuario podrá introducir sus propios datos y comprobar su porcentaje de riesgo.

La estructura del proyecto es la siguiente:

📁 Trabajo_IA/
│
├── 📄 main.py                  Interfaz gráfica y flujo principal de Streamlit
├── 📄 pyproject.toml           Configuración del proyecto y dependencias modernas
├── 📄 uv.lock                  Bloqueo de versiones exactas para reproducibilidad
├── 📄 README.md                Este manual
│
├── 📁 data/
│   └── 📄 heart_failure_dataset.csv  Dataset utilizado
│
├── 📁 src/                     Núcleo lógico del sistema en Python
│   ├── 📄 data_processing.py   Pipeline de importación (Dask), filtrado y ordenación
│   ├── 📄 models.py            Entrenamiento, métricas y predicción de los modelos
│   ├── 📄 r_integration.py     Puente de comunicación por subprocess con R
│   └── 📄 visualization.py     Gráficos en Python
│
├── 📁 r_scripts/               Código en R
│   └── 📄 advanced_viz.R       Script de ggplot2 para generación de densidades y violines
│
└── 📁 r_outputs/               Directorio temporal de salida para los gráficos de R


Para configurar el entorno de forma local se deben seguir los siguientes pasos:

**Python 3.10 o superior instalada**

**uv instalado**

**R instalado**

Con estos prerrequisitos cumplidos, desde la carpeta principal del proyecto:

**Crear el entorno**: uv venv

**Instalar las dependencias**: uv sync

**Activar el entorno**: .venv/Scripts/activate

**Ejecutar la aplicación**: streamlit run main.py

Datos extra:

Si se observa el archivo de datos se observarán 300 filas, mientras que en la app solo se hace mención a 299 pacientes, esta fila "perdida" corresponde a la cabecera de los datos, ningún paciente ha sido eliminado ni ha habido ninguna observación incompleta.

Para garantizar la reproductividad, se ha utilizado la semilla random_state= 42.

Al tratar con 299 pacientes, el deep learning no cuenta con suficientes datos para ser eficaz, quedándose atrás con respecto a modelos de árbol.

Para reproducir de manera local los gráficos de R, será necesario tener las librerías correspondientes instaladas: install.packages(c("tidyverse", "gridExtra"))