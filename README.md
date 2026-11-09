# Taller 2 de MLOps — puntos 1 a 9

Esta carpeta contiene el notebook completo con explicaciones y ejemplos de los nueve puntos, además de los archivos fuente independientes de los puntos 8 y 9.

## Contenido

- `Taller_2_MLOps_Completo.ipynb`: entrega integral, con teoría y ejemplos de Linux, Bash, Git, Optuna y MLflow.
- `Punto9_iris_registry.py`: script del punto 9 para registrar dos modelos Iris con MLflow, comparar versiones, asignar alias y hacer rollback.
- `Punto9_MLOps_respuestas.docx`: explicación, resultados y respuestas del punto 9.
- `requirements.txt`: dependencias principales para ejecutar los ejemplos Python.

## Configuración recomendada

Usar Python 3.10 o 3.11 en un entorno virtual. Desde esta carpeta, crear y activar el entorno e instalar dependencias:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

En Jupyter local, abrir el notebook desde esta carpeta. Las celdas de instalación incluyen Optuna y MLflow. Las bases SQLite y los artefactos quedarán en el directorio de trabajo.

## Punto 8 — Optuna y SQLite

El notebook usa la base SQLite local `optuna_cancer.db`, creada en el directorio de trabajo. Optuna crea y administra las tablas al crear el estudio con `storage=STORAGE`. Las consultas SQL del notebook inspeccionan esas tablas. Mantener el mismo archivo SQLite y el mismo `STUDY_NAME` permite reabrir el estudio y continuar añadiendo trials.

## Punto 9 — MLflow Model Registry

El script configura un backend SQLite con `sqlite:///mlflow.db`; el archivo se crea en el directorio de trabajo. Los artefactos del modelo también se guardan localmente. Ejecutar el script desde esta carpeta para mantener los archivos generados aquí:

```bash
python Punto9_iris_registry.py
```

El script incluye compatibilidad de logging para MLflow 2.x y 3.x. El `requirements.txt` fija MLflow en 2.19.0, la versión 2.x con la que se documentó la ejecución. Para usar otra versión compatible de MLflow, cambiar esa línea en `requirements.txt` y volver a instalar las dependencias.

## Notas

- Los comandos de Linux/Bash y los ejemplos de Git están escritos para ejecutarse en Linux, macOS, WSL o Google Colab; PowerShell no implementa todos los mismos comandos.
- El punto 6 explica cómo crear la rama y abrir el Pull Request. El PR real debe abrirse desde el repositorio del compañero.
- Las bases de datos (`optuna_cancer.db`, `mlflow.db`) y el directorio `mlruns/` son archivos generados al ejecutar los ejemplos; no se incluyen como entregables iniciales.

