"""
Ejemplo: MLflow Model Registry con versiones y alias (champion / candidate)
usando el conjunto de datos Iris y dos RandomForestClassifier.
"""
import inspect

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# --- Configuración -----------------------------------------------------------
# El Model Registry necesita un backend con base de datos; SQLite basta en local.
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris_registry_demo")
client = MlflowClient()

MODEL_NAME = "iris_classifier"
SEED = 42

# Compatibilidad entre versiones de MLflow:
# - MLflow 2.x usa artifact_path="model"; MLflow 3.x usa name="model".
# - Las versiones recientes de 3.x serializan con skops y exigen declarar
#   como confiable el tipo interno del árbol de decisión.
_log_params = inspect.signature(mlflow.sklearn.log_model).parameters
LOG_MODEL_KWARGS = {"name": "model"} if "name" in _log_params else {"artifact_path": "model"}
if "skops_trusted_types" in _log_params:
    LOG_MODEL_KWARGS["skops_trusted_types"] = ["sklearn.tree._tree.Tree"]
print(f"MLflow {mlflow.__version__}")

# --- Datos --------------------------------------------------------------------
X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)


def train_and_log(max_depth: int) -> str:
    """Entrena un modelo en una ejecución independiente y devuelve su run_id."""
    with mlflow.start_run(run_name=f"rf_max_depth_{max_depth}") as run:
        params = {"n_estimators": 50, "max_depth": max_depth, "random_state": SEED}
        model = RandomForestClassifier(**params).fit(X_train, y_train)
        acc = accuracy_score(y_val, model.predict(X_val))

        mlflow.log_params(params)
        mlflow.log_metric("val_accuracy", acc)
        # Guardar el modelo como artefacto de la ejecución (todavía NO está registrado)
        mlflow.sklearn.log_model(
            model,
            signature=infer_signature(X_train, model.predict(X_train)),
            input_example=X_train.head(3),
            **LOG_MODEL_KWARGS,
        )
        print(f"Run {run.info.run_id[:8]}  max_depth={max_depth}  val_accuracy={acc:.4f}")
        return run.info.run_id


# --- 1. Dos ejecuciones independientes ---------------------------------------
run_1 = train_and_log(max_depth=1)
run_2 = train_and_log(max_depth=4)

# --- 2. Registrar ambos modelos bajo el mismo nombre -> versiones 1 y 2 -------
v1 = mlflow.register_model(f"runs:/{run_1}/model", MODEL_NAME).version
v2 = mlflow.register_model(f"runs:/{run_2}/model", MODEL_NAME).version
print(f"\nVersiones registradas de '{MODEL_NAME}': {v1}, {v2}")

# --- 3. Alias iniciales -------------------------------------------------------
client.set_registered_model_alias(MODEL_NAME, "champion", v1)
client.set_registered_model_alias(MODEL_NAME, "candidate", v2)


def alias_version(alias: str) -> str:
    return client.get_model_version_by_alias(MODEL_NAME, alias).version


def val_acc(version: str) -> float:
    run_id = client.get_model_version(MODEL_NAME, version).run_id
    return client.get_run(run_id).data.metrics["val_accuracy"]


# --- 4. Comparar y decidir la promoción ---------------------------------------
champ_v, cand_v = alias_version("champion"), alias_version("candidate")
champ_acc, cand_acc = val_acc(champ_v), val_acc(cand_v)
print(f"\nchampion  -> v{champ_v}  val_accuracy={champ_acc:.4f}")
print(f"candidate -> v{cand_v}  val_accuracy={cand_acc:.4f}")

if cand_acc > champ_acc:
    client.set_registered_model_alias(MODEL_NAME, "champion", cand_v)
    print(f"Decisión: la candidata (v{cand_v}) es mejor -> se promueve a champion.")
else:
    print(f"Decisión: empate o peor -> se conserva champion en v{champ_v}.")

# --- 5. Cargar por alias y predecir -------------------------------------------
URI = f"models:/{MODEL_NAME}@champion"
sample = X_val.head(5)

model = mlflow.pyfunc.load_model(URI)
print(f"\nCargado {URI} -> versión {alias_version('champion')}")
print("Predicciones:", model.predict(sample).tolist(), " reales:", y_val.head(5).tolist())

# --- 6. Rollback: reasignar champion a la otra versión, sin reentrenar --------
current = alias_version("champion")
other = v1 if current == v2 else v2
client.set_registered_model_alias(MODEL_NAME, "champion", other)
print(f"\nRollback: champion pasa de v{current} a v{other}")

# El objeto 'model' en memoria sigue siendo la versión anterior:
print("Objeto ya cargado (sin recargar):", model.predict(sample).tolist())

# Hay que volver a cargar con la MISMA URI para obtener la nueva versión:
model = mlflow.pyfunc.load_model(URI)
print(f"Recargado {URI} -> versión {alias_version('champion')}")
print("Predicciones:", model.predict(sample).tolist())
