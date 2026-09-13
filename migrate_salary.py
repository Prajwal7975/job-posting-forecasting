from __future__ import annotations

import mlflow
import mlflow.sklearn

OLD_TRACKING_URI = "sqlite:///mlflow.db"

NEW_TRACKING_URI = "http://127.0.0.1:5000"

REGISTERED_MODEL_NAME = "salary_prediction_model"
MODEL_ALIAS = "production"


# ==========================================================
# STEP 1: Load existing production model
# ==========================================================

print("Loading existing production model...")

mlflow.set_tracking_uri(OLD_TRACKING_URI)

model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}")

print("Existing model loaded successfully.")
print(f"Model type: {type(model)}")


# ==========================================================
# STEP 2: Switch to new Dockerized MLflow
# ==========================================================

print()
print("Connecting to new MLflow server...")

mlflow.set_tracking_uri(NEW_TRACKING_URI)

print(f"Tracking URI: {mlflow.get_tracking_uri()}")


# ==========================================================
# STEP 3: Log model into new MLflow
# ==========================================================

print()
print("Logging model into new MLflow server...")

with mlflow.start_run(run_name="salary-model-migration"):
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="salary_model",
        registered_model_name=REGISTERED_MODEL_NAME,
        skops_trusted_types=[
            "numpy.dtype",
            "src.components.salary_predict.salary_preprocessor_builder.SafeCategoricalTransformer",
            "src.components.salary_predict.salary_preprocessor_builder.SafeTextTransformer",
        ],
    )

    print(f"Model logged successfully: " f"{model_info.model_uri}")


# ==========================================================
# STEP 4: Find newly created model version
# ==========================================================

client = mlflow.MlflowClient(tracking_uri=NEW_TRACKING_URI)

versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")

if not versions:
    raise RuntimeError("No model version was created.")

latest_version = max(
    versions,
    key=lambda version: int(version.version),
)

print()
print(f"New model version: {latest_version.version}")


# ==========================================================
# STEP 5: Assign production alias
# ==========================================================

client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias=MODEL_ALIAS,
    version=latest_version.version,
)

print()
print(f"Alias '{MODEL_ALIAS}' assigned to " f"version {latest_version.version}.")

print()
print("========================================")
print("MODEL MIGRATION SUCCESSFUL")
print("========================================")
