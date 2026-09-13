from __future__ import annotations
from mlflow.tracking import MlflowClient
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar
import shutil
from types import MappingProxyType
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from contextlib import contextmanager
from src.logger import logging
from src.exception import CustomException
from src.configs.salary_predict.salary_experiment_config import SalaryExperimentConfig
from src.configs.salary_predict.salary_ML_flow_config import SalaryMLflowConfig
from src.components.salary_predict.salary_single_experiment_runner import (
    SalaryTrainingResult,
)

T = TypeVar("T")


# ======================================================================
# RUN INFO
# ======================================================================


@dataclass(frozen=True)
class SalaryMLflowRunInfo:
    """Stable, minimal identifiers and timing for one MLflow run."""

    run_id: Optional[str]
    mlflow_experiment_id: Optional[str]
    artifact_uri: Optional[str]
    run_name: Optional[str]
    status: str  # "FINISHED" | "DISABLED"
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    duration_seconds: Optional[float] = None


# ======================================================================
# TRACKER
# ======================================================================


class SalaryMLflowTracker:

    def __init__(self, config: Optional[SalaryMLflowConfig] = None) -> None:
        self.config = config or SalaryMLflowConfig()
        # Cache git metadata once during initialization to avoid redundant subprocess calls on every run
        self._cached_git_commit = self._fetch_git_commit()
        self._cached_git_branch = self._fetch_git_branch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def track_training_result(
        self,
        experiment_config: SalaryExperimentConfig,
        training_result: SalaryTrainingResult,
        log_model: bool = False,
        register_model: bool = False,
    ) -> SalaryMLflowRunInfo:

        started_at_timestamp = time.time()
        started_at_perf = time.perf_counter()

        try:
            # 1. Type validation
            if not isinstance(experiment_config, SalaryExperimentConfig):
                raise TypeError(
                    f"Expected experiment_config to be SalaryExperimentConfig, "
                    f"got {type(experiment_config).__name__}"
                )
            if not isinstance(training_result, SalaryTrainingResult):
                raise TypeError(
                    f"Expected training_result to be SalaryTrainingResult, "
                    f"got {type(training_result).__name__}"
                )

            # 2. Experiment ID alignment guard
            if experiment_config.experiment_id != training_result.experiment_id:
                raise ValueError(
                    f"Experiment ID mismatch: experiment_config has "
                    f"'{experiment_config.experiment_id}', but training_result has "
                    f"'{training_result.experiment_id}'."
                )

            # 3. Disabled tracking handling
            if not self.config.is_tracking_enabled:
                logging.info(
                    f"MLflow tracking disabled; skipping run for "
                    f"experiment_id={experiment_config.experiment_id}."
                )
                ended_at_timestamp = time.time()
                ended_at_perf = time.perf_counter()
                return SalaryMLflowRunInfo(
                    run_id=None,
                    mlflow_experiment_id=None,
                    artifact_uri=None,
                    run_name=None,
                    status="DISABLED",
                    started_at=started_at_timestamp,
                    ended_at=ended_at_timestamp,
                    duration_seconds=ended_at_perf - started_at_perf,
                )

            self._ensure_tracking_uri()
            mlflow_experiment_id = self._resolve_experiment_id()
            run_name = self.config.build_run_name(experiment_config.experiment_id)
            tags = self._build_tags(experiment_config)
            params, overflow_features = self._build_params(experiment_config)
            metrics = self._build_metrics(training_result)

            logging.info("=" * 70)
            logging.info("MLFLOW TRACKING STARTED")
            logging.info("=" * 70)
            logging.info(f"Experiment ID    : {experiment_config.experiment_id}")
            logging.info(f"MLflow experiment: {self.config.experiment_name}")
            logging.info(f"Run name         : {run_name}")
            logging.info(f"Model family     : {experiment_config.model_name}")
            logging.info(f"Tracking URI     : {self.config.tracking_uri}")
            logging.info(
                "Prepared "
                f"{len(params)} parameters, "
                f"{len(metrics)} metrics, "
                f"{len(tags)} tags."
            )

            # Start MLflow run
            with mlflow.start_run(
                experiment_id=mlflow_experiment_id,
                run_name=run_name,
                tags=tags,
            ) as active_run:

                # Log params and metrics with targeted retries for network resilience
                self._execute_with_retry(lambda: mlflow.log_params(params))
                logging.info(f"Logged parameters : {len(params)}")

                self._execute_with_retry(lambda: mlflow.log_metrics(metrics))
                logging.info(f"Logged metrics    : {len(metrics)}")

                # Log artifacts (config, overflow features, model)
                self._log_config_artifact(experiment_config)

                if overflow_features:
                    self._log_overflow_features_artifact(overflow_features)

                if (
                    log_model or register_model
                ) and training_result.fitted_workflow is not None:
                    registered_name = (
                        f"{self.config.registered_model_name}_{experiment_config.model_name}"
                        if register_model
                        else None
                    )
                    self._log_model_artifact(
                        training_result.fitted_workflow,
                        registered_model_name=registered_name,
                    )

                ended_at_timestamp = time.time()
                ended_at_perf = time.perf_counter()

                run_info = SalaryMLflowRunInfo(
                    run_id=active_run.info.run_id,
                    mlflow_experiment_id=active_run.info.experiment_id,
                    artifact_uri=active_run.info.artifact_uri,
                    run_name=run_name,
                    status="FINISHED",
                    started_at=started_at_timestamp,
                    ended_at=ended_at_timestamp,
                    duration_seconds=ended_at_perf - started_at_perf,
                )

            logging.info(f"MLflow Run ID     : {run_info.run_id}")
            logging.info(f"Run Duration      : {run_info.duration_seconds:.2f}s")
            logging.info("=" * 70)
            logging.info("MLFLOW TRACKING COMPLETED")
            logging.info("=" * 70)

            return run_info

        except CustomException:
            raise
        except Exception as e:
            logging.error(
                f"MLflow tracking failed for experiment_id="
                f"{getattr(experiment_config, 'experiment_id', '?')}: {e}",
                exc_info=True,
            )
            raise CustomException(e, sys) from e

    # ------------------------------------------------------------------
    # Tracking infrastructure
    # ------------------------------------------------------------------
    def _ensure_tracking_uri(self) -> None:
        if (
            self.config.is_local_tracking
            and self.config.local_tracking_path is not None
        ):
            Path(self.config.local_tracking_path).mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(self.config.tracking_uri)

    def _resolve_experiment_id(self) -> str:
        client = MlflowClient(tracking_uri=self.config.tracking_uri)
        experiment = client.get_experiment_by_name(self.config.experiment_name)
        if experiment is not None:
            return experiment.experiment_id

        create_kwargs: Dict[str, Any] = {}
        if self.config.artifact_location is not None:
            create_kwargs["artifact_location"] = str(self.config.artifact_location)
        return client.create_experiment(self.config.experiment_name, **create_kwargs)

    # ------------------------------------------------------------------
    # Tags / params (Modularized & Cached)
    # ------------------------------------------------------------------
    def _build_tags(self, experiment_config: SalaryExperimentConfig) -> Dict[str, str]:
        tags: Dict[str, Any] = dict(self.config.default_tags())
        tags.update(
            {
                "experiment_id": experiment_config.experiment_id,
                "experiment_name": experiment_config.experiment_name,
                "model_family": experiment_config.model_name,
                "config_signature": experiment_config.config_signature,
                "evaluation_split": "validation",
                "python_version": sys.version.split()[0],
                "os_platform": platform.platform(),
                "hostname": socket.gethostname(),
                "git_commit": self._cached_git_commit,
                "git_branch": self._cached_git_branch,
            }
        )
        return {str(k): str(v) for k, v in tags.items() if v is not None}

    def _build_params(
        self, experiment_config: SalaryExperimentConfig
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        params: Dict[str, str] = {}
        params.update(self._build_model_params(experiment_config))

        feature_params, overflow_features = self._build_feature_params(
            experiment_config
        )
        params.update(feature_params)

        params.update(self._build_dataset_params(experiment_config))
        return params, overflow_features

    def _build_model_params(
        self, experiment_config: SalaryExperimentConfig
    ) -> Dict[str, str]:
        params: Dict[str, str] = {
            "experiment_id": experiment_config.experiment_id,
            "model.name": experiment_config.model_name,
        }
        for key, value in experiment_config.model_params.items():
            params[f"model.{key}"] = self._safe_param_value(value)
        return params

    def _build_feature_params(
        self, experiment_config: SalaryExperimentConfig
    ) -> tuple[Dict[str, str], Dict[str, Any]]:
        params: Dict[str, str] = {
            "use_title": self._safe_param_value(experiment_config.use_title),
            "skill_encoding": self._safe_param_value(experiment_config.skill_encoding),
        }
        overflow_features: Dict[str, Any] = {}
        max_len = self.config.max_parameter_length

        for feat_key, feat_val in [
            ("categorical_features", experiment_config.categorical_features),
            ("numeric_features", experiment_config.numeric_features),
        ]:
            if feat_val:
                str_val = self._safe_param_value(feat_val)
                if len(str_val) > max_len:
                    params[feat_key] = "logged_in_artifact_config/feature_lists.json"
                    overflow_features[feat_key] = feat_val
                else:
                    params[feat_key] = str_val

        return params, overflow_features

    @staticmethod
    def _build_dataset_params(
        experiment_config: SalaryExperimentConfig,
    ) -> Dict[str, str]:
        return {
            "training_target_col": experiment_config.training_target_col,
            "annual_target_col": experiment_config.annual_target_col,
        }

    @staticmethod
    def _safe_param_value(value: Any) -> str:
        if isinstance(value, Enum):
            return str(value.value)
        if value is None:
            return "none"
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, float, str)):
            return str(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return ",".join(SalaryMLflowTracker._safe_param_value(v) for v in value)
        return str(value)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _build_metrics(self, training_result: SalaryTrainingResult) -> Dict[str, float]:
        raw_metrics: Dict[str, Any] = {
            "validation.log_mae": training_result.log_metrics.get("mae"),
            "validation.log_rmse": training_result.log_metrics.get("rmse"),
            "validation.log_r2": training_result.log_metrics.get("r2"),
            "validation.annual_mae": training_result.annual_metrics.get("mae"),
            "validation.annual_rmse": training_result.annual_metrics.get("rmse"),
            "validation.annual_r2": training_result.annual_metrics.get("r2"),
            "validation.median_ape": training_result.annual_metrics.get("median_ape"),
            "timing.training_seconds": training_result.training_seconds,
            "timing.prediction_seconds": training_result.validation_prediction_seconds,
            "data.train_row_count": training_result.train_row_count,
            "data.validation_row_count": training_result.validation_row_count,
            "data.raw_feature_count": training_result.raw_feature_count,
        }
        if training_result.transformed_feature_count is not None:
            raw_metrics["data.transformed_feature_count"] = (
                training_result.transformed_feature_count
            )

        metrics: Dict[str, float] = {}
        for name, value in raw_metrics.items():
            if value is None:
                continue
            numeric_value = float(value)
            if not np.isfinite(numeric_value):
                logging.warning(
                    f"Skipping non-finite metric '{name}'={numeric_value!r}; not logged to MLflow."
                )
                continue
            metrics[name] = numeric_value
        return metrics

    # ------------------------------------------------------------------
    # Artifact Logging Helpers
    # ------------------------------------------------------------------
    def _log_config_artifact(self, experiment_config: SalaryExperimentConfig) -> None:
        """Dumps the full experiment configuration dataclass under config/ folder."""
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                config_path = Path(tmp_dir) / "experiment_config.json"
                config_dict = self._dataclass_to_dict(experiment_config)
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config_dict, f, indent=2, default=str)
                mlflow.log_artifact(str(config_path), artifact_path="config")
                logging.info(
                    "Logged experiment configuration to 'config/experiment_config.json'."
                )
        except Exception as e:
            logging.warning(f"Could not log config artifact (non-fatal): {e}")

    def _log_overflow_features_artifact(
        self, overflow_features: Dict[str, Any]
    ) -> None:
        """Dumps truncated long feature parameters as an artifact."""
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                feat_path = Path(tmp_dir) / "feature_lists.json"
                with open(feat_path, "w", encoding="utf-8") as f:
                    json.dump(overflow_features, f, indent=2, default=str)
                mlflow.log_artifact(str(feat_path), artifact_path="config")
                logging.info(
                    "Logged overflow feature lists to 'config/feature_lists.json'."
                )
        except Exception as e:
            logging.warning(f"Could not log feature list artifact (non-fatal): {e}")

    def _log_model_artifact(
        self, fitted_workflow: Any, registered_model_name: Optional[str] = None
    ) -> None:
        """Logs model under model/ artifact folder, with optional registry integration."""
        try:
            mlflow.sklearn.log_model(
                fitted_workflow,
                artifact_path="model",
                registered_model_name=registered_model_name,
            )
            logging.info(
                "Logged fitted pipeline as MLflow model artifact under 'model/'."
            )
            if registered_model_name:
                logging.info(f"Registered model as '{registered_model_name}'.")
        except Exception as e:
            logging.warning(f"Could not log model artifact (non-fatal): {e}")

    # ------------------------------------------------------------------
    # System Metadata & Targeted Retry Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _fetch_git_commit() -> Optional[str]:
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .decode("ascii")
                .strip()
            )
        except Exception:
            return None

    @staticmethod
    def _fetch_git_branch() -> Optional[str]:
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode("ascii")
                .strip()
            )
        except Exception:
            return None

    @staticmethod
    def _dataclass_to_dict(obj: Any) -> Any:
        if is_dataclass(obj):
            return {
                field.name: SalaryMLflowTracker._dataclass_to_dict(
                    getattr(obj, field.name)
                )
                for field in obj.__dataclass_fields__.values()
            }

        if isinstance(obj, Enum):
            return obj.value

        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, MappingProxyType):
            return dict(obj)

        if isinstance(obj, (list, tuple)):
            return [SalaryMLflowTracker._dataclass_to_dict(v) for v in obj]

        if isinstance(obj, dict):
            return {
                k: SalaryMLflowTracker._dataclass_to_dict(v) for k, v in obj.items()
            }

        return obj

    @staticmethod
    def _execute_with_retry(
        func: Callable[[], T], max_retries: int = 3, delay: float = 2.0
    ) -> T:
        """Retries only transient network/storage errors (e.g. MlflowException, IOError, ConnectionError)."""
        last_exception = None
        for attempt in range(1, max_retries + 1):
            try:
                return func()
            except (MlflowException, IOError, ConnectionError, TimeoutError) as e:
                last_exception = e
                if attempt < max_retries:
                    logging.warning(
                        f"MLflow transient error (attempt {attempt}/{max_retries}): {e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    logging.error(
                        f"MLflow operation failed after {max_retries} attempts."
                    )
            except (TypeError, ValueError) as e:
                # Instantly raise deterministic coding/typing errors without retrying
                raise e
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry operation failed without recording exception.")

    # ------------------------------------------------------------------
    # Generic Artifact Logging API
    # ------------------------------------------------------------------

    def log_artifact(
        self, artifact_path: str | Path, artifact_folder: str | None = None
    ) -> None:

        artifact_path = Path(artifact_path)

        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found: {artifact_path}")

        active_run = mlflow.active_run()

        if active_run is None:
            logging.warning(
                "No active MLflow run. Skipping artifact logging "
                f"for '{artifact_path.name}'."
            )
            return

        self._execute_with_retry(
            lambda: mlflow.log_artifact(
                local_path=str(artifact_path),
                artifact_path=artifact_folder,
            )
        )

        logging.info(
            f"Logged artifact '{artifact_path.name}' " f"to '{artifact_folder or '/'}'."
        )

    def log_artifacts(
        self,
        directory: str | Path,
        artifact_folder: str | None = None,
    ) -> None:

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        active_run = mlflow.active_run()

        if active_run is None:
            logging.warning(
                "No active MLflow run. "
                f"Skipping directory logging for '{directory.name}'."
            )
            return

        self._execute_with_retry(
            lambda: mlflow.log_artifacts(
                local_dir=str(directory),
                artifact_path=artifact_folder,
            )
        )

        logging.info(f"Logged artifact directory '{directory.name}'.")

    def log_dict(
        self,
        data: Dict[str, Any],
        filename: str,
        artifact_folder: str | None = None,
    ) -> None:

        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / filename

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(
                    data,
                    f,
                    indent=2,
                    default=str,
                )

            self.log_artifact(
                file_path,
                artifact_folder,
            )

    def log_text(
        self,
        text: str,
        filename: str,
        artifact_folder: str | None = None,
    ) -> None:

        with tempfile.TemporaryDirectory() as tmp:

            file_path = Path(tmp) / filename

            file_path.write_text(
                text,
                encoding="utf-8",
            )

            self.log_artifact(
                file_path,
                artifact_folder,
            )

    def log_directory_copy(
        self,
        source_dir: str | Path,
        artifact_folder: str | None = None,
    ) -> None:

        source_dir = Path(source_dir)

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / source_dir.name

            shutil.copytree(
                source_dir,
                destination,
            )

            self.log_artifacts(
                destination,
                artifact_folder,
            )

    @contextmanager
    def start_run(self, run_name: str):

        if not self.config.is_tracking_enabled:

            logging.info("MLflow tracking disabled. " "No MLflow run will be started.")

            yield None
            return

        self._ensure_tracking_uri()

        experiment = self._get_or_create_experiment()

        with mlflow.start_run(
            experiment_id=experiment.experiment_id,
            run_name=run_name,
        ) as run:

            mlflow.set_tags(self.config.default_tags())

            logging.info(
                "Started MLflow run '%s'",
                run.info.run_id,
            )

            try:
                yield run

            finally:

                logging.info(
                    "Finished MLflow run '%s'",
                    run.info.run_id,
                )

    def _get_or_create_experiment(self):
        client = MlflowClient(tracking_uri=self.config.tracking_uri)

        experiment = client.get_experiment_by_name(self.config.experiment_name)

        if experiment is None:
            experiment_id = client.create_experiment(self.config.experiment_name)
            experiment = client.get_experiment(experiment_id)

        return experiment

    def log_tags(self, tags: Dict[str, Any]) -> None:

        active_run = mlflow.active_run()

        if active_run is None:
            logging.warning("No active MLflow run. Skipping tag logging.")
            return

        cleaned = {str(k): str(v) for k, v in tags.items() if v is not None}

        self._execute_with_retry(lambda: mlflow.set_tags(cleaned))

        logging.info("Logged %d MLflow tags.", len(cleaned))

    def log_metrics(self, metrics: Dict[str, Any]) -> None:

        active_run = mlflow.active_run()

        if active_run is None:
            logging.warning("No active MLflow run. Skipping metric logging.")
            return

        cleaned_metrics: Dict[str, float] = {}

        for key, value in metrics.items():

            if value is None:
                continue

            numeric_value = float(value)

            if not np.isfinite(numeric_value):
                continue

            cleaned_metrics[str(key)] = numeric_value

        if not cleaned_metrics:
            logging.warning("No valid metrics to log.")
            return

        self._execute_with_retry(lambda: mlflow.log_metrics(cleaned_metrics))

        logging.info(
            "Logged %d MLflow metrics.",
            len(cleaned_metrics),
        )

    def log_params(self, params: Dict[str, Any]) -> None:
        active_run = mlflow.active_run()

        if active_run is None:
            logging.warning("No active MLflow run. Skipping parameter logging.")
            return

        cleaned = {
            str(k): self._safe_param_value(v)
            for k, v in params.items()
            if v is not None
        }

        if not cleaned:
            logging.warning("No valid parameters to log.")
            return

        self._execute_with_retry(lambda: mlflow.log_params(cleaned))

        logging.info("Logged %d MLflow parameters.", len(cleaned))

    def log_final_model(
        self,
        fitted_workflow: Any,
        registered_model_name: Optional[str] = None,
        register_model: bool = True,
    ) -> Optional[str]:
        """
        Log the final sklearn Pipeline as an MLflow model artifact.

        Parameters
        ----------
        fitted_workflow:
            The fitted preprocessor+model sklearn Pipeline.

        registered_model_name:
            Name to register under, if ``register_model`` is True.
            Falls back to ``self.config.registered_model_name``.

        register_model:
            When True (default, preserves prior behavior), passes
            ``registered_model_name`` to ``mlflow.sklearn.log_model``,
            which causes MLflow to auto-create a new Model Registry
            version as a side effect of logging.

            When False, the model is logged as a plain MLflow model
            artifact (still governed by the same run) WITHOUT creating
            any Model Registry version. Use this when a separate
            component (e.g. SalaryModelRegistry) is responsible for
            explicitly creating the registered version and assigning
            aliases, to avoid creating two versions for one logged
            model.

        Returns
        -------
        The MLflow model URI (e.g. ``runs:/<run_id>/final_model``),
        or None if tracking is disabled.
        """

        if fitted_workflow is None:
            raise ValueError("fitted_workflow must not be None.")

        if not self.config.is_tracking_enabled:
            logging.info(
                "MLflow tracking disabled. Final model will not be registered."
            )
            return None

        model_name = registered_model_name or self.config.registered_model_name

        if not model_name or not model_name.strip():
            raise ValueError("registered_model_name must not be empty.")

        try:

            logging.info("Logging final sklearn Pipeline to MLflow.")

            log_model_kwargs: Dict[str, Any] = {
                "sk_model": fitted_workflow,
                "name": "final_model",
                "skops_trusted_types": [
                    "numpy.dtype",
                    "src.components.salary_predict.salary_preprocessor_builder.SafeCategoricalTransformer",
                    "src.components.salary_predict.salary_preprocessor_builder.SafeTextTransformer",
                ],
            }

            if register_model:

                logging.info("Registering model as: %s", model_name)

                log_model_kwargs["registered_model_name"] = model_name

            else:

                logging.info(
                    "Logging model WITHOUT auto-registration "
                    "(register_model=False). A separate registry "
                    "component is expected to create the version."
                )

            model_info = mlflow.sklearn.log_model(**log_model_kwargs)

            model_uri = model_info.model_uri

            logging.info("Final model logged successfully.")

            logging.info("MLflow model URI: %s", model_uri)

            if register_model:

                logging.info(
                    "Model registered successfully: %s",
                    model_name,
                )

            return model_uri

        except Exception as e:

            logging.error(
                "Failed to log/register final model: %s",
                e,
                exc_info=True,
            )

            raise

    def get_latest_model_version(
        self,
        registered_model_name: Optional[str] = None,
    ) -> Optional[str]:

        if not self.config.is_tracking_enabled:
            return None

        model_name = registered_model_name or self.config.registered_model_name

        if not model_name or not model_name.strip():
            raise ValueError("registered_model_name must not be empty.")

        try:

            client = MlflowClient(tracking_uri=self.config.tracking_uri)

            versions = client.search_model_versions(f"name='{model_name}'")

            if not versions:
                logging.warning(
                    "No registered versions found for model '%s'.",
                    model_name,
                )
                return None

            latest_version = max(versions, key=lambda item: int(item.version))

            logging.info(
                "Latest registered model version | " "name=%s | version=%s",
                model_name,
                latest_version.version,
            )

            return str(latest_version.version)

        except Exception as e:

            logging.error(
                "Failed to retrieve latest model version " "for '%s': %s",
                model_name,
                e,
                exc_info=True,
            )

            raise