from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os

CONFIG_VERSION = "1.0"


@dataclass(frozen=True)
class SalaryModelRegistryConfig:
    """
    Configuration contract for registering and promoting
    a validated salary model in MLflow Model Registry.
    """

    config_version: str = CONFIG_VERSION

    # --------------------------------------------------------------
    # Registry
    # --------------------------------------------------------------

    registered_model_name: str = "salary_prediction_model"

    # --------------------------------------------------------------
    # MLflow
    # --------------------------------------------------------------

    tracking_uri: Optional[str] = None

    # --------------------------------------------------------------
    # Promotion
    # --------------------------------------------------------------

    # This alias is stable.
    #
    # The pipeline decides WHICH model version receives this alias.
    #
    production_alias: str = "production"

    # --------------------------------------------------------------
    # Behavior
    # --------------------------------------------------------------

    allow_existing_model: bool = True
    environment: str = "local"

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "SalaryModelRegistryConfig":
        return cls(
            registered_model_name=os.getenv(
                "SALARY_REGISTERED_MODEL_NAME",
                "salary_prediction_model",
            ),
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
            production_alias=os.getenv(
                "SALARY_MODEL_ALIAS",
                "production",
            ),
            allow_existing_model=True,
            environment=os.getenv("ENVIRONMENT", "local"),
        )

    def __post_init__(self) -> None:

        if (
            not isinstance(
                self.registered_model_name,
                str,
            )
            or not self.registered_model_name.strip()
        ):
            raise ValueError("registered_model_name must be a non-empty string.")

        if self.tracking_uri is not None:

            if (
                not isinstance(
                    self.tracking_uri,
                    str,
                )
                or not self.tracking_uri.strip()
            ):
                raise ValueError("tracking_uri must be None or a non-empty string.")

        if (
            not isinstance(
                self.production_alias,
                str,
            )
            or not self.production_alias.strip()
        ):
            raise ValueError("production_alias must be a non-empty string.")

        if not isinstance(
            self.allow_existing_model,
            bool,
        ):
            raise ValueError("allow_existing_model must be a bool.")

        if not isinstance(
            self.environment,
            str,
        ):
            raise ValueError("environment must be a string.")
