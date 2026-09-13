"""
src/components/salary_predict/salary_model_registry.py

Salary Model Registry
=====================

Responsibilities
----------------
- Register an already validated MLflow model.
- Create a model version.
- Attach useful lineage and evaluation metadata.
- Optionally assign the production alias.
- Return a structured registration result.

This component does NOT:
    - train models
    - preprocess data
    - evaluate models
    - perform tuning
    - select model families
    - create predictions
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.configs.salary_model_registry_config import (
    SalaryModelRegistryConfig,
)

from src.entity.salary_model_registry_entity import (
    SalaryModelRegistryResult,
)

from src.logger import logging


class SalaryModelRegistry:
    """
    Registers validated salary models in MLflow Model Registry.
    """

    def __init__(
        self,
        config: Optional[SalaryModelRegistryConfig] = None,
    ) -> None:

        self.config = config or SalaryModelRegistryConfig()

        # ----------------------------------------------------------
        # Configure MLflow tracking URI
        # ----------------------------------------------------------

        if self.config.tracking_uri:

            mlflow.set_tracking_uri(self.config.tracking_uri)

        self.client = MlflowClient()

    # ==============================================================
    # PUBLIC API
    # ==============================================================

    def register(
        self,
        *,
        model_uri: str,
        source_run_id: Optional[str] = None,
        model_artifact_path: Optional[str] = None,
        validation_passed: bool = False,
        validation_metrics: Optional[Dict[str, float]] = None,
        test_metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        promote_to_production: bool = True,
    ) -> SalaryModelRegistryResult:
        """
        Register a validated MLflow model.

        Parameters
        ----------
        model_uri:
            MLflow model artifact URI.

        source_run_id:
            MLflow run that produced the model.

        model_artifact_path:
            Local model artifact path.

        validation_passed:
            Final validation quality-gate result.

        validation_metrics:
            Validation metrics.

        test_metrics:
            Final test metrics.

        metadata:
            Additional model metadata.

        promote_to_production:
            Whether the newly registered version should receive
            the configured production alias.
        """

        validation_metrics = validation_metrics or {}

        test_metrics = test_metrics or {}

        metadata = metadata or {}

        # ----------------------------------------------------------
        # Safety gate
        # ----------------------------------------------------------

        if not validation_passed:

            raise ValueError(
                "Model registration blocked because " "validation_passed=False."
            )

        if not model_uri:

            raise ValueError("model_uri must be provided.")

        # ----------------------------------------------------------
        # Track state as it becomes available so that a failure
        # partway through can still report what actually happened
        # (e.g. a version was created before an alias update failed).
        # ----------------------------------------------------------

        model_version: Optional[str] = None

        previous_production_version: Optional[str] = None

        try:

            logging.info("")
            logging.info("=" * 70)
            logging.info("SALARY MODEL REGISTRATION STARTED")
            logging.info("=" * 70)

            logging.info(
                "Registered model : %s",
                self.config.registered_model_name,
            )

            logging.info(
                "Model URI        : %s",
                model_uri,
            )

            logging.info(
                "Tracking URI     : %s",
                self.config.tracking_uri,
            )

            # ======================================================
            # 1. ENSURE REGISTERED MODEL EXISTS
            # ======================================================

            self._ensure_registered_model()

            # ======================================================
            # 2. DETERMINE CURRENT PRODUCTION VERSION
            #
            # Must happen BEFORE the new version is created and
            # BEFORE the alias is reassigned, so we can report which
            # version production is moving FROM.
            # ======================================================

            if promote_to_production:

                previous_production_version = self._get_current_alias_version(
                    self.config.production_alias
                )

                logging.info(
                    "Current '%s' alias -> version %s",
                    self.config.production_alias,
                    (
                        previous_production_version
                        if previous_production_version is not None
                        else "None (no existing alias)"
                    ),
                )

            # ======================================================
            # 3. CREATE MODEL VERSION
            #
            # Exactly one new version is created per register() call.
            # ======================================================

            version = self.client.create_model_version(
                name=(self.config.registered_model_name),
                source=model_uri,
                run_id=source_run_id,
            )

            model_version = str(version.version)

            logging.info(
                "Registered model version: %s",
                model_version,
            )

            # ======================================================
            # 4. ATTACH METADATA
            # ======================================================

            self._set_tags(
                model_version=model_version,
                source_run_id=source_run_id,
                validation_metrics=(validation_metrics),
                test_metrics=test_metrics,
                metadata=metadata,
            )

            # ======================================================
            # 5. PROMOTE TO PRODUCTION
            #
            # The model version above already exists in MLflow at
            # this point regardless of what happens next. If alias
            # assignment fails, we must NOT report overall success,
            # but we DO preserve model_version/model_uri so the
            # orphaned version can be found and fixed manually.
            # ======================================================

            alias_updated = False

            promoted_model_version: Optional[str] = None

            promotion_approved = False

            if promote_to_production:

                try:

                    self.client.set_registered_model_alias(
                        name=(self.config.registered_model_name),
                        alias=(self.config.production_alias),
                        version=model_version,
                    )

                    alias_updated = True

                    promoted_model_version = model_version

                    promotion_approved = True

                    logging.info(
                        "Production alias '%s' -> version %s "
                        "(previously: %s)",
                        self.config.production_alias,
                        model_version,
                        (
                            previous_production_version
                            if previous_production_version is not None
                            else "none"
                        ),
                    )

                except Exception as alias_exc:

                    logging.exception(
                        "Model version %s was created successfully, "
                        "but promoting it to the '%s' alias failed.",
                        model_version,
                        self.config.production_alias,
                    )

                    logging.info(
                        "SALARY MODEL REGISTRATION COMPLETED WITH ERRORS"
                    )

                    logging.info("=" * 70)

                    return SalaryModelRegistryResult(
                        success=False,
                        registered_model_name=(self.config.registered_model_name),
                        model_version=model_version,
                        model_uri=model_uri,
                        source_run_id=source_run_id,
                        production_alias=(self.config.production_alias),
                        alias_updated=False,
                        previous_production_version=(previous_production_version),
                        promoted_model_version=None,
                        promotion_approved=False,
                        model_artifact_path=(model_artifact_path),
                        validation_passed=True,
                        test_metrics=dict(test_metrics),
                        validation_metrics=dict(validation_metrics),
                        error=(
                            f"Model version {model_version} was created "
                            "successfully, but promoting it to the "
                            f"'{self.config.production_alias}' alias "
                            f"failed: {alias_exc}"
                        ),
                    )

            # ======================================================
            # 6. RESULT
            # ======================================================

            logging.info("SALARY MODEL REGISTRATION COMPLETED")

            logging.info("=" * 70)

            return SalaryModelRegistryResult(
                success=True,
                registered_model_name=(self.config.registered_model_name),
                model_version=model_version,
                model_uri=model_uri,
                source_run_id=source_run_id,
                production_alias=(
                    self.config.production_alias if promote_to_production else None
                ),
                alias_updated=alias_updated,
                previous_production_version=(previous_production_version),
                promoted_model_version=(promoted_model_version),
                promotion_approved=(promotion_approved),
                model_artifact_path=(model_artifact_path),
                validation_passed=True,
                test_metrics=dict(test_metrics),
                validation_metrics=dict(validation_metrics),
            )

        except Exception as exc:

            logging.exception("Salary model registration failed.")

            return SalaryModelRegistryResult(
                success=False,
                registered_model_name=(self.config.registered_model_name),
                model_version=model_version,
                model_uri=model_uri,
                source_run_id=source_run_id,
                previous_production_version=(previous_production_version),
                model_artifact_path=(model_artifact_path),
                validation_passed=(validation_passed),
                test_metrics=dict(test_metrics),
                validation_metrics=dict(validation_metrics),
                error=str(exc),
            )

    # ==============================================================
    # REGISTERED MODEL
    # ==============================================================

    def _ensure_registered_model(self) -> None:
        """
        Ensure that the registered model exists.

        If it already exists, reuse it.

        If it does not exist, create it.
        """

        name = self.config.registered_model_name

        try:

            self.client.get_registered_model(name)

            logging.info(
                "Registered model already exists: %s",
                name,
            )

        except MlflowException as exc:

            # ------------------------------------------------------
            # Model does not exist.
            # ------------------------------------------------------

            if not self.config.allow_existing_model:

                raise RuntimeError(
                    f"Registered model '{name}' "
                    "does not exist and "
                    "allow_existing_model=False."
                ) from exc

            self.client.create_registered_model(name=name)

            logging.info(
                "Created registered model: %s",
                name,
            )

    # ==============================================================
    # PRODUCTION ALIAS LOOKUP
    # ==============================================================

    def _get_current_alias_version(
        self,
        alias: str,
    ) -> Optional[str]:
        """
        Return the version currently holding ``alias``, or None if the
        registered model has no such alias yet (e.g. this is the very
        first registration).
        """

        name = self.config.registered_model_name

        try:

            model_version = self.client.get_model_version_by_alias(
                name,
                alias,
            )

            return str(model_version.version)

        except MlflowException:

            logging.info(
                "No existing '%s' alias found for model '%s'. "
                "This is expected for the first registration.",
                alias,
                name,
            )

            return None

    # ==============================================================
    # TAGS
    # ==============================================================

    def _set_tags(
        self,
        *,
        model_version: str,
        source_run_id: Optional[str],
        validation_metrics: Dict[str, float],
        test_metrics: Dict[str, float],
        metadata: Dict[str, Any],
    ) -> None:

        name = self.config.registered_model_name

        # ----------------------------------------------------------
        # Lineage
        # ----------------------------------------------------------

        if source_run_id:

            self.client.set_model_version_tag(
                name=name,
                version=model_version,
                key="source_run_id",
                value=str(source_run_id),
            )

        # ----------------------------------------------------------
        # Validation
        # ----------------------------------------------------------

        self.client.set_model_version_tag(
            name=name,
            version=model_version,
            key="validation_passed",
            value="true",
        )

        # ----------------------------------------------------------
        # Validation metrics
        # ----------------------------------------------------------

        for key, value in validation_metrics.items():

            self.client.set_model_version_tag(
                name=name,
                version=model_version,
                key=f"validation_{key}",
                value=str(value),
            )

        # ----------------------------------------------------------
        # Test metrics
        # ----------------------------------------------------------

        for key, value in test_metrics.items():

            self.client.set_model_version_tag(
                name=name,
                version=model_version,
                key=f"test_{key}",
                value=str(value),
            )

        # ----------------------------------------------------------
        # Additional metadata
        # ----------------------------------------------------------

        for key, value in metadata.items():

            self.client.set_model_version_tag(
                name=name,
                version=model_version,
                key=str(key),
                value=str(value),
            )