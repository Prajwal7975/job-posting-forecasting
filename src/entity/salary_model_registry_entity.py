from __future__ import annotations

import json

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class SalaryModelRegistryResult:

    # ==============================================================
    # RESULT
    # ==============================================================

    success: bool

    # ==============================================================
    # MODEL REGISTRY
    # ==============================================================

    registered_model_name: str

    model_version: Optional[str] = None

    model_uri: Optional[str] = None

    source_run_id: Optional[str] = None

    # ==============================================================
    # PRODUCTION
    # ==============================================================

    production_alias: Optional[str] = None

    alias_updated: bool = False

    # Version that was serving production BEFORE this registration.
    previous_production_version: Optional[str] = None

    # Version that is now serving production.
    promoted_model_version: Optional[str] = None

    # Whether the candidate passed the promotion decision.
    promotion_approved: bool = False

    # ==============================================================
    # MODEL ARTIFACT
    # ==============================================================

    model_artifact_path: Optional[str] = None

    # ==============================================================
    # VALIDATION / TEST
    # ==============================================================

    validation_passed: bool = False

    validation_metrics: Dict[str, float] = field(
        default_factory=dict
    )

    test_metrics: Dict[str, float] = field(
        default_factory=dict
    )

    # ==============================================================
    # ERROR
    # ==============================================================

    error: Optional[str] = None

    # ==============================================================
    # TIMESTAMP
    # ==============================================================

    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ==============================================================
    # SERIALIZATION
    # ==============================================================

    def to_dict(self) -> Dict[str, Any]:

        return {
            "success": self.success,

            "registered_model_name": self.registered_model_name,

            "model_version": self.model_version,

            "model_uri": self.model_uri,

            "source_run_id": self.source_run_id,

            "production_alias": self.production_alias,

            "alias_updated": self.alias_updated,

            "previous_production_version": (
                self.previous_production_version
            ),

            "promoted_model_version": (
                self.promoted_model_version
            ),

            "promotion_approved": (
                self.promotion_approved
            ),

            "model_artifact_path": (
                self.model_artifact_path
            ),

            "validation_passed": (
                self.validation_passed
            ),

            "validation_metrics": dict(
                self.validation_metrics
            ),

            "test_metrics": dict(
                self.test_metrics
            ),

            "error": self.error,

            "generated_at": self.generated_at,
        }

    def to_json(self, indent: int = 2) -> str:

        return json.dumps(
            self.to_dict(),
            indent=indent,
            default=str,
        )