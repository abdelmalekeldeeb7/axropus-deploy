"""NemoClaw Integration — NVIDIA NemoClaw stack on Axropus.

NemoClaw = Nemotron models + OpenClaw + NVIDIA OpenShell
Axropus adds AMF optimization on top of the NemoClaw stack.

Stack: NemoClaw (Nemotron + OpenShell) → Axropus (AMF + Dynamo) → GPU
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class NemoClawIntegration:
    """Detect and configure NemoClaw stack with AMF optimization."""

    def __init__(self) -> None:
        self._nemoclaw_detected = False
        self._nemotron_model_id: Optional[str] = None
        self._openshell_enabled = False

    def detect(self) -> bool:
        """Detect if NemoClaw stack is installed."""
        # Check for NemoClaw environment markers
        self._nemoclaw_detected = (
            os.environ.get("NEMOCLAW_ENABLED", "").lower() in ("1", "true") or
            os.path.exists("/opt/nvidia/nemoclaw") or
            os.environ.get("NVIDIA_OPENSHELL", "") != ""
        )

        if self._nemoclaw_detected:
            self._nemotron_model_id = os.environ.get(
                "NEMOCLAW_MODEL", "nvidia/Nemotron-70B"
            )
            self._openshell_enabled = (
                os.environ.get("NVIDIA_OPENSHELL", "").lower() in ("1", "true")
            )
            logger.info(
                "[NEMOCLAW] Detected: model=%s, openshell=%s",
                self._nemotron_model_id, self._openshell_enabled,
            )

        return self._nemoclaw_detected

    @property
    def is_available(self) -> bool:
        return self._nemoclaw_detected

    def get_amf_config(self) -> Dict[str, Any]:
        """Get AMF configuration optimized for Nemotron models."""
        return {
            "quant_mode": "int4",
            "vram_pool_gb": 50,
            "pin_system_prompt": True,
            "prefix_sharing": True,
            "nemotron_optimized": True,
            # Nemotron uses GQA with 8 KV heads — same as Llama 70B
            # AMF block layout is compatible
        }

    def get_openshell_config(self) -> Dict[str, Any]:
        """Get OpenShell security/privacy configuration."""
        if not self._openshell_enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "privacy_mode": os.environ.get("OPENSHELL_PRIVACY", "standard"),
            "data_retention": os.environ.get("OPENSHELL_RETENTION", "none"),
            "audit_logging": True,
        }

    def create_nemoclaw_claw_config(
        self,
        name: str,
        system_prompt: str,
        tools: list,
        channels: list,
    ) -> Dict[str, Any]:
        """Create a claw configuration optimized for NemoClaw stack."""
        return {
            "name": name,
            "model_id": self._nemotron_model_id or "nvidia/Nemotron-70B",
            "system_prompt": system_prompt,
            "tools": tools,
            "channels": channels,
            "openclaw_config": {
                "skills": tools,
                "heartbeat_interval_s": 60,
                "max_steps_per_task": 20,
                "nemoclaw_mode": True,
            },
            "amf_config": self.get_amf_config(),
            "openshell_config": self.get_openshell_config(),
            "tags": ["nemoclaw", "nvidia"],
        }
