"""
Tool for writing large agent payloads to disk so agents can reference them
without streaming thousands of tokens back through the LLM.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "output" / "temp" / "pending_payloads"


class SavePayloadInput(BaseModel):
    """Schema for persisting agent payloads for later retrieval."""

    label: str = Field(..., description="Short identifier for the payload.")
    payload: Dict[str, Any] = Field(..., description="JSON payload to persist.")
    scenario_id: Optional[str] = Field(default=None, description="Optional scenario identifier.")


def _save_payload(label: str, payload: Dict[str, Any], scenario_id: Optional[str] = None) -> Dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    safe_label = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label).strip("_") or "payload"
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")
    filename_parts = [part for part in (scenario_id, safe_label, timestamp) if part]
    filename = "_".join(filename_parts)
    file_path = OUTPUT_ROOT / f"{filename}.json"

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    size_bytes = file_path.stat().st_size
    logger.info(
        "Saved large payload '%s' for scenario '%s' to %s (%d bytes)",
        label,
        scenario_id or "N/A",
        file_path,
        size_bytes,
    )

    return {
        "success": True,
        "path": str(file_path),
        "label": label,
        "scenario_id": scenario_id,
        "size_bytes": size_bytes,
        "payload_keys": list(payload.keys()),
    }


def save_json_payload(label: str, payload: Dict[str, Any], scenario_id: Optional[str] = None) -> Dict[str, Any]:
    """Convenience wrapper for saving payloads outside of the tool interface."""
    return _save_payload(label=label, payload=payload, scenario_id=scenario_id)


save_json_payload_tool = StructuredTool(
    name="save_json_payload",
    description=(
        "Write a large JSON payload to disk and return the file path. "
        "Useful when an agent needs to share very large structured data without streaming it back "
        "directly in the final LLM response."
    ),
    func=_save_payload,
    args_schema=SavePayloadInput,
    return_direct=False,
)

