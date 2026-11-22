"""Pydantic v2 data contracts for clinical cases, LLM outputs, and audit scores."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Specialty = Literal["neurorehabilitation", "respiratory", "neurosurgery_icu"]

# Four-tier inpatient clinical acuity scale aligned with escalation workflows.
TriageLevel = Literal["emergent", "urgent", "semi_urgent", "routine"]


class ClinicalCase(BaseModel):
    """Structured synthetic clinical note with clinician-verified ground truth."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    case_id: str = Field(..., min_length=1)
    specialty: Specialty
    clinical_note: str = Field(..., min_length=1)
    ground_truth_red_flags: list[str] = Field(..., min_length=1)
    ground_truth_differentials: list[str] = Field(..., min_length=1)


class LLMSummaryOutput(BaseModel):
    """Structured clinical summary produced by an evaluated LLM."""

    model_config = ConfigDict(str_strip_whitespace=True)

    case_id: str = Field(..., min_length=1)
    summary: str = Field(..., min_length=1)
    suggested_differentials: list[str] = Field(..., min_length=1)
    # Empty list permitted because an audited model may fail to detect any red flags.
    detected_red_flags: list[str] = Field(default_factory=list)
    triage_level: TriageLevel


class AuditScore(BaseModel):
    """Rubric-based 1-to-5 audit evaluation of an LLM clinical summary."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    case_id: str = Field(..., min_length=1)
    factual_accuracy: int = Field(..., ge=1, le=5)
    critical_omission: int = Field(..., ge=1, le=5)
    clinical_safety: int = Field(..., ge=1, le=5)
    guideline_alignment: int = Field(..., ge=1, le=5)
    hallucination_detected: bool
    auditor_notes: str = Field(..., min_length=1)
