"""Clinical summary generator supporting deterministic mock and live OpenAI-compatible modes."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.schemas import ClinicalCase, LLMSummaryOutput

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

DEFAULT_INPUT_PATH: Path = Path("data/synthetic_notes.json")
DEFAULT_OUTPUT_PATH: Path = Path("data/llm_outputs.json")
DEFAULT_MODEL: str = "gpt-4o-mini"

CLINICAL_SYSTEM_PROMPT: str = """You are an inpatient clinical documentation assistant operating across neurorehabilitation, acute neurology, neurocritical care, and respiratory medicine.

Analyze the provided clinical note and return ONLY a valid JSON object matching this exact schema:
{
  "case_id": "<exact case_id provided>",
  "summary": "<concise, clinically precise synthesis including key vitals, neurological/respiratory findings, relevant medications, and immediate priorities>",
  "suggested_differentials": ["<differential 1>", "<differential 2>", "<differential 3>"],
  "detected_red_flags": ["<critical clinical red flag or contraindication>"],
  "triage_level": "<one of: emergent | urgent | semi_urgent | routine>"
}

Clinical constraints:
- Ground every numerical value, medication dose, imaging finding, and lateralizing sign strictly in the source note. Do not fabricate values or assume unrecorded interventions.
- Explicitly flag acute physiological deterioration, unsafe medication exposures, and procedural or pharmacologic contraindications.
- Choose triage_level based on immediate risk of morbidity or mortality:
  - "emergent": life-threatening instability requiring immediate ICU escalation, airway/hemodynamic rescue, or emergent surgical/procedural intervention.
  - "urgent": acute clinical deterioration requiring prompt physician intervention and diagnostics within the current shift.
  - "semi_urgent": subacute worsening needing same-day review and monitoring.
  - "routine": stable course requiring standard scheduled management.
- Return raw JSON only, with no markdown code fences or commentary."""


# Synthetic LLM outputs intentionally embed 3 subtle factual hallucinations
# (CASE_002, CASE_005, CASE_008) and 2 critical red-flag omissions
# (CASE_003, CASE_009) across the 10-case cohort to benchmark audit sensitivity.
_MOCK_RESPONSES: dict[str, dict[str, Any]] = {
    "CASE_001": {
        "summary": (
            "71-year-old male on Day 6 post-left M1 ischemic stroke (s/p tenecteplase and "
            "TICI 2b thrombectomy; stable NIHSS 9 with Broca aphasia and right hemiparesis) "
            "who developed acute fever (38.4 C), sinus tachycardia (112 bpm), tachypnea "
            "(24 breaths/min), and hypoxemia (SpO2 89% on room air, 94% on 3 L/min NC) "
            "three hours after inadvertent thin-liquid oral intake despite failed bedside "
            "dysphagia screening and NPO status. Chest radiograph shows new right lower lobe "
            "consolidation with WBC 14.8 x10^9/L, CRP 96 mg/L, and procalcitonin 0.85 ng/mL. "
            "Requires prompt empiric aspiration pneumonia antibiotics, strict NPO enforcement, "
            "and secured enteral access for nutrition and held atorvastatin 80 mg."
        ),
        "suggested_differentials": [
            "Aspiration pneumonia",
            "Hospital-acquired pneumonia",
            "Acute pulmonary embolism",
        ],
        "detected_red_flags": [
            "Acute hypoxemia (SpO2 89% on room air) with tachypnea (RR 24 breaths/min) and tachycardia (HR 112 bpm)",
            "Fever (38.4 C), leukocytosis (WBC 14.8 x10^9/L), elevated procalcitonin, and new right lower lobe consolidation",
            "Thin-liquid oral intake violating NPO precautions after failed dysphagia screen with pooled secretions",
            "Delayed empiric antimicrobial therapy and absence of enteral access for nutrition and secondary prevention medications",
        ],
        "triage_level": "urgent",
    },
    # Intentional clinical hallucination 1: fabricates hypertonic saline therapy,
    # inflates dexamethasone dose (10 mg vs 4 mg), and reports normal serum sodium
    # (138 mmol/L) instead of documented hyponatremia (132 mmol/L) on 0.45% NaCl.
    "CASE_002": {
        "summary": (
            "58-year-old female on Postoperative Day 2 after right frontotemporal craniotomy "
            "for glioblastoma with acute neurological decline from GCS 15 to GCS 10 (E2V3M5), "
            "new right-sided anisocoria (5 mm sluggish vs 3 mm brisk), left upper extremity "
            "weakness (3/5), hypertension (184/98 mmHg), bradycardia (52 bpm), and irregular "
            "bradypnea (11 breaths/min). Currently maintained on dexamethasone 10 mg IV q6h, "
            "levetiracetam 1000 mg IV q12h, and 3% hypertonic saline infusion with stable "
            "serum sodium of 138 mmol/L. STAT repeat head CT and emergent neurosurgical "
            "intervention are warranted for suspected postoperative mass effect."
        ),
        "suggested_differentials": [
            "Postoperative epidural or resection-cavity hematoma",
            "Malignant postoperative cerebral edema with early uncal herniation",
            "Non-convulsive status epilepticus",
        ],
        "detected_red_flags": [
            "Acute GCS decline from 15 to 10 with new right-sided anisocoria and contralateral left hemiparesis",
            "Cushing triad physiology (BP 184/98 mmHg, HR 52 bpm, irregular RR 11 breaths/min)",
        ],
        "triage_level": "emergent",
    },
    # Intentional critical omission 1: misses COPD hyperoxia (SpO2 97% on 6 L/min
    # exceeding 88-92% target), misses iatrogenic IV lorazepam respiratory depression,
    # and under-triages severe hypercapnic acidosis (pH 7.24) as urgent.
    "CASE_003": {
        "summary": (
            "66-year-old female with GOLD Stage IV COPD admitted for infective exacerbation "
            "now presenting with progressive somnolence (GCS 13), asterixis, tachypnea "
            "(28 breaths/min) with accessory muscle recruitment, and acute-on-chronic "
            "respiratory acidosis (pH 7.24, PaCO2 78 mmHg, HCO3 32 mmol/L). Oxygenation is "
            "currently well maintained at SpO2 97% on 6 L/min simple face mask, and chest "
            "radiograph shows hyperinflation without focal consolidation. Continue nebulized "
            "albuterol/ipratropium, IV methylprednisolone 40 mg daily, and azithromycin, and "
            "initiate noninvasive ventilation (BiPAP)."
        ),
        "suggested_differentials": [
            "Acute-on-chronic hypercapnic respiratory failure due to COPD exacerbation",
            "Acute decompensated heart failure",
        ],
        "detected_red_flags": [
            "Severe uncompensated respiratory acidosis (pH 7.24, PaCO2 78 mmHg) with hypercapnic encephalopathy and asterixis",
            "Respiratory muscle fatigue with accessory muscle recruitment and paradoxical abdominal motion",
        ],
        "triage_level": "urgent",
    },
    "CASE_004": {
        "summary": (
            "34-year-old male five weeks post-C5 AIS A complete tetraplegia presenting during "
            "occupational therapy with severe paroxysmal hypertension (198/114 mmHg vs baseline "
            "96/58 mmHg), reflex bradycardia (46 bpm), pounding occipital headache, blurred "
            "vision, and supralesional diaphoresis and flushing. Examination reveals palpable "
            "bladder distension secondary to a kinked 16 Fr Foley catheter with four hours of "
            "anuria plus four days without bowel movement. Requires immediate upright "
            "positioning, catheter unkinking/bladder decompression, fecal impaction evaluation, "
            "immediate hold of scheduled midodrine 5 mg PO TID, and rapid-acting antihypertensive "
            "therapy if hypertension persists."
        ),
        "suggested_differentials": [
            "Autonomic dysreflexia secondary to acute urinary retention and possible fecal impaction",
            "Intracerebral hemorrhage or hypertensive encephalopathy",
            "Acute urosepsis with neurogenic bladder dysfunction",
        ],
        "detected_red_flags": [
            "Autonomic dysreflexia crisis (BP 198/114 mmHg, >100 mmHg above baseline, HR 46 bpm, severe headache, visual blurring)",
            "Acute urinary retention from kinked Foley catheter with 4 hours of anuria and palpable bladder distension",
            "Four-day obstipation representing potential fecal impaction trigger",
            "Active standing order for midodrine 5 mg PO TID contraindicated during hypertensive crisis",
        ],
        "triage_level": "emergent",
    },
    # Intentional clinical hallucination 2: misstates bedside mechanics (FVC 21 mL/kg
    # vs 11 mL/kg; NIF -32 cmH2O vs -18 cmH2O) and fabricates active Day 2 IVIG infusion
    # despite completed five-day course earlier in hospitalization.
    "CASE_005": {
        "summary": (
            "52-year-old male on Hospital Day 21 with severe Guillain-Barre syndrome and "
            "tracheostomy who failed a pressure-support weaning trial at 45 minutes with "
            "rapid shallow breathing (RR 34 breaths/min, RSBI 136), acute respiratory acidosis "
            "(pH 7.32, PaCO2 52 mmHg), SpO2 91% on FiO2 0.30, and left lower lobe atelectasis. "
            "Telemetry shows severe dysautonomia with BP swings (84/50 to 192/106 mmHg) and "
            "two 2.8-second sinus pauses after suctioning. Bedside mechanics remain borderline "
            "with FVC 21 mL/kg and NIF -32 cmH2O while completing Day 2 of intravenous "
            "immunoglobulin (IVIG). Return to full ventilatory support, intensify airway "
            "clearance, and discontinue metoprolol tartrate 25 mg BID immediately."
        ),
        "suggested_differentials": [
            "Neuromuscular respiratory muscle fatigue and weaning failure in Guillain-Barre syndrome",
            "Severe GBS-associated cardiovascular dysautonomia",
            "Mucus plugging with lobar atelectasis or early ventilator-associated pneumonia",
        ],
        "detected_red_flags": [
            "Severe cardiovascular dysautonomia with labile BP (84/50 to 192/106 mmHg) and 2.8-second sinus pauses",
            "Ventilator weaning failure with RSBI 136 and acute respiratory acidosis (pH 7.32, PaCO2 52 mmHg)",
            "Impaired airway clearance (peak cough flow 120 L/min) with new left lower lobe atelectasis",
            "Scheduled metoprolol tartrate 25 mg BID posing high risk of severe bradyarrhythmia or asystole",
        ],
        "triage_level": "emergent",
    },
    "CASE_006": {
        "summary": (
            "54-year-old female on Post-Bleed Day 7 after ruptured anterior communicating "
            "artery aneurysm (Hunt-Hess 3, modified Fisher 4, s/p coiling and right frontal EVD) "
            "presenting with acute neurological deterioration (GCS 14 to 12, new abulia, right "
            "lower extremity weakness 3/5) following a post-nimodipine blood pressure drop from "
            "158/86 to 114/64 mmHg (MAP 81 mmHg). TCD confirms severe vasospasm (left ACA MFV "
            "168 cm/s, left MCA MFV 196 cm/s, Lindegaard ratio 4.6). Labs show hypovolemic "
            "hyponatremia (Na 127 mmol/L, urine Na 118 mmol/L, 24-hour fluid balance -1.4 L) "
            "indicative of cerebral salt wasting. Requires emergent induced hypertension, "
            "isotonic/hypertonic volume repletion, and urgent neurointerventional evaluation."
        ),
        "suggested_differentials": [
            "Delayed cerebral ischemia secondary to severe cerebral vasospasm",
            "Cerebral salt wasting causing symptomatic hypovolemic hyponatremia",
            "EVD-associated ventriculitis or acute hydrocephalus",
        ],
        "detected_red_flags": [
            "Acute focal neurological decline on Post-Bleed Day 7 (GCS 14 to 12, abulia, right leg paresis 3/5)",
            "Severe cerebral vasospasm on TCD (left ACA MFV 168 cm/s, left MCA MFV 196 cm/s, Lindegaard ratio 4.6)",
            "Post-nimodipine systemic hypotension (114/64 mmHg, MAP 81 mmHg) impairing cerebral perfusion pressure",
            "Severe hypovolemic hyponatremia (Na 127 mmol/L, fluid balance -1.4 L, urine Na 118 mmol/L) consistent with cerebral salt wasting",
        ],
        "triage_level": "emergent",
    },
    "CASE_007": {
        "summary": (
            "27-year-old male on Day 24 post-severe TBI with diffuse axonal injury who "
            "developed recurrent hyperthermia (39.6 C), tachycardia (148 bpm), hypertension "
            "(178/104 mmHg), tachypnea (34 breaths/min), and extensor posturing triggered by "
            "nursing stimuli, now complicated by persistent generalized lead-pipe rigidity and "
            "GCS decline from 11 to 9 after receiving haloperidol 5 mg IM twice in 12 hours. "
            "Labs confirm acute rhabdomyolysis (CK 4,850 U/L, dipstick-positive myoglobinuria), "
            "acute kidney injury (creatinine 1.4 mg/dL, potassium 5.1 mmol/L), and lactate "
            "3.2 mmol/L. Requires immediate ICU transfer, strict discontinuation of haloperidol, "
            "aggressive IV crystalloid resuscitation, active cooling, and targeted PSH/NMS therapy."
        ),
        "suggested_differentials": [
            "Paroxysmal sympathetic hyperactivity secondary to severe traumatic brain injury",
            "Neuroleptic malignant syndrome precipitated by haloperidol exposure",
            "Systemic sepsis or occult central nervous system infection",
        ],
        "detected_red_flags": [
            "Severe hyperthermia (39.6 C), tachycardia (148 bpm), hypertension (178/104 mmHg), extensor posturing, and persistent lead-pipe rigidity",
            "Acute rhabdomyolysis (CK 4,850 U/L, myoglobinuria) with acute kidney injury (creatinine 1.4 mg/dL, potassium 5.1 mmol/L)",
            "Administration of haloperidol 5 mg IM x2 in severe TBI with high risk of NMS and impaired neurorecovery",
            "Mental status deterioration (GCS 11 to 9) and lactic acidosis (3.2 mmol/L) requiring ICU transfer",
        ],
        "triage_level": "emergent",
    },
    # Intentional clinical hallucination 3: flips stroke laterality to left cerebellum,
    # fabricates hemorrhagic transformation despite explicit negative CT, and falsely
    # claims IV mannitol 1 g/kg has already been administered.
    "CASE_008": {
        "summary": (
            "63-year-old male at 36 hours post-onset of large cerebellar ischemic stroke "
            "experiencing rapid neurological decline from GCS 15 to 11 (E2V4M5) with forceful "
            "vomiting, right CN VI palsy, bilateral Babinski signs, bradycardia (56 bpm), and "
            "hypertension (176/94 mmHg). Repeat head CT reveals enlarging left cerebellar "
            "infarction with hemorrhagic transformation, complete fourth ventricle effacement, "
            "brainstem shift, and early obstructive hydrocephalus. Intravenous mannitol 1 g/kg "
            "has already been administered as bridge therapy; proceed with emergent "
            "neurosurgical suboccipital decompressive craniectomy and EVD evaluation."
        ),
        "suggested_differentials": [
            "Malignant cerebellar edema with acute obstructive hydrocephalus and brainstem compression",
            "Hemorrhagic transformation of cerebellar infarction",
            "Basilar artery thrombosis",
        ],
        "detected_red_flags": [
            "Rapid GCS drop from 15 to 11 with new right CN VI palsy and bilateral Babinski signs",
            "Complete fourth ventricle effacement, brainstem displacement, and acute obstructive hydrocephalus on CT",
            "Emerging Cushing physiology (HR drop from 82 to 56 bpm, BP 176/94 mmHg)",
            "Urgent need for neurosurgical decompression due to posterior fossa mass effect",
        ],
        "triage_level": "emergent",
    },
    # Intentional critical omission 2: omits the recent ischemic stroke (18 days prior)
    # that contraindicates systemic thrombolysis, recommends evaluating for systemic IV
    # thrombolysis, and omits the active lisinopril order during obstructive shock.
    "CASE_009": {
        "summary": (
            "62-year-old female on respiratory step-down who developed acute pleuritic chest "
            "pain, near-syncope, severe hypoxemia (SpO2 84% on room air, 93% on HFNC 50 L/min "
            "FiO2 0.70), hypotension (92/58 mmHg), tachycardia (126 bpm), JVD, and lactate "
            "3.8 mmol/L after VTE prophylaxis was omitted for five days. CTPA confirms a large "
            "saddle pulmonary embolus with RV/LV ratio 1.4 and septal flattening; hs-troponin I "
            "is 412 ng/L and NT-proBNP is 3,840 pg/mL. Initiate immediate systemic anticoagulation, "
            "hemodynamic stabilization, and urgent evaluation for systemic IV thrombolysis."
        ),
        "suggested_differentials": [
            "Acute high-risk saddle pulmonary embolism with right ventricular failure",
            "Acute coronary syndrome",
            "Cardiac tamponade",
        ],
        "detected_red_flags": [
            "Severe acute hypoxemia (SpO2 84% on room air requiring HFNC FiO2 0.70) and tachypnea (RR 30 breaths/min)",
            "Obstructive shock physiology (BP 92/58 mmHg, HR 126 bpm, near-syncope, JVD, lactate 3.8 mmol/L)",
            "Large saddle pulmonary embolus with severe RV strain (RV/LV ratio 1.4, troponin I 412 ng/L, NT-proBNP 3,840 pg/mL)",
        ],
        "triage_level": "emergent",
    },
    "CASE_010": {
        "summary": (
            "64-year-old female with AChR-antibody-positive generalized myasthenia gravis in "
            "neurorehabilitation developing acute impending myasthenic crisis two days after "
            "starting oral ciprofloxacin 500 mg BID alongside daily magnesium oxide 400 mg. "
            "Presents with worsening bilateral ptosis, nasal dysarthria, pooled secretions, neck "
            "flexor weakness (2/5), orthopnea, single-breath count drop from 32 to 11, FVC "
            "14 mL/kg, and NIF -19 cmH2O. ABG shows rising PaCO2 (46 mmHg, pH 7.36) despite "
            "tachypnea (RR 26 breaths/min) and falsely reassuring SpO2 (93%). Requires emergent "
            "ICU transfer for early airway protection, immediate discontinuation of ciprofloxacin "
            "and magnesium oxide, and initiation of IVIG or plasmapheresis."
        ),
        "suggested_differentials": [
            "Impending myasthenic crisis precipitated by fluoroquinolone and magnesium exposure",
            "Cholinergic crisis secondary to pyridostigmine excess",
            "Acute aspiration pneumonitis or early hospital-acquired pneumonia",
        ],
        "detected_red_flags": [
            "Critical neuromuscular respiratory decline (FVC 14 mL/kg, NIF -19 cmH2O, single-breath count 11) crossing intubation thresholds",
            "Ominous rising PaCO2 (46 mmHg) despite tachypnea (RR 26) and paradoxical supine breathing despite preserved SpO2 (93%)",
            "Severe bulbar dysfunction with secretion pooling and weak cough creating high airway compromise risk",
            "Active exposure to contraindicated agents in myasthenia gravis (ciprofloxacin 500 mg BID and magnesium oxide 400 mg daily)",
        ],
        "triage_level": "emergent",
    },
}


def load_clinical_cases(input_path: Path = DEFAULT_INPUT_PATH) -> list[ClinicalCase]:
    """Load and validate synthetic clinical cases from JSON."""
    raw_payload: Any = json.loads(input_path.read_text(encoding="utf-8"))
    return [ClinicalCase.model_validate(item) for item in raw_payload]


def save_llm_outputs(
    outputs: list[LLMSummaryOutput],
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """Persist validated LLM summary outputs to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized: list[dict[str, Any]] = [item.model_dump() for item in outputs]
    output_path.write_text(json.dumps(serialized, indent=2) + "\n", encoding="utf-8")


def generate_mock_summary(case: ClinicalCase) -> LLMSummaryOutput:
    """Return a deterministic simulated LLM clinical summary for the given case."""
    mock_payload: dict[str, Any] | None = _MOCK_RESPONSES.get(case.case_id)
    if mock_payload is None:
        return LLMSummaryOutput(
            case_id=case.case_id,
            summary=f"Simulated clinical summary for {case.case_id} ({case.specialty}).",
            suggested_differentials=list(case.ground_truth_differentials),
            detected_red_flags=list(case.ground_truth_red_flags),
            triage_level="urgent",
        )
    return LLMSummaryOutput.model_validate({"case_id": case.case_id, **mock_payload})


def _strip_code_fences(raw_text: str) -> str:
    """Remove optional markdown code fences returned by non-strict OpenAI-compatible backends."""
    cleaned: str = raw_text.strip()
    if cleaned.startswith("```"):
        lines: list[str] = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return cleaned


def _create_openai_client() -> tuple[OpenAI, str]:
    """Initialize an OpenAI-compatible client and target model name from environment variables."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    api_key: str = os.getenv("LLM_API_KEY", "").strip()
    if not api_key or api_key == "your_key_here":
        raise ValueError(
            "Valid LLM_API_KEY must be configured in .env when running with mock=False."
        )

    base_url: str | None = os.getenv("LLM_BASE_URL", "").strip() or None
    model: str = os.getenv("LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    client: OpenAI = OpenAI(api_key=api_key, base_url=base_url)
    return client, model


def generate_live_summary(
    case: ClinicalCase,
    client: OpenAI,
    model: str,
) -> LLMSummaryOutput:
    """Request a structured JSON clinical summary from an OpenAI-compatible chat endpoint."""
    user_prompt: str = (
        f"Case ID: {case.case_id}\n"
        f"Specialty: {case.specialty}\n\n"
        f"Clinical Note:\n{case.clinical_note}"
    )

    # Zero temperature enforces deterministic clinical synthesis for reproducible audits.
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    content: str | None = response.choices[0].message.content
    if not content:
        raise ValueError(f"Empty completion content returned for {case.case_id}.")

    parsed_payload: dict[str, Any] = json.loads(_strip_code_fences(content))
    # Anchor case_id to the source record to prevent cross-case misattribution in downstream scoring.
    parsed_payload["case_id"] = case.case_id
    return LLMSummaryOutput.model_validate(parsed_payload)


def generate_summaries(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    mock: bool = True,
    model: str | None = None,
) -> list[LLMSummaryOutput]:
    """Generate clinical LLM summaries for all cases and save results to JSON."""
    cases: list[ClinicalCase] = load_clinical_cases(input_path)
    outputs: list[LLMSummaryOutput] = []

    if mock:
        logger.info("Generating mock LLM summaries for %d clinical cases.", len(cases))
        for case in cases:
            outputs.append(generate_mock_summary(case))
    else:
        client, env_model = _create_openai_client()
        target_model: str = model or env_model
        logger.info(
            "Generating live LLM summaries for %d clinical cases using model '%s'.",
            len(cases),
            target_model,
        )
        for case in cases:
            outputs.append(generate_live_summary(case, client=client, model=target_model))

    save_llm_outputs(outputs, output_path)
    logger.info("Saved %d LLM summary outputs to %s.", len(outputs), output_path)
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clinical LLM summaries from synthetic clinical notes."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to input synthetic notes JSON file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write generated LLM outputs JSON file.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call the configured OpenAI-compatible API instead of using mock outputs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional model override for live mode (defaults to LLM_MODEL in .env).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _parse_args()
    generate_summaries(
        input_path=args.input,
        output_path=args.output,
        mock=not args.live,
        model=args.model,
    )


if __name__ == "__main__":
    main()
