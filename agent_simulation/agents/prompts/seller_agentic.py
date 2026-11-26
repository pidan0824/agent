"""Seller theory as optional hypotheses inside one autonomous pricing policy."""

from pathlib import Path

from agent_simulation.agents.prompts.instructions import FACT_RULES
from agent_simulation.agents.agentic import TEACHER_OUTPUT_NOTE

_ROOT = Path(__file__).with_name("skills") / "seller-pricing"


def skill_reference_names(*, theory=True):
    return ["evidence", *(["pricing-hypotheses"] if theory else [])]


def read_reference(name, *, theory=True):
    if name not in skill_reference_names(theory=theory):
        raise ValueError("Unknown or disabled Seller reference")
    return (_ROOT / "references" / (name + ".md")).read_text(encoding="utf-8")


def agentic_seller_instructions(*, theory=True):
    body = (_ROOT / "SKILL.md").read_text(encoding="utf-8").split("---", 2)[-1].strip()
    return (body + "\n" + FACT_RULES + "\n" + read_reference("evidence") + "\n"
            + (read_reference("pricing-hypotheses") if theory else
               "Theory hypotheses are disabled; justify the prediction directly from available evidence.")
            + "\n" + TEACHER_OUTPUT_NOTE
            + "\nIn each rationale.evidence entry cite at least one supplied evidence_id in square brackets, "
              "for example [S0123456789ab]. Cite only input or actual tool observations, not your own hypotheses. "
              "A citation identifies an available source, not proof of the inference.")
