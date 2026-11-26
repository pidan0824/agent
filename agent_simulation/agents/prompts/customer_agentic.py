"""Load Customer agentic instructions; theory references remain optional."""

from pathlib import Path

from agent_simulation.agents.prompts.instructions import FACT_RULES


_ROOT = Path(__file__).with_name("skills") / "customer-shopping"


def skill_references(*, theory=True):
    references = {"evidence": "Interpret historical observations, missing values and descriptive features."}
    if theory:
        references["decision-types"] = "Optional EKB and search-effort hypotheses; evidence may contradict them."
    return references


def read_skill_reference(name, *, theory=True):
    if name not in skill_references(theory=theory):
        raise ValueError("Unknown or disabled Customer skill reference")
    return (_ROOT / "references" / (name + ".md")).read_text(encoding="utf-8")


def skill_reference_names(*, theory=True):
    return list(skill_references(theory=theory))


read_reference = read_skill_reference


def agentic_customer_instructions(*, theory=True):
    from agent_simulation.agents.agentic import TEACHER_OUTPUT_NOTE
    source = (_ROOT / "SKILL.md").read_text(encoding="utf-8")
    # Frontmatter is packaging metadata, not a behavioral instruction.
    body = source.split("---", 2)[-1].strip()
    if not theory:
        body = body.replace(
            "Theory-enabled runs include the EKB hypotheses directly in the prompt; these may be rejected and do not prescribe a query sequence.",
            "Theory references are disabled in this run; choose queries directly from the observed evidence.",
        )
    theory_text = ("\nDecision hypotheses available in this run:\n" + read_skill_reference("decision-types")
                   if theory else "")
    from agent_simulation.agents.prompts.customer_ekb import ekb_instructions
    rationale = ekb_instructions() if theory else TEACHER_OUTPUT_NOTE
    return body + "\n" + FACT_RULES + "\n" + read_skill_reference("evidence", theory=theory) + theory_text + "\n" + rationale


def fixed_customer_instructions(*, concise=False):
    """Fixed-information EKB: retain upstream core once; omit tool-planning wrapper."""
    from agent_simulation.agents.prompts.customer_ekb import ekb_instructions, concise_ekb_instructions
    if concise:
        return FACT_RULES + "\n" + concise_ekb_instructions()
    return FACT_RULES + "\n" + read_skill_reference("evidence") + "\n" + ekb_instructions()
