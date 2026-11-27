"""UserMirrorer's EKB template, with explicit adaptations for the observed action space."""
from agent_simulation.vendor.usermirrorer.template import default_system
from agent_simulation.evaluation.customer_evidence import EVIDENCE_SCHEMA, EVIDENCE_NOTE

_TEXT = {"type": "string", "minLength": 1}
_FACTORS = {"type": "array", "items": {"type": "string", "minLength": 1}}
EKB_SCHEMA = {"type": "object", "properties": {
    "stimulus": _TEXT, "stimulus_factors": _FACTORS,
    "knowledge": _TEXT, "knowledge_factors": _FACTORS,
    "evaluation": _TEXT,
    "evaluation_style": {"type": "string", "enum": ["Logical", "Intuitive", "Impulsive", "Habitual", "Unknown"]},
    "evidence_refs": EVIDENCE_SCHEMA},
    "required": ["stimulus", "stimulus_factors", "knowledge", "knowledge_factors", "evaluation", "evaluation_style", "evidence_refs"],
    "additionalProperties": False}


_CHOICE_NOTE = """
Make the explanation distinguish between the actual candidates, including items
in the same category. In knowledge, connect a specific query requirement or
history-derived preference to visible attributes of concrete candidate item IDs.
In evaluation, compare the selected item(s) with the strongest plausible unselected
alternative: identify which difference matters to this user and why. Cite the
supporting facts; do not list every candidate. Category interest shared by the
whole list is not a reason to select one item over another.
Past interaction counts indicate familiarity, not a rule to choose the most
frequent item. Compare relevant attributes, including differences from past items.
Use only the named item's own price evidence; unknown prices cannot support a
price ranking. Write the explanation first, then evidence_refs: each preference
comparison needs both the historical B/H facts and the candidate P attribute IDs.
If only one candidate exists, do not invent a rival.
If the available information cannot distinguish the alternatives, state that
ambiguity without inventing a difference. Uncertainty alone does not mean no action.
Explain clicks and purchases separately; no historical purchase does not imply
no future purchase. Relevant simulated motives can be tentative hypotheses, but
must not replace the preference-to-attribute comparison or alter observed facts.
"""


def ekb_instructions():
    return default_system + """
PROJECT ADAPTATIONS (override the single-choice and text output requirements above):
Return raw JSON with rationale and action, according to the supplied schema; never use Markdown fences.
Map the three original stages to stimulus, knowledge, evaluation, with their factor lists
and evaluation_style. This is a simulated explanation, not the person's
observed thoughts. Ground it in supplied facts; explicitly state missing evidence.
Distinguish simulated motives from observed facts. Factor lists
may be empty and evaluation_style may be Unknown if evidence is insufficient.
Response action contains clicked_item_ids and purchased_item_ids. Follow the
supplied action schema: synthetic single-positive lists allow at most one distinct
item across both fields; full-exposure decisions may contain multiple items.
Both fields may be empty. Purchase need not imply a recorded click.
Query generation is a separate direct-action task, outside this EKB contract.
""" + _CHOICE_NOTE + EVIDENCE_NOTE


def concise_ekb_instructions():
    """Budgeted task adaptation; the original author template remains unchanged."""
    return """Predict the user's recorded clicks and purchases from the supplied facts.
Return the supplied rationale/action JSON. Keep stimulus to one short sentence;
use one or two short sentences each for knowledge and evaluation, prioritizing
candidate-specific comparison over a general summary of history.
Write the explanation in Chinese, preserving original query/product wording and IDs.
Different item IDs are not interchangeable even when their titles match.
Use zero to two supported factor labels per list. Use Unknown evaluation_style
unless observations justify a more specific description. An explanation is a
tentative account, not the person's observed thoughts. Do not invent product
attributes, interactions or current prices. Relevance does not imply a click,
and a purchase need not have a recorded click. Empty actions are valid.
Historical catalog prices do not prove this user saw those prices.
Absence of a behavior in the supplied history means only no record in that window.
This request's clicks and purchases are predictions, not already observed facts.
""" + _CHOICE_NOTE + EVIDENCE_NOTE
