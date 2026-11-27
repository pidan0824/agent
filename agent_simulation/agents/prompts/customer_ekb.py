"""UserMirrorer's EKB template, with explicit adaptations for the observed action space."""
from agent_simulation.vendor.usermirrorer.template import default_system
from agent_simulation.evaluation.customer_evidence import EVIDENCE_SCHEMA, EVIDENCE_NOTE

_TEXT = {"type": "string", "minLength": 1}
_FACTORS = {"type": "array", "items": {"type": "string", "minLength": 1}}
EKB_SCHEMA = {"type": "object", "properties": {
    "evidence_refs": EVIDENCE_SCHEMA, "stimulus": _TEXT, "stimulus_factors": _FACTORS,
    "knowledge": _TEXT, "knowledge_factors": _FACTORS,
    "evaluation": _TEXT,
    "evaluation_style": {"type": "string", "enum": ["Logical", "Intuitive", "Impulsive", "Habitual", "Unknown"]}},
    "required": ["evidence_refs", "stimulus", "stimulus_factors", "knowledge", "knowledge_factors", "evaluation", "evaluation_style"],
    "additionalProperties": False}


def ekb_instructions():
    return default_system + """
PROJECT ADAPTATIONS (override the single-choice and text output requirements above):
Return raw JSON with rationale and action, according to the supplied schema; never use Markdown fences.
Map the three original stages to stimulus, knowledge, evaluation, with their factor lists
and evaluation_style. This is a simulated explanation, not the person's
observed thoughts. Ground it in supplied facts; explicitly state missing evidence.
Do not infer weather, hunger, finances or other unobserved states. Factor lists
may be empty and evaluation_style may be Unknown if evidence is insufficient.
Response action contains clicked_item_ids and purchased_item_ids. Follow the
supplied action schema: synthetic single-positive lists allow at most one distinct
item across both fields; full-exposure decisions may contain multiple items.
Both fields may be empty. Purchase need not imply a recorded click.
Query generation is a separate direct-action task, outside this EKB contract.
""" + EVIDENCE_NOTE


def concise_ekb_instructions():
    """Budgeted task adaptation; the original author template remains unchanged."""
    return """Predict the user's recorded clicks and purchases from the supplied facts.
Return the supplied rationale/action JSON. Keep stimulus, knowledge and evaluation
to one short sentence each: current need, the most relevant observed comparison,
and the resulting choice or abstention. Do not repeat the candidate list.
Write the explanation in Chinese, preserving original query/product wording and IDs.
Different item IDs are not interchangeable even when their titles match.
Use zero to two supported factor labels per list. Use Unknown evaluation_style
unless observations justify a more specific description. An explanation is a
tentative account, not the person's observed thoughts. Do not invent motives,
finances, emotional states or current prices. Relevance does not imply a click,
and a purchase need not have a recorded click. Empty actions are valid.
This request's clicks and purchases are predictions, not already observed facts.
""" + EVIDENCE_NOTE
