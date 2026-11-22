"""UserMirrorer's EKB template, with explicit adaptations for the observed action space."""
from agent_simulation.vendor.usermirrorer.template import default_system

_TEXT = {"type": "string", "minLength": 1}
_FACTORS = {"type": "array", "items": {"type": "string", "minLength": 1}}
EKB_SCHEMA = {"type": "object", "properties": {
    "stimulus": _TEXT, "stimulus_factors": _FACTORS,
    "knowledge": _TEXT, "knowledge_factors": _FACTORS,
    "evaluation": _TEXT,
    "evaluation_style": {"type": "string", "enum": ["Logical", "Intuitive", "Impulsive", "Habitual", "Unknown"]}},
    "required": ["stimulus", "stimulus_factors", "knowledge", "knowledge_factors", "evaluation", "evaluation_style"],
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
For Query, predict the next Chinese query without access to the current query or
exposure. For Response, action contains clicked_item_ids and purchased_item_ids:
each may contain multiple displayed IDs or be empty. A purchase need not imply a
recorded click. Do not force a single choice or invent an extra no-click item.
Query generation remains a separate action and must not be replaced by a choice ID.
"""
