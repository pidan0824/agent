"""Render shop facts and price decisions as Seller messages."""

import json

from agent_simulation.agents.context.pricing_evidence import EVIDENCE_SCHEMAS, project_seller
from agent_simulation.data.common import require
from agent_simulation.agents.schemas import validate_arguments
from agent_simulation.agents.actions import price_schema
from agent_simulation.agents.prompts.context_format import pack, unpack
from agent_simulation.agents.prompts.instructions import (
    FACT_RULES, SELLER_STEPS, SELLER_EVIDENCE_NOTE, SELLER_SUPERVISED_SCOPE, SELLER_SYSTEM, SELLER_ANSWER_RULES,
    SHOP_BATCH_NOTE, TABLE_NOTE, VALUE_NOTE,
)


TABLE_PATHS = frozenset({
    ("current_state", "products", "recent", "price_runs"),
    ("history", "price_changes", "runs"),
    ("history", "similar_evidence", "price_runs"),
    ("external_reference", "competitors"),
})

SELLER_RUNTIME_PRICES = (
    ' Use {"prices":[{"item_id":ID,"price_raw":PRICE}]} with IDs from '
    'decision.requested_item_ids. Repeat a known active price to keep it. Submit a positive '
    'number to set a price, or send null to keep an unknown active price.'
)


def encode_input(state):
    """Render a projected Seller Evidence View."""
    require(isinstance(state, dict) and state.get("schema") in EVIDENCE_SCHEMAS,
            "Expected a projected Seller Evidence View")
    encoded = json.dumps(
        pack(state, table_paths=TABLE_PATHS), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return encoded


def decode_input(text):
    """Decode the current Seller decision input for diagnostics."""
    state = unpack(json.loads(text))
    require(isinstance(state, dict) and state.get("schema") in EVIDENCE_SCHEMAS,
            "Unknown Seller input format")
    return state


def seller_messages(state, target=None, *, system=SELLER_SYSTEM, rules=SELLER_ANSWER_RULES,
                    audit_sink=None):
    """Use one model-facing representation in supervision and the runtime harness."""
    visible, audit = project_seller(state, include_audit=audit_sink is not None)
    if audit_sink is not None:
        audit_sink.update(audit)
    from agent_simulation.agents.context import context_model
    model = context_model()
    if model is not None:
        # Summarize only historical evidence; original statistics and current quotes stay intact.
        history = visible["history"]
        records = [{"scope": key, "record": record}
                   for key in ("price_changes", "similar_evidence") for record in history[key]]
        summary = model.summarize(records, scope="Seller past price changes and retrieved business history")
        if summary is not None:
            history.update(price_changes=[], similar_evidence=[], summary=summary)
        for product in visible["current_state"]["products"]:
            summary = model.summarize(product["recent"], scope=f"Seller recent history for item {product['item_id']}")
            if summary is not None:
                product.update(recent=[], recent_summary=summary)
    content = encode_input(visible)
    note = TABLE_NOTE if '"$table":{' in content else VALUE_NOTE
    messages = [
        {"role": "system", "content": system + rules + "\n" + note + SELLER_EVIDENCE_NOTE},
        {"role": "user", "content": content},
    ]
    if target is not None:
        requested = state["requested_price_item_ids"]
        require(len(requested) == 1, "Expected one requested Seller supervision ID")
        validate_arguments(target, price_schema(requested))
        messages.append({"role": "assistant", "content": json.dumps(target, ensure_ascii=False, allow_nan=False)})
    return messages


def pricing_messages(context, target=None, audit_sink=None):
    result = seller_messages(context, target, audit_sink=audit_sink)
    result[0]["content"] += SELLER_SUPERVISED_SCOPE
    if "product_batch" in context:
        result[0]["content"] += SHOP_BATCH_NOTE
    return result


def seller_decision_messages(context, *, audit_sink=None):
    rules = (FACT_RULES + SELLER_STEPS + "\nOutput\n"
             + "Return one JSON object with the requested prices." + SELLER_RUNTIME_PRICES)
    if "product_batch" in context:
        context = {**context, "price_scope": "Submit one price per owned product in this batch."}
    if "requested_price_item_ids" in context:
        rules += ' Include each requested ID exactly once.'
    if "decision_time" in context:
        rules += ' Prices take effect together after all Sellers finish.'
    else:
        rules += ' Prices take effect after the shop review is complete.'
    return seller_messages(context, rules=rules, audit_sink=audit_sink)
