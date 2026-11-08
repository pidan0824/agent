"""Explicit model operations and decision-local references; storage keeps canonical IDs."""

from copy import deepcopy

from agent_simulation.agents.memory.validation import MAX_MEMORY_SOURCES
from agent_simulation.data.common import require


def prepare(payload):
    events = [*payload['experiences'], *payload.get('supplemental_evidence', [])]
    new = set(payload['new_evidence_ids'])
    customer = payload['actor_role'] == 'customer'
    if customer:
        events.sort(key=lambda event: event['source_id'] not in new)
    evidence = {f'E{i}': event['source_id'] for i, event in enumerate(events, 1)}
    sources = set(evidence.values())
    require(len(sources) == len(events), 'Duplicate reflection evidence source_id')
    memories = {f'M{i}': entry['memory_id'] for i, entry in
                enumerate(payload['previous_long_term_memory'], 1)}
    cards = [
        {'ref': ref, 'new': event['source_id'] in new,
         **deepcopy({k: v for k, v in event.items() if k != 'source_id'})}
        for ref, event in zip(evidence, events)
    ]
    previous = [{'ref': ref, 'content': entry['content'],
                 'generated_step': entry['generated_step']}
                for ref, entry in zip(memories, payload['previous_long_term_memory'])]
    visible = {'actor_role': payload['actor_role'], 'profile': payload['profile'],
               'evidence': cards, 'previous_memory': previous}
    if customer:
        purchases = {}
        for card in cards:
            for product in card.get('content', {}).get('purchased_items', []):
                key = (product['item_id'], product['seller_id'])
                purchases.setdefault(key, []).append(card['ref'])
        visible['purchase_evidence'] = [
            {'item_id': item, 'seller_id': seller, 'evidence_refs': list(dict.fromkeys(refs))}
            for (item, seller), refs in purchases.items()]
    citable = evidence if new & sources else {}
    schema = output_schema(citable, memories)
    return visible, schema, (evidence, memories)


def output_schema(evidence, memories):
    def refs(values, maximum):
        return {'type': 'array', 'items': {'type': 'string', 'enum': list(values)},
                'minItems': 1, 'maxItems': min(len(values), maximum), 'uniqueItems': True}

    def choice(action, *, replacing=False):
        fields = {'action': {'type': 'string', 'enum': [action]}}
        if replacing:
            fields['memory_refs'] = refs(memories, len(memories))
        if action in ('add', 'revise'):
            fields['evidence_refs'] = refs(evidence, MAX_MEMORY_SOURCES)
            fields['content'] = {'type': 'string', 'minLength': 1}
        return {'type': 'object', 'properties': fields,
                'required': list(fields), 'additionalProperties': False}

    choices = [choice('no_update')]
    if evidence:
        choices.append(choice('add'))
        if memories:
            choices.extend((choice('revise', replacing=True), choice('retract', replacing=True)))
    return {'anyOf': choices}


def decode(raw, schema, references):
    """Translate a declared operation; never infer one from a missing optional field."""
    from agent_simulation.agents.schemas import strict_json, validate_arguments

    draft = strict_json(raw)
    validate_arguments(draft, schema)
    evidence, memories = references
    action = draft['action']
    sources = [evidence[ref] for ref in draft.get('evidence_refs', [])]
    result = {'action': action, 'content': draft.get('content', ''), 'source_ids': sources}
    if action in ('revise', 'retract'):
        result['supersedes'] = [memories[ref] for ref in draft['memory_refs']]
    return result
