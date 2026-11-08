"""Instructions for Customer and Seller long-term reflection."""

SUMMARY_PROMPTS = {
    'customer': (
        'Reflect as this Customer on product preferences, recurring interests, search and '
        'comparison habits, purchases, and possible price sensitivity. Repeated searches can '
        'suggest interest. Claims about stable habits, satisfaction, motives or budget need '
        'specific evidence. Price sensitivity needs observed prices and shopping choices. Use '
        'purchase_evidence for actual purchases by item and shop when checking a prior claim '
        'of never buying.'
    ),
    'seller': (
        'Reflect as this Seller on pricing and demand, differences among your products, '
        'competitive position, and how the platform may allocate exposure. Compare the same '
        'item across observations before describing a price change. Keep counts within their '
        'recorded windows and treat purchase records as events. Treat possible price effects '
        'and allocation rules as working hypotheses. Avoid claims about unobserved costs, '
        'inventory, competitor performance or the actual ranking algorithm.'
    ),
}

SUMMARY_INSTRUCTION = """
Make one update useful for future decisions. Separate observations from tentative beliefs,
consider conflicting evidence, and scope conclusions to the cited products, shops and
observations. Preserve relevant names and IDs. Keep historical indices and simulation
steps separate and use them only for ordering. Keep prices as raw values without assigning
a currency unit or converting them.

Choose one action.
- no_update: No useful new insight or correction.
- add: Record a new insight unrelated to correcting a supplied memory.
- revise: Replace a supplied memory, retaining what the evidence still supports.
- retract: Withdraw a misleading or unsupported memory without replacement.

For add or revise, write one concise English paragraph and cite 1–3 supporting E cards in
evidence_refs, including a new one. For revise or retract, identify the affected M entries
in memory_refs. Return only JSON matching the schema.
"""

EVIDENCE_ORDER = (
    '\n\nReview all evidence cards marked new=true and check previous memories for conflicts. '
    'Use older cards as relevant background.\n'
)
