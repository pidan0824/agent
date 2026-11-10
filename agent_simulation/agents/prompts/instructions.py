"""Shared decision instructions, price convention and supervised outputs."""

PRICE_UNIT_NOTE = 'The sandbox interprets price and estimated amount fields in RMB fen. This is a modeling convention; the original source unit is unconfirmed. Preserve stored numbers and submit prices on this same scale. Do not label these numbers as yuan or convert them.'

VALUE_NOTE = 'Values are preserved; null means unknown.'
TABLE_NOTE = 'In each $table, columns defines row positions and shared applies to every row. ' + VALUE_NOTE

FACT_RULES = (
    "\nEvidence\nOriginal records, statistics and quotes take precedence over model_summary interpretations. "
    + PRICE_UNIT_NOTE +
    " Historical time_index and simulation indices are separate ordering clocks, not calendar durations. "
    "Counts belong to their stated products and observation windows; overlapping sections must not be added. "
    "Behavioral/business 30-day snapshots stay frozen at observed_time_index during simulation; "
    "simulated feedback does not update those totals. Item descriptions are attributes, not historical prices. "
    "Missing values are unknown, not zero.\n"
)

MEMORY_NOTE = (
    "Long-term memory is revisable experience, not additional events. Original facts take precedence."
)
TITLE_NOTE = "An item_title object {title_ref:i} copies item_title from behavioral_summary.item_details[i] (zero-based)."


SELLER_STEPS = "\nDecision\nUse the shop overview and each requested product's price, exposure, click and purchase history. Compare available market distributions and public reference prices within their stated coverage. Consider differences between your products and relevant experience. Any inferred platform exposure rule is a hypothesis. Observed prices do not establish actual backend price-change actions. Purchase counts count records; estimated units assume one unit per purchase record. The environment activates prices and settles outcomes.\n"

CUSTOMER_STEPS = '\nDecision\nUse the profile, recent history and retrieved experiences to assess shopping preferences. Other search topics may also inform consumption habits. Use recent behavior lists and 30-day snapshots as background, not action quotas. Repeated IDs in those lists are not quantities or independently confirmed events. Logged exposure does not prove attention; absence of a click is not explicit dislike. Past clicks and purchases belong to history, separate from the currently displayed products.\n'

HISTORY_NOTE = 'In recent and retrieved history, clicked_item_refs and purchased_item_refs index short_term_memory.item_attributes. Its sellers_ref, brands_ref and categories_ref index short_term_memory.attribute_sets (zero-based). Historical snapshots keep their original prices.'

SELLER_ANSWER_RULES = FACT_RULES + SELLER_STEPS + '\nOutput\nReturn one JSON object with the prices for the requested scope: {"prices":[{"item_id":ID,"price_raw":PRICE}]}. Keeping a price is allowed. Return only the JSON object.\n'

SELLER_SYSTEM = 'Task\nYou manage this shop. Use the shop overview, visible product histories and business experience to decide prices for the requested products.\n'

SELLER_EVIDENCE_NOTE = (
    " Price-run counts total the stated observations. Inclusive ranges mark observed points, "
    "not continuous activity or exact change times; gaps imply neither unchanged prices nor zero activity. "
    "observed_market covers all eligible same-category public quotes. Quantiles weight products equally; "
    "own_quote.percentile is a 0..1 midrank. known_products includes unknown quotes; products counts valid ones. "
    "same_brand is a subset, and largest_shop_product_share is offer concentration, not sales share. "
    "Use original titles to account for differences in size, material, and pack count before comparing quotes. "
    "Older quotes remain historical references. "
    "Logged exposures do not establish attention, and missing clicks or purchases do not establish rejection. "
    "decision.requested_item_ids gives the quote scope."
)

SHOP_OVERVIEW_NOTE = 'Each product overview covers its retained recent records only. Windows may differ across products; inclusive start/end indices mark observations, not actual price-change times. Observation gaps do not mean zero activity. Detailed attributes for the selected products are in product_details.'

SHOP_BATCH_NOTE = ' This capacity-limited batch belongs to the same Seller. Shop reporting periods retain their stated coverage; current_state.products lists this batch only.'


QUERY_TASK = 'Task\nYou are an e-commerce customer. Generate the next product search query from your profile, shopping history and accumulated experience. Write the query in Chinese; preserve original product and brand names.\n'

QUERY_OUTPUT = '\nOutput\nReturn one JSON object with exactly one key, query. Its value must be the specific Chinese product search query you choose from the context. Do not output a placeholder, Markdown, explanations or extra fields.'

RESPONSE_SYSTEM = 'Task\nYou are an e-commerce customer. Evaluate the currently displayed products for the current query.\n' + FACT_RULES + CUSTOMER_STEPS + 'Identify the need expressed by the query, inspect product descriptions, attributes and prices, and assess their fit with your preferences. Do not invent missing attributes. Stored product order does not establish actual screen position. If candidate_batch is present, evaluate only this subset; a capacity batch is not a recorded page.\n\nOutput\nReturn only {"clicked_item_ids":[integer IDs],"purchased_item_ids":[integer IDs]}. Include both fields with no extra text. Each set may independently be empty or contain multiple currently displayed item IDs.'

SELLER_SUPERVISED_SCOPE = (" Submit one strictly positive price for each item in decision.requested_item_ids, including unchanged prices. "
                           "Use only those IDs. An empty prices list is invalid.")
