"""One decision view for SFT and simulation; full record identities stay in audit."""
from collections import defaultdict
from copy import deepcopy
from agent_simulation.data.common import require

SCHEMA = "seller-decision-evidence-v5"
EVIDENCE_SCHEMAS = frozenset((SCHEMA,))
COUNTS = ("exposure_records", "click_records", "purchase_records")


def runs(rows):
    """Summarize consecutive observations, breaking at a skipped record."""
    result = []
    current = None
    previous_index = -1
    for index, row in rows:
        require(index > previous_index, "Price observations must be chronological")
        previous_index = index
        if row is None:
            current = None
            continue
        price, status = row["price_raw"], row["price_status"]
        if current is None or (current["price_raw"], current["price_status"]) != (price, status):
            current = dict(price_raw=price, price_status=status, observation_range=[index, index],
                           observations=0, **dict.fromkeys(COUNTS, 0))
            result.append(current)
        current["observation_range"][1] = index
        current["observations"] += 1
        for field in COUNTS:
            current[field] += row[field]
    return result


def historical_runs(history, recent, bound):
    """Omit complete runs only where the known recent suffix covers the source records."""
    total = history["all_observations"]
    require(0 <= total["first_index"] <= total["last_index"] < bound,
            "Invalid price summary observation range")
    # Both views use the same strictly ordered per-item, per-clock observations.
    # If the recent suffix ends at the summary's latest record, matching run
    # boundaries and row counts prove coverage of the complete run.
    covered = {
        (*r["observation_range"], r["observations"], r["price_raw"], r["price_status"])
        for r in recent
    } if recent and recent[-1]["observation_range"][1] == total["last_index"] else set()
    result = []
    previous_end = -1
    for run in history["recent_observation_runs"]:
        start, end = run["first_index"], run["last_index"]
        require(total["first_index"] <= start <= end <= total["last_index"]
                and start > previous_end, "Invalid price run observation range")
        previous_end = end
        price = run["price_raw"]
        # Older saved contexts omit run status. Only a numeric quote is unambiguous.
        status = run.get("price_status", "observed" if price is not None else None)
        key = (start, end, run["observations"], price, status)
        if key not in covered:
            # A numeric price already denotes an observed quote.
            result.append({k: v for k, v in run.items()
                           if k != "price_status" or run["price_raw"] is None})
    return result


def project_seller(state, *, include_audit=True):
    """Project authorized internal state into decision evidence and optional provenance."""
    seller=state['profile']['seller_id']; cutoff=state['history_cutoff_exclusive']
    sim_end=state.get('simulation_cutoff_exclusive')
    if sim_end is None and 'simulation_state' in state:
        sim_end=state['simulation_state']['through_step_inclusive']+1
    logical=state.get('decision_time',{}).get('logical_time')
    sources={'observed':{'kind':'own_observed_facts','owner':seller,'clock':'observed_time_index','end_exclusive':cutoff},
             'public_observed':{'kind':'public_other_shop_quotes','clock':'observed_time_index','end_exclusive':cutoff}}
    if sim_end is not None:
        sources['settled']={'kind':'own_settled_facts','owner':seller,'clock':'simulation_settled_step','end_exclusive':sim_end}
        sources['public_simulation']={'kind':'public_simulation_prices','clock':'simulation_logical_time' if logical is not None else 'simulation_settled_step',
                                      'as_of':logical if logical is not None else sim_end-1}
        sources['active']={**sources['public_simulation'],'kind':'own_active_prices','owner':seller}
    audit = {'schema': SCHEMA, 'owner': seller, 'evidence': {}, 'memories': []} if include_audit else None
    facts = {}
    memories = state.get('long_term_memory', [])
    citation_refs = {}

    def add_fact(clock, item, index, row, ref):
        require(clock in ('observed_time_index', 'simulation_settled_step'), 'Unknown evidence clock')
        end = cutoff if clock == 'observed_time_index' else sim_end
        require(type(index) is int and index >= 0 and end is not None and index < end, 'Future evidence')
        require(row.get('item_id', item) == item and row.get('seller_id', seller) == seller,
                'Foreign evidence')
        key = (clock, item, index)
        value = tuple(row[k] for k in ('price_raw', 'price_status', *COUNTS))
        existing = facts.get(key)
        if existing is not None:
            require(existing[0] == value, 'Conflicting evidence for one record')
            return key, existing[1]
        facts[key] = (value, ref)
        return key, None

    def register(ref, source, path, keys=(), **extra):
        if audit is not None:
            audit['evidence'][ref] = dict(source=source, path=path, record_keys=[list(k) for k in keys],
                                          **deepcopy(extra))

    rows_by_item_clock = defaultdict(list)
    for event in state.get('short_term_memory', []):
        clock = 'observed_time_index' if 'time_index' in event else 'simulation_settled_step'
        for row in event['content']:
            index = (row.get('time_index', event.get('time_index')) if clock == 'observed_time_index'
                     else row.get('settled_step', event.get('simulation_step')))
            rows_by_item_clock[row['item_id'], clock].append((index, row))
    details={p['item_id']:p for p in state.get('product_details',[])}
    products = state['owned_products']
    owned_ids = {product['item_id'] for product in products}
    current=[]; summaries=[]; external=[]
    for n,p in enumerate(products):
        item=p['item_id'];profile=details.get(item,p)
        own=dict(item_id=item,product={k:deepcopy(v) for k,v in profile.items()
                if k in ('item_title','brand_name','category_level1_name','category_level2_name','category_level3_name')},
                last_observed_quote={'source':'observed','price_raw':p['last_observed_price_raw'],
                  'price_status':p['price_status'],'observed_at':p['last_observed_time_index']},recent=[])
        require(p['last_observed_time_index']<cutoff,'Future own quote')
        if 'business_summary' in p:
            require(p['business_summary']['observed_time_index'] < cutoff, 'Future business summary')
            own['business_summary'] = deepcopy(p['business_summary'])
        if p.get('last_valid_reference'):
            require(p['last_valid_reference']['time_index']<cutoff,'Future fallback quote')
            own['last_observed_quote']['last_valid_reference']=deepcopy(p['last_valid_reference'])
        for clock,source in [('observed_time_index','observed'),('simulation_settled_step','settled')]:
            records = sorted(rows_by_item_clock.get((item, clock), ()), key=lambda row: row[0])
            if not records: continue
            ref=f'R{n+1}:{source}'; keys=[]; kept=[]
            for index,r in records:
                key, existing_ref = add_fact(clock, item, index, r, ref)
                if existing_ref is None:
                    kept.append((index, r))
                    keys.append(key)
            own['recent'].append(dict(evidence_ref=ref,source=source,price_runs=runs(kept)))
            register(ref,source,['short_term_memory'],keys)
        active=state.get('simulation_state',{}).get('active_prices',{}).get(str(item))
        if active is not None:
            own['active_quote']=dict(source='active',price_raw=active['price_raw'])
            register(f'A{item}','active',['simulation_state','active_prices',str(item)],quote_provenance=active)
        if 'new_feedback' in p:
            own['since_previous_decision']={'source':'settled',**{k:p['new_feedback'][k] for k in COUNTS}}
        current.append(own)
        for field,source in [('observed_price_history','observed'),('settled_price_history','settled')]:
            h=p.get(field)
            if h is None:continue
            bound=cutoff if source=='observed' else sim_end
            clock='observed_time_index' if source=='observed' else 'simulation_settled_step'
            require(h['clock']==clock,'Price summary clock mismatch')
            require(bound is not None and h['end_exclusive']==bound
                    and h['all_observations']['last_index']<bound,'Future price summary')
            ref=f'P{n+1}:{source}'
            recent = next((r for r in own['recent'] if r['source']==source), None)
            kept_runs = historical_runs(h, recent['price_runs'] if recent else [], bound)
            summaries.append(dict(evidence_ref=ref,source=source,item_id=item,
               observation_range=[h['all_observations']['first_index'],h['all_observations']['last_index']],
               observations=h['all_observations']['observations'],distinct_prices=h['distinct_valid_prices'],
               observed_quote_changes=h['valid_quote_changes'],unknown_price_observations=h['unknown_price_observations'],
               omitted_earlier_runs=h['omitted_earlier_runs'],runs=kept_runs,
               **({'recent_evidence_ref':recent['evidence_ref']}
                  if len(kept_runs)<len(h['recent_observation_runs']) else {})))
            register(ref,source,['owned_products',n,field],derivation={'item_id':item,'owner':seller,'end_exclusive':bound,
                       'clock':h['clock'],'policy':'incremental_observed_price_history_v1'})
        m=p.get('market_position')
        if m is None:continue
        entry={'item_id':item}
        if m.get('status'):entry['status']=m['status']
        if 'recent' in m:
            require(m['window']['end_exclusive']==cutoff,'Market cutoff mismatch')
            entry['observed_market']={'source':'public_observed','window':deepcopy(m['window']),**deepcopy(m['recent'])}
        if 'simulation' in m:
            require(sim_end is not None,'Simulation market without simulation clock')
            basis=m['simulation'].get('price_as_of')
            if basis is not None:
                require(basis['clock'] in ('simulation_logical_time','simulation_settled_step'),
                        'Unknown simulation market clock')
                expected=logical if basis['clock']=='simulation_logical_time' else sim_end-1
                require(basis['as_of']==expected,'Stale/future simulation market')
            entry['simulation_market']={'source':'public_simulation',**deepcopy(m['simulation'])}
        entry['competitors']=[]
        for c in m.get('competitor_examples',{}).get('products',[]):
            require(c['seller_id']!=seller and c['reference_price_time_index']<cutoff,'Foreign/future public quote')
            card={k:deepcopy(v) for k,v in c.items() if k not in ('purchase_records','simulation_price')}
            card['source']='public_observed'
            card['quote_recency']='recent_window' if c['reference_price_time_index']>=m['window']['start_inclusive'] else 'older_reference'
            if 'simulation_price' in c:
                require(sim_end is not None,'Simulation price without simulation clock')
                sp=c['simulation_price']
                require(sp['clock'] in ('simulation_logical_time','simulation_settled_step'),
                        'Unknown simulation quote clock')
                expected=logical if sp['clock']=='simulation_logical_time' else sim_end-1
                require(sp['as_of']==expected,'Stale/future active public price')
                card['active_quote']={'source':'public_simulation',**deepcopy(sp)}
            entry['competitors'].append(card)
        external.append(entry)
        register(f'E{n+1}','public_observed',['owned_products',n,'market_position'],
                 selection=m.get('competitor_examples',{}).get('selection'))
    similar = []
    described = set(owned_ids)
    segments = state.get('retrieved_context', {}).get('history', {}).get('segments', [])
    for number, segment in enumerate(segments, 1):
        require(segment['owner_id'] == seller, 'Foreign retrieved history')
        clock = segment['clock']
        source = 'observed' if clock == 'observed_time_index' else 'settled'
        item = segment['item_id']
        ref = f'H{number}'
        rows, keys = [], []
        for row_number, row in enumerate(segment['records'], 1):
            index = row['index']
            key, existing_ref = add_fact(clock, item, index, row, ref)
            if memories:
                citation_refs[number, row_number] = existing_ref or ref
            rows.append((index, row if existing_ref is None else None))
            if existing_ref is None:
                keys.append(key)
        price_runs = runs(rows)
        if not price_runs:
            continue
        entry = dict(evidence_ref=ref, source=source, item_id=item)
        if item not in described:
            require(item in details, 'Missing retrieved product description')
            entry['product'] = {k: v for k, v in details[item].items()
                                if k in ('item_title', 'brand_name', 'category_level3_name')}
            described.add(item)
        entry['price_runs'] = price_runs
        similar.append(entry)
        register(ref, source, ['retrieved_context', 'history', 'segments', number - 1],
                 keys, match=segment.get('match'))
    history = {'price_changes': summaries, 'similar_evidence': similar}
    for memory in memories:
        require(memory.get('owner')==f'seller:{seller}','Foreign or unattributed long-term memory')
        require(sim_end is not None and type(memory.get('simulation_step')) is int and memory['simulation_step']<sim_end,'Future long-term memory')
        require(memory.get('source_ids') and len(memory['source_ids'])==len(memory.get('evidence_times',[])), 'Uncited long-term memory')
        require(memory.get('memory_id') and isinstance(memory.get('content'),str),'Invalid long-term memory')
        generated=memory.get('generated_logical_time')
        require(generated is None or logical is not None and generated<=logical,'Future logical-time memory')
        refs=set();support=[]
        for evidence in memory['evidence_times']:
            for field,bound in [('time_index',cutoff),('settled_step',sim_end),('available_step',sim_end)]:
                require(field not in evidence or bound is not None and evidence[field]<bound,'Future memory citation')
            source_cutoff = evidence.get('historical_cutoff_exclusive', cutoff)
            require(type(source_cutoff) is int and 0 < source_cutoff <= cutoff
                    and evidence.get('time_index', -1) < source_cutoff, 'Future memory citation cutoff')
            t=evidence.get('available_logical_time')
            require(t is None or logical is not None and t<=logical,'Future citation logical time')
            matches=evidence.get('same_observations_in_retrieved_history',[])
            for match in matches:
                pos=(match['segment_number'],match['record_number'])
                require(pos in citation_refs, 'Unresolved memory history citation')
                refs.add(citation_refs[pos])
            support.append({k:v for k,v in evidence.items() if k!='same_observations_in_retrieved_history'})
        entry={k:deepcopy(memory[k]) for k in ('memory_id','content','evidence_kind')}
        entry.update(source='memory',supporting_evidence_refs=sorted(refs))
        # Non-overlapping references keep their type/time, never an expanded copy of facts.
        if any(support):entry['other_support']=support
        history.setdefault('long_term_memory',[]).append(entry)
        if audit is not None:
            audit['memories'].append({k: deepcopy(memory[k])
                                      for k in ('memory_id', 'source_ids', 'evidence_times')})
    if history.get('long_term_memory'):
        sources['memory']={'kind':'derived_memory','owner':seller,'clock':'simulation_settled_step'}
    decision={'requested_item_ids':state.get('requested_price_item_ids',[p['item_id'] for p in current]),'history_end_exclusive':cutoff}
    requested=decision['requested_item_ids']
    require(len(requested)==len(set(requested)) and set(requested)<=owned_ids,
            'Requested product missing from own decision evidence')
    for field in ('decision_time','product_batch','price_scope'):
        if field in state:decision[field]=deepcopy(state[field])
    view={'schema':SCHEMA,'seller':deepcopy(state['profile']),'decision':decision,'sources':sources,
          'current_state':{'products':current},'history':history,'external_reference':external}
    if 'observed_shop_period' in state:
        require(state['observed_shop_period']['end_exclusive']==cutoff,'Shop period cutoff mismatch')
        view['current_state']['observed_shop_period']={'source':'observed',**deepcopy(state['observed_shop_period'])}
    if 'business_report' in state:
        view['current_state']['settled_shop_period']={'source':'settled',**deepcopy(state['business_report'])}
    if 'shop_scope' in state:view['current_state']['shop_scope']=deepcopy(state['shop_scope'])
    if audit is not None:
        audit['sources'] = deepcopy(sources)
    return view,audit
