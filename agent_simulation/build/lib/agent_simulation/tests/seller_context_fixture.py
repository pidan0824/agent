"""Valid source-bearing state for prompt/projection tests."""
from agent_simulation.tests.data.fixtures import event
from agent_simulation.agents.context.pricing import build_pricing_context,overview_context

def seller_state():
    state,_=build_pricing_context([event(1,1,100),event(1,2,120),event(2,1,200)],
        owner_id=10,cutoff=10,observations_per_item=2,public_pages=[],requested_item_ids=[1],
        simulation_cutoff=4)
    return overview_context(state,[1,2])

def memory(content='Fixture experience', *, step=0):
    return dict(memory_id='seller:10:fixture',owner='seller:10',content=content,
        simulation_step=step,evidence_kind='model_summary_or_hypothesis',
        source_ids=['fixture:observed:1'],evidence_times=[{'origin':'observed','time_index':1}])


def decision_context(state):
    from agent_simulation.agents.context.pricing_evidence import project_seller
    return project_seller(state, include_audit=False)[0]
