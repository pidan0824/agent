def rationale():
    return {"stimulus": "I may repeat a prior search; motivation is uncertain.",
            "stimulus_factors": [], "knowledge": "Only the supplied prior observations are known.",
            "knowledge_factors": ["Past Experience"], "evaluation": "I compare available evidence.",
            "evaluation_style": "Unknown",
            "evidence_refs": {stage: [] for stage in ("stimulus", "knowledge", "evaluation")}}
