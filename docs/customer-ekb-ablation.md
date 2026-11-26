# Customer EKB controlled evaluation

This is an evaluation-only alternative to training-data curation. It does not
export SFT/DPO rows or select using observed behavior. No generation has been run
to validate efficacy of this new protocol.

## Protocol

The Response task receives a frozen recorded Query, the latest 10 prior history
events, and the same candidates in every arm. Query generation remains available
through the existing independent Query workflow and is not replaced by EKB.
Candidate serialization uses item ID, with no claim that exposure positions are
known. Clicks and purchases remain separate; either or both may be empty, and
full exposures retain multiple choices.

| Arm | Prediction |
| --- | --- |
| A | Prespecified direct sample 0 |
| B | Prespecified simplified-EKB sample 0 |
| C | Plurality of complete joint click/purchase actions from N EKB samples |
| A10 | Same plurality rule from N direct samples; N defaults to 10 |

C/A10 vote on entire action sets, not independently on each item/head. A tie
uses the earliest sample index. Invalid outputs do not vote; all-invalid is a
failure, not a successful no-click prediction. A/B never replace a failed sample
0 with a later valid sample. The selector receives generated records only.
Choices are saved and hashed before labels enter the evaluator.

B minus A tests this simplified EKB prompt under single sampling; C minus B tests
EKB sampling/aggregation. C minus A10 helps separate EKB from sampling effects.
Equal sample counts and output ceilings do not imply equal actual compute.
Use the existing per-call streaming log for input/output tokens, TTFT, decode rate
and total time; report costs alongside behavior metrics. First-sample subsets
share randomness with the larger pools and are not independent experiments.

The simplified EKB contains stimulus, knowledge and evaluation, followed by the
action. It asks for relevant observed history, candidate comparison, and explicit
uncertainty. It neither requires fact IDs nor forces factors, personality or
internal motives. It is a task adaptation, not the author's original template.
Do not interpret a syntactically valid explanation as factually verified.

Default evaluation uses full real exposures. Supplying --scenes enables the
existing single-positive split shortlists, which are constructed using outcomes.
Even with a label-blind selector, those are conditional diagnostic tasks, not
an unbiased full-exposure test. Reports distinguish split_reconstructed from
full_exposure and merge sibling actions at the original-request level.

Metrics include per-request and macro click/purchase precision, recall, F1 and
exact match. Invalid/incomplete requests contribute zero; recall is undefined
for empty true sets. Vote frequencies are not calibrated probabilities, so this
entry point does not fabricate Log Loss from voting or self-confidence.
Optional D is saved separately as a label-assisted any-correct behavior ceiling.
It does not check semantic reliability and therefore is not an accepted-data
count or test score. No training export occurs.

## Usage (only when generation is authorized)

From the repository with the existing inference service running:

    python -m agent_simulation.evaluation.customer_ekb_ablation \
      --dataset /absolute/path/to/release \
      --output result/customer-ekb-controlled-NEW \
      --count 10 --samples 10 \
      --model Qwen3.5-9B --tokenizer models/Qwen3.5-9B \
      --url http://127.0.0.1:8011/v1

Defaults: split=test, seed=42, temperature=0.7, concurrency=2, history=10,
max-new-tokens=2048, context-limit=16384, streamed timing. No word-count limit.
Each scene costs 2N requests, not 2+2N; A/B reuse prespecified index 0.
For 32B, set both model and tokenizer explicitly. No server is launched by this
entry point. Full inputs that exceed capacity are failures, not silently shortened.
Existing test data/checkpoints have historical model-selection exposure; label
a new run diagnostic unless a truly untouched test partition is established.

## Interpretation and next diagnostic

Do not select a prompt, voting rule, checkpoint or sample count using the final
test outcomes. Freeze these using a development partition grouped by original
request. Shortlist accuracy alone cannot prove full-exposure improvements.
Small previews check plumbing, not efficacy; a later evaluation needs paired
request-level uncertainty intervals and enough click/purchase/no-action cases.

Joint generation is insufficient for a causal claim about explanation use.
Action agreement/diversity is only descriptive. For an intervention diagnostic,
hold the evidence and action-prediction model/configuration fixed, remove the
original action from each EKB, and replay:
1. no explanation;
2. each original explanation;
3. the same explanation with a specifically annotated unsupported claim removed.

Annotations/edit rules must be label-blind. Do not rewrite an explanation to
justify the observed target. Include a neutral edit control when comparing
deletions, since length/format changes can matter. Record changes, actions and
per-request effects. Unchanged actions do not prove explanations were ignored;
changed actions do not establish that the influence was useful or factual.

This replay/intervention diagnostic is specified here but is not yet implemented.
The current entry point generates joint simplified explanations/actions only.
