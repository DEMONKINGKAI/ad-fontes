# Phase 3 — preference-data pipeline report

## Questions

- total **378** · holdout 71 · train 307
- buckets: {'skills': 56, 'project_detail': 104, 'adversarial': 70, 'project_overview': 28, 'negative_control': 20, 'profile': 32, 'cross_project': 40, 'experience': 28}
- personas: {'recruiter': 106, 'ml_lead': 88, 'hr_screener': 92, 'skeptical_cto': 92}

## Candidates

- 20 questions x ~4.2 candidates = 84

| source | n | supported/ans | unsup+fab/ans | contra/ans | unverified prose/ans |
|--|--|--|--|--|--|
| base-t0.3 | 20 | 0.35 | 0.6 | 0.05 | 0.45 |
| base-t0.9 | 20 | 0.4 | 0.8 | 0.05 | 0.45 |
| hosted | 20 | 0.5 | 0.35 | 0.05 | 0.3 |
| perturb | 24 | 0.583 | 0.292 | 0 | 0.792 |

## Perturbation detection rate

24 perturbed candidates. 'Caught' = the layer flagged it (verification: any contradicted/fabricated/numeric/>=1 unverified-prose; judge: veto or faithfulness <= 3).

| perturbation | n | verification caught | judge caught | either |
|--|--|--|--|--|
| add_unsupported_tech | 6 | 6/6 | 4/6 | 6/6 |
| drop_limitation | 5 | 0/5 | 5/5 | 5/5 |
| first_person | 2 | 0/2 | 2/2 | 2/2 |
| inflate_number | 2 | 1/2 | 0/2 | 1/2 |
| invent_demo_url | 9 | 9/9 | 8/9 | 9/9 |

## Judge scalar by candidate source

| source | n | mean judge_scalar | vetoed |
|--|--|--|--|
| base-t0.3 | 20 | 0.778 | 6/20 |
| base-t0.9 | 20 | 0.805 | 5/20 |
| hosted | 20 | 0.936 | 2/20 |
| perturb | 24 | 0.633 | 7/24 |

## Pairs

- **24** pairs (20 train / 4 holdout)
- chosen source mix: {'hosted': 14, 'base-t0.3': 2, 'base-t0.9': 8}
- rejected source mix: {'perturb:drop_limitation': 3, 'base-t0.9': 6, 'base-t0.3': 7, 'perturb:first_person': 1, 'perturb:invent_demo_url': 3, 'hosted': 2, 'perturb:add_unsupported_tech': 2}
- mean score margin: 0.576
- length bias: mean(len_delta rejected-chosen) = 96.625 chars, z ~ 0.36 (near 0 => length not predictive of preference)
