<!-- Template. `run_eval.py` fills a dated copy under data/eval/. Do not hand-edit numbers here. -->
# ad fontes — evaluation report

- **Run:** {{run_id}} ({{timestamp}})
- **Corpus version:** {{corpus_version}}
- **Index size:** {{index_size}} chunks
- **Commit:** {{git_sha}}

## Retrieval (Phase 1)
| metric | value |
|---|---|
| hit@6 (chunk-level) | {{hit_at_6_chunk}} |
| hit@6 (file-level) | {{hit_at_6_file}} |
| negative-control correct declines | {{neg_control_decline_rate}} |

## Generation + verification (Phase 2 / 5)
| metric | base | tuned |
|---|---|---|
| citation hit rate | {{cite_hit_base}} | {{cite_hit_tuned}} |
| supported | {{supported_base}} | {{supported_tuned}} |
| unsupported | {{unsupported_base}} | {{unsupported_tuned}} |
| contradicted | {{contradicted_base}} | {{contradicted_tuned}} |
| fabricated_citation | {{fab_base}} | {{fab_tuned}} |
| **unsupported + fabricated / 100 answers** | {{headline_base}} | {{headline_tuned}} |
| numeric-guard violations / 100 | {{numeric_base}} | {{numeric_tuned}} |
| p50 / p95 latency (2 vCPU) | {{p50_base}} / {{p95_base}} | {{p50_tuned}} / {{p95_tuned}} |
| judge win rate (tuned vs base) | — | {{judge_win_rate}} |

## Variance
{{variance_notes}}
