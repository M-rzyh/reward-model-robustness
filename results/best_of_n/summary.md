Best-of-8 on 500 HH-test prompts (pool sha1 8afc0922); mean ± sample std over pairings / seeds.

| pairing | pairs | flip rate (%) | Kendall τ |
|---|---|---|---|
| within_hh | 3 | 42.1 ± 4.5 | 0.697 ± 0.035 |
| within_uf | 3 | 38.0 ± 1.0 | 0.635 ± 0.012 |
| cross_hh_uf | 9 | 66.7 ± 3.6 | 0.374 ± 0.031 |
| two independent random pickers | – | 87.5 | 0 |

| flip-rate gap (percentage points) | mean | 95% CI (bootstrap over prompts) |
|---|---|---|
| cross minus within hh | 24.7 | [21.2, 28.6] |
| cross minus within uf | 28.7 | [24.4, 32.6] |
| cross minus within both | 26.7 | [23.5, 29.8] |

| reward model | seeds | selected length (tokens) | picks the longest (%) |
|---|---|---|---|
| trained on hh | 3 | 203.7 ± 1.6 | 28.9 ± 0.6 |
| trained on uf | 3 | 204.6 ± 5.9 | 25.7 ± 2.2 |
| pool mean (all 4000) | – | 158.2 | – |
| always pick the longest | – | 274.0 | 100 |
| random pick (chance) | – | 158.2 | 12.5 |
