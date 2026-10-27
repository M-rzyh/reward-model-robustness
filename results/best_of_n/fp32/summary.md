Best-of-8 on 500 HH-test prompts (pool sha1 8afc0922); mean ± sample std over pairings / seeds.

| pairing | pairs | flip rate (%) | Kendall τ |
|---|---|---|---|
| within_hh | 3 | 41.8 ± 4.3 | 0.709 ± 0.034 |
| within_uf | 3 | 37.9 ± 0.9 | 0.645 ± 0.016 |
| cross_hh_uf | 9 | 66.2 ± 3.7 | 0.378 ± 0.029 |
| two independent random pickers | – | 87.5 | 0 |

| flip-rate gap (percentage points) | mean | 95% CI (bootstrap over prompts) |
|---|---|---|
| cross minus within hh | 24.4 | [20.8, 28.1] |
| cross minus within uf | 28.3 | [24.0, 32.2] |
| cross minus within both | 26.3 | [23.3, 29.5] |

| reward model | seeds | selected length (tokens) | picks the longest (%) |
|---|---|---|---|
| trained on hh | 3 | 205.4 ± 2.0 | 29.1 ± 1.4 |
| trained on uf | 3 | 204.6 ± 5.2 | 26.3 ± 1.0 |
| pool mean (all 4000) | – | 158.2 | – |
| always pick the longest | – | 274.0 | 100 |
| random pick (chance) | – | 158.2 | 12.5 |
