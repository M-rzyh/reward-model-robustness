Best-of-8 on 500 HH-test prompts (pool sha1 8afc0922); mean ± sample std over pairings / seeds.

| pairing | pairs | flip rate (%) | Kendall τ |
|---|---|---|---|
| within_hh | 10 | 41.1 ± 3.6 | 0.702 ± 0.020 |
| within_uf | 10 | 38.3 ± 2.5 | 0.649 ± 0.027 |
| cross_hh_uf | 25 | 64.9 ± 3.5 | 0.392 ± 0.032 |
| two independent random pickers | – | 87.5 | 0 |

| flip-rate gap (percentage points) | mean | 95% CI (bootstrap over prompts) |
|---|---|---|
| cross minus within hh | 23.7 | [20.6, 27.1] |
| cross minus within uf | 26.5 | [23.2, 29.5] |
| cross minus within both | 25.1 | [22.6, 27.7] |

| reward model | seeds | selected length (tokens) | picks the longest (%) |
|---|---|---|---|
| trained on hh | 5 | 208.0 ± 6.0 | 30.6 ± 2.4 |
| trained on uf | 5 | 205.5 ± 5.3 | 25.9 ± 1.7 |
| pool mean (all 4000) | – | 158.2 | – |
| always pick the longest | – | 274.0 | 100 |
| random pick (chance) | – | 158.2 | 12.5 |
