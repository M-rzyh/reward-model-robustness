Best-of-8 on 500 HH-test prompts (pool sha1 8afc0922); mean ± sample std over pairings / seeds.

| pairing | pairs | flip rate (%) | Kendall τ |
|---|---|---|---|
| within_hh | 10 | 40.0 ± 3.1 | 0.715 ± 0.019 |
| within_uf | 10 | 37.6 ± 2.8 | 0.659 ± 0.030 |
| cross_hh_uf | 25 | 64.2 ± 3.5 | 0.397 ± 0.032 |
| two independent random pickers | – | 87.5 | 0 |

| flip-rate gap (percentage points) | mean | 95% CI (bootstrap over prompts) |
|---|---|---|
| cross minus within hh | 24.1 | [21.2, 27.3] |
| cross minus within uf | 26.6 | [23.1, 29.6] |
| cross minus within both | 25.4 | [22.8, 27.9] |

| reward model | seeds | selected length (tokens) | picks the longest (%) |
|---|---|---|---|
| trained on hh | 5 | 209.5 ± 5.8 | 30.8 ± 2.7 |
| trained on uf | 5 | 205.7 ± 4.5 | 26.4 ± 1.3 |
| pool mean (all 4000) | – | 158.2 | – |
| always pick the longest | – | 274.0 | 100 |
| random pick (chance) | – | 158.2 | 12.5 |
