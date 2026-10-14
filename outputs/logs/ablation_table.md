### depth (n_layers)

| value | params | train loss | val loss | perplexity | train time (s) | tokens/s | peak mem (MB) |
|---|---|---|---|---|---|---|---|
| 2 | 1,613,377 | 1.3877 | 1.6221 | 5.063 | 69.4 | 235982 | 399 |
| 4 | 3,192,897 | 1.2907 | 1.5501 | 4.712 | 157.0 | 104335 | 714 |
| 6 | 4,772,417 | 1.2367 | 1.5221 | 4.582 | 235.9 | 69466 | 1028 |

### heads (n_heads)

| value | params | train loss | val loss | perplexity | train time (s) | tokens/s | peak mem (MB) |
|---|---|---|---|---|---|---|---|
| 2 | 3,192,897 | 1.3001 | 1.5472 | 4.698 | 152.0 | 107798 | 682 |
| 4 | 3,192,897 | 1.2907 | 1.5501 | 4.712 | 157.0 | 104335 | 714 |
| 8 | 3,192,897 | 1.288 | 1.5491 | 4.707 | 172.7 | 94869 | 778 |

### context (context_length)

| value | params | train loss | val loss | perplexity | train time (s) | tokens/s | peak mem (MB) |
|---|---|---|---|---|---|---|---|
| 64 | 3,192,897 | 1.3634 | 1.589 | 4.899 | 73.9 | 110923 | 368 |
| 128 | 3,192,897 | 1.2907 | 1.5501 | 4.712 | 157.0 | 104335 | 714 |
| 256 | 3,192,897 | 1.2381 | 1.5229 | 4.586 | 348.5 | 94031 | 1489 |

### position (position)

| value | params | train loss | val loss | perplexity | train time (s) | tokens/s | peak mem (MB) |
|---|---|---|---|---|---|---|---|
| sinusoidal | 3,192,897 | 1.2907 | 1.5501 | 4.712 | 157.0 | 104335 | 714 |
| learned | 3,225,665 | 1.2995 | 1.5419 | 4.674 | 158.2 | 103586 | 714 |

### sampling (no retraining)

| setting | real-word rate | repetition | distinct samples (of 5) |
|---|---|---|---|
| greedy (temp 0) | 1.0 | 0.515 | 1 |
| temp 0.5 | 0.983 | 0.033 | 5 |
| temp 0.8 | 0.97 | 0.007 | 5 |
| temp 1.2 | 0.859 | 0.0 | 5 |
| temp 0.8 + top-k 10 | 0.964 | 0.01 | 5 |
