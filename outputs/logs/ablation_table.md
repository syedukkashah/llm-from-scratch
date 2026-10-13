### sampling (no retraining)

| setting | real-word rate | repetition | distinct samples (of 5) |
|---|---|---|---|
| greedy (temp 0) | 1.0 | 0.515 | 1 |
| temp 0.5 | 0.983 | 0.033 | 5 |
| temp 0.8 | 0.97 | 0.007 | 5 |
| temp 1.2 | 0.859 | 0.0 | 5 |
| temp 0.8 + top-k 10 | 0.964 | 0.01 | 5 |
