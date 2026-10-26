# FCC vs performance gap: NIST datasets

| Method           | Spearman          | Pearson           | Weighted Kendall |
|------------------|-------------------|-------------------|------------------|
| OTDD †           | 0.410             | —                 | —                |
| s-OTDD †         | 0.400             | 0.430             | —                |
| FCC-pixels mean  | 0.133 (p = 0.576) | 0.151 (p = 0.524) | 0.124 (p n/a)    |
| FCC-pixels ratio | 0.280 (p = 0.232) | 0.269 (p = 0.251) | 0.111 (p n/a)    |

† Taken from the paper (not recomputed); "—" = not reported there. scipy.stats.weightedtau returns no p-value.
