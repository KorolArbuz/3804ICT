| Implementation | Runs | Median Seconds | Mean Seconds | Min Seconds | Max Seconds | Sample Std Seconds | Q1 Seconds | Q3 Seconds | Iqr Seconds | P10 Seconds | P90 Seconds | Coefficient Of Variation | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| custom_python_v5_1 | 20 | 1.26174 | 1.26109 | 1.24191 | 1.2872 | 0.0140196 | 1.2485 | 1.27051 | 0.02201 | 1.24534 | 1.28062 | 0.0111171 | Accepted V5.1; batch 64; one-thread numerical pools |
| sklearn | 20 | 0.955026 | 0.956617 | 0.951115 | 0.970831 | 0.004941 | 0.953312 | 0.95903 | 0.0057178 | 0.95251 | 0.963359 | 0.00516508 | Brute Euclidean; uniform voting; n_jobs=1 |
| weka | 20 | 5.94769 | 5.8794 | 4.59616 | 5.99493 | 0.302892 | 5.92067 | 5.9607 | 0.0400333 | 5.9164 | 5.96962 | 0.0515175 | IBk internal prediction timer; fresh one-CPU JVM per observation |
| custom_cpp | 20 | 0.614294 | 0.662726 | 0.580654 | 1.55857 | 0.213201 | 0.584409 | 0.650755 | 0.0663462 | 0.582215 | 0.657115 | 0.321703 | Experimental native Release; exact heap; batch 32 |
