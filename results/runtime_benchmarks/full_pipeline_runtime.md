| Implementation | Runs | Median Seconds | Mean Seconds | Min Seconds | Max Seconds | Sample Std Seconds | Q1 Seconds | Q3 Seconds | Iqr Seconds | P10 Seconds | P90 Seconds | Coefficient Of Variation | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| custom_python_v5_1 | 5 | 1.39655 | 1.40284 | 1.39031 | 1.43337 | 0.0174217 | 1.39409 | 1.39987 | 0.0057807 | 1.39182 | 1.41997 | 0.0124189 | Prepared input loading, fit/build, prediction, output, and metrics |
| sklearn | 5 | 1.05596 | 1.05712 | 1.05506 | 1.06203 | 0.00282118 | 1.05569 | 1.05686 | 0.0011626 | 1.05531 | 1.05996 | 0.00266874 | Prepared input loading, fit/build, prediction, output, and metrics |
| weka | 5 | 14.8416 | 14.8333 | 14.7998 | 14.869 | 0.0299953 | 14.8048 | 14.8511 | 0.0463402 | 14.8018 | 14.8619 | 0.00202216 | Prepared input loading, fit/build, prediction, output, and metrics |
| custom_cpp | 5 | 0.843516 | 1.01911 | 0.780341 | 1.78847 | 0.431125 | 0.822956 | 0.860261 | 0.0373052 | 0.797387 | 1.41719 | 0.42304 | Prepared input loading, fit/build, prediction, output, and metrics |
