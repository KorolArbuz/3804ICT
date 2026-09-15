| Implementation | Category | Bytes | Mib | Basis |
|---|---|---|---|---|
| custom_python_v5_1 | persistent_model_storage | 19592200 | 18.6846 | Measured NumPy array nbytes after fit |
| custom_python_v5_1 | major_prediction_workspace | 13409408 | 12.7882 | Estimated major arrays; batch workspace, neighbours, and one fallback vector |
| custom_python_v5_1 | returned_probability_buffer | 96000 | 0.0915527 | Estimated float64 probability result |
| sklearn | persistent_model_storage | 6432016 | 6.13405 | Measured _fit_X, encoded labels, and classes nbytes |
| custom_cpp | persistent_model_storage | 6432000 | 6.13403 | Estimated vector capacities for training doubles and int labels |
| custom_cpp | major_prediction_workspace | 19200 | 0.0183105 | Executable estimate for native heap/batch 32 excluding result buffers |
| custom_cpp | result_buffers | 72000 | 0.0686646 | Estimated int labels plus size_t vote counts |
