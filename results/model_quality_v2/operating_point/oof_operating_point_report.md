# Frozen V2 operating points: training-only OOF evidence

The frozen k=101 Model Quality V2 score model was evaluated with five-fold stratified OOF prediction inside the original 24,000-row training partition. Every row was scored once by a model that did not train on it. The legacy test was not used to choose a constraint, objective, candidate threshold, or tie-break.

The decision rule is `class 1 if probability_class_1 >= threshold`. The current reference threshold is 0.280150124169 and was read from the already-frozen final V2 configuration rather than reselected.

## Frozen operating points

| operating_point         |   threshold |          FP |          FN |          TP |           TN |   accuracy |   precision |   recall |       f1 |     f0_5 |   balanced_accuracy |   predicted_positive_rate |
|:------------------------|------------:|------------:|------------:|------------:|-------------:|-----------:|------------:|---------:|---------:|---------:|--------------------:|--------------------------:|
| current_v2_threshold    |    0.280150 | 2569.000000 | 2399.000000 | 2910.000000 | 16122.000000 |   0.793000 |    0.531119 | 0.548126 | 0.539488 | 0.534435 |            0.705340 |                  0.228292 |
| min_fp_recall_50        |    0.331541 | 2033.000000 | 2654.000000 | 2655.000000 | 16658.000000 |   0.804708 |    0.566340 | 0.500094 | 0.531159 | 0.551723 |            0.695663 |                  0.195333 |
| min_fp_f1_52            |    0.382471 | 1605.000000 | 2878.000000 | 2431.000000 | 17086.000000 |   0.813208 |    0.602329 | 0.457902 | 0.520278 | 0.566587 |            0.686016 |                  0.168167 |
| min_fp_accuracy_guard   |    0.356009 | 1818.000000 | 2760.000000 | 2549.000000 | 16873.000000 |   0.809250 |    0.583696 | 0.480128 | 0.526871 | 0.559556 |            0.691431 |                  0.181958 |
| max_precision_recall_50 |    0.331541 | 2033.000000 | 2654.000000 | 2655.000000 | 16658.000000 |   0.804708 |    0.566340 | 0.500094 | 0.531159 | 0.551723 |            0.695663 |                  0.195333 |
| f0_5_optimized          |    0.440138 | 1263.000000 | 3079.000000 | 2230.000000 | 17428.000000 |   0.819083 |    0.638420 | 0.420041 | 0.506703 | 0.578290 |            0.676234 |                  0.145542 |

The independently predeclared objectives that converged to one numerical threshold were: min_fp_recall_50, max_precision_recall_50.

ROC-AUC and Average Precision describe the shared continuous score ranking and therefore do not vary by threshold.

## Deltas versus the current V2 threshold

| operating_point         |     delta_FP |    delta_FN |    delta_TP |     delta_TN |   delta_accuracy |   delta_precision |   delta_recall |   delta_f1 |   delta_f0_5 |
|:------------------------|-------------:|------------:|------------:|-------------:|-----------------:|------------------:|---------------:|-----------:|-------------:|
| current_v2_threshold    |    +0.000000 |   +0.000000 |   +0.000000 |    +0.000000 |        +0.000000 |         +0.000000 |      +0.000000 |  +0.000000 |    +0.000000 |
| min_fp_recall_50        |  -536.000000 | +255.000000 | -255.000000 |  +536.000000 |        +0.011708 |         +0.035221 |      -0.048032 |  -0.008329 |    +0.017287 |
| min_fp_f1_52            |  -964.000000 | +479.000000 | -479.000000 |  +964.000000 |        +0.020208 |         +0.071210 |      -0.090224 |  -0.019210 |    +0.032152 |
| min_fp_accuracy_guard   |  -751.000000 | +361.000000 | -361.000000 |  +751.000000 |        +0.016250 |         +0.052577 |      -0.067998 |  -0.012618 |    +0.025120 |
| max_precision_recall_50 |  -536.000000 | +255.000000 | -255.000000 |  +536.000000 |        +0.011708 |         +0.035221 |      -0.048032 |  -0.008329 |    +0.017287 |
| f0_5_optimized          | -1306.000000 | +680.000000 | -680.000000 | +1306.000000 |        +0.026083 |         +0.107301 |      -0.128084 |  -0.032785 |    +0.043854 |

Fewer false positives usually come with fewer true positives and lower recall. Each constrained operating point is preferred only when its stated objective and constraints match the intended use. None is universally superior.
