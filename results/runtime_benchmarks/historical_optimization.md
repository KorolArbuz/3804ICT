| Version | Main Change | Accepted | Measurement Type | Trial Count | Median Runtime Seconds | Min Seconds | Max Seconds | Speedup Vs Original | Notes |
|---|---|---|---|---|---|---|---|---|---|
| Original | Initial custom NumPy implementation | true | historical_single_or_unspecified |  | 33.2812 | 33.2812 | 33.2812 | 1 | Historical development point |
| V1 | Matrix distances and partial selection | true | historical_five_run_warm_median | 5 | 1.98031 | 1.95058 | 2.0491 | 16.806 | Historical development point |
| V2 | Workspace reuse and lower allocation overhead | true | historical_five_run_warm_median | 5 | 1.50517 | 1.49646 | 1.51212 | 22.1112 | Historical development point |
| V3 | Ranking-only score and less unnecessary ordering | true | fresh_interleaved_five_run_warm_median | 5 | 1.41906 | 1.4147 | 1.42606 | 23.4529 | Historical development point |
| V4 | Exact deterministic threshold prefilter | true | fresh_interleaved_nine_run_warm_median | 9 | 0.812958 | 0.808413 | 0.825906 | 40.9384 | Historical development point |
| V5 | Augmented GEMM and batch 64 | true | fresh_interleaved_nine_run_warm_median | 9 | 0.754123 | 0.73007 | 0.763707 | 44.1323 | Historical development point |
| V5.1 | Feature-major direct fallback layout | true | fresh_interleaved_nine_run_warm_median | 9 | 1.2781 | 0.677609 | 1.34903 | 26.0395 | Accepted history point |
| V6 | Partial-distance lower-bound screening | false | rejected experimental paired benchmark | 2 | 1.16799 | 1.16793 | 1.16806 | 28.4943 | The best exact raw-feature lower-bound candidate was about 1.70x slower than paired V5.1, far below the required stable 10% speedup. |
| V7 | Exact block pruning | false | rejected experimental paired benchmark | 9 | 1.84059 | 0.923009 | 1.88712 | 18.0818 | The best exact block candidate achieved only a 0.6809x median paired V5.1/V7 timing ratio and was about 1.47x slower than V5.1; it did not meet the required stable 10% speedup. |
| C++ experimental | Fused exact distance accumulation and bounded heap | false | current controlled 20-run median | 20 | 0.614294 | 0.580654 | 1.55857 | 54.1779 | Experimental candidate; exact output, one-machine evidence |
