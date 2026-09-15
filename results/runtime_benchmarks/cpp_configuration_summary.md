| Build | Selection Method | Batch Size | Runs | Median Seconds | Min Seconds | Max Seconds | Iqr Seconds | Compiler | Executable Sha256 | Selected Main Configuration |
|---|---|---|---|---|---|---|---|---|---|---|
| native | bounded max-heap | 16 | 5 | 0.714798 | 0.713052 | 0.71942 | 0.0015161 | MSVC _MSC_VER=1935 _MSC_FULL_VER=193532216 | 52122e2d5264c0f894df409f356ff48a991fd33fcb7cc7e46d098db83f7fd34b | false |
| native | bounded max-heap | 32 | 5 | 0.656964 | 0.652392 | 0.680387 | 0.0055622 | MSVC _MSC_VER=1935 _MSC_FULL_VER=193532216 | 52122e2d5264c0f894df409f356ff48a991fd33fcb7cc7e46d098db83f7fd34b | true |
| native | bounded max-heap | 64 | 5 | 0.913397 | 0.909618 | 0.919632 | 0.0075957 | MSVC _MSC_VER=1935 _MSC_FULL_VER=193532216 | 52122e2d5264c0f894df409f356ff48a991fd33fcb7cc7e46d098db83f7fd34b | false |
| native | nth_element | 32 | 5 | 1.8764 | 1.81722 | 2.76758 | 0.700462 | MSVC _MSC_VER=1935 _MSC_FULL_VER=193532216 | 52122e2d5264c0f894df409f356ff48a991fd33fcb7cc7e46d098db83f7fd34b | false |
| portable | bounded max-heap | 16 | 5 | 1.14576 | 1.13984 | 1.15609 | 0.010378 | MSVC 1935 | 146d43918075d4ae5c3c70abca1e4379358aed308439e830b9059f57b766378d | false |
| portable | bounded max-heap | 32 | 5 | 1.25486 | 1.25149 | 1.28478 | 0.005263 | MSVC 1935 | 146d43918075d4ae5c3c70abca1e4379358aed308439e830b9059f57b766378d | false |
| portable | bounded max-heap | 64 | 5 | 1.36556 | 1.34644 | 1.57723 | 0.0057296 | MSVC 1935 | 146d43918075d4ae5c3c70abca1e4379358aed308439e830b9059f57b766378d | false |
| portable | nth_element | 32 | 5 | 2.58274 | 2.20842 | 3.05896 | 0.830215 | MSVC 1935 | 146d43918075d4ae5c3c70abca1e4379358aed308439e830b9059f57b766378d | false |
