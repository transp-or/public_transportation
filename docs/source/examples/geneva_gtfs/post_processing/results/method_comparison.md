# Geneva estimation-method comparison

All methods use the same timetable, observations, fixed theta, OD support, and frozen-cell layout.

| Method | Runtime (s) | Relative runtime | OD RMSE | Link RMSE | Link correlation | 90% coverage | Mean 90% width |
|:--|--:|--:|--:|--:|--:|--:|--:|
| ML | 112.7 | 2.7x | 12.372 | 0.231 | 0.999951 | — | — |
| MAP | 41.6 | 1.0x | 15.912 | 0.282 | 0.999910 | — | — |
| VI | 590.1 | 14.2x | 16.026 | 0.370 | 0.999807 | 64.6% | 21.437 |

ML reached its configured iteration cap; MAP satisfied the optimizer termination criterion; VI completed its fixed 1,000-step schedule.
The prior is intentionally inaccurate, so this benchmark is a stress test of regularization rather than a favorable MAP/VI setup.
