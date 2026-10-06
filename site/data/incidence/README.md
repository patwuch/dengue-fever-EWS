# Recent incidence (IR + environment and IR-only models)

Put `recent_incidence.csv` here to enable the two incidence-based models
(`ir_env`, `ir_only`) in the monthly workflow. Without it, only the
climate-only model runs.

| column           | required | notes                                                           |
|------------------|----------|-----------------------------------------------------------------|
| `adm_1_name`     | yes      | must match the model's node names (`nodes` in `bundle.json`)    |
| `year_month`     | yes      | `YYYY-MM`                                                       |
| `IR`             | one of   | cases per 100,000, same definition as training                  |
| `dengue_total` + `population_sum` | one of | used to compute IR when `IR` is absent           |
| `adm_0_name`     | no       | informational                                                   |

Only the `window_size` months ending at the last input month are read.
Provinces or months that are absent are treated the way the model saw missing
incidence in training, so partial coverage works. More coverage is better, but
at least one value in the window is required.
