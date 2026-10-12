# Problem statement

## Target
Forecast the **probability that a raion is under an air raid alert in each of the next 24 hours**.

- **Unit:** raion × UTC hour. Data: `data/gold/raion_hourly.parquet` (4.67M rows, 24% of hours under alert overall).
- **Label:** 1 if the raion was under alert at any moment in that hour, else 0. An oblast alert puts all of its raions under alert; a hromada alert puts its whole raion under alert.
- **Forecast origin:** the end of hour t (UTC), i.e. the full hour t is known. Only data available by then may be used, for features as well as training.
- **Horizon:** hours t+1 … t+24, the 24 hours right after the origin.

This replaces the earlier target (daily alert hours per raion for the next 7 days). The daily table `data/gold/raion_daily.parquet` is kept.

## Evaluation
- **Splits by time:**

  | Split | Period (UTC) | Used for |
  |---|---|---|
  | Train | 2022-03-16 → 2026-05-17 | fitting |
  | Validation | 2026-05-18 → 2026-07-12 (8 weeks) | choosing features and settings |
  | Test | 2026-07-13 → 2026-09-06 (8 weeks) | final score, used once |

- **No leakage across splits:** a training example at origin t has labels up to t+24 h, so examples whose labels fall past the end of their split are dropped.
- **Training origins:** a random third of all hours (not a fixed 3-hour grid), so every origin hour of the day appears in training.
- **Test origins:** every hour (rolling-origin backtest).
- **Uncertainty:** forecasts from neighbouring hours overlap heavily, so confidence intervals come from a block bootstrap over whole days, not over rows.

## Baselines
- **Persistence:** the state at the origin (0 or 1) for all 24 hours.
- **28-day rate:** the share of the last 28 days the raion was under alert. **This is the baseline to beat.**
- **Hour-of-day rate:** the same, counting only the target hour of the day.

## Models
- **Logistic regression** (scikit-learn): one model per horizon (24 models), so the effect of the current state can differ by horizon. Numeric features are scaled, raion/oblast/hour are one-hot encoded, the current spell length is capped at 48 h, default L2 penalty. Same training rows, features and test scoring as LightGBM.
- **LightGBM classifier:** one global model for all raions and horizons, log-loss objective.
- **Choice rule:** LightGBM is kept as the final model only if it beats logistic regression by more than the bootstrap uncertainty. Otherwise logistic regression is the final model.

**Features** (all computed at the origin):
- Current state: under alert now; how long the current alert or quiet spell has lasted.
- Recent history: share of the last 1, 3, 6, 24 and 168 hours under alert; number of alerts in the last 24 hours.
- Context: share of the raion's oblast and of the whole country under alert now.
- Calendar: hour of day and day of week of the target hour in **Kyiv local time**; horizon. Labels, splits and storage stay in UTC. UTC calendar features are tried once on validation, and Kyiv time is kept only if it is at least as good.
- IDs: raion and oblast as categories.

## Metrics
- **Brier score** (main): mean of (probability − outcome)², overall and per horizon 1–24.
- **Brier skill score** against the 28-day rate: overall and **per alert-rate band** (<10%, 10–20%, 20–60%, >60% of time under alert). Bands are fixed from the 12 weeks before the test start (2026-04-20 → 2026-07-12), never from the test period.
- **Log loss**, **calibration** (reliability diagram) and **ROC-AUC** (secondary).

Plots: Brier by horizon (models vs baselines), reliability diagram, probability vs actual alerts for 2–3 example raions, feature importance.

Libraries: pandas, numpy, scikit-learn, lightgbm, matplotlib.

## Decisions
- **Horizon: 24 hours**, hourly alert probability. Replaces the earlier 7-day daily-hours target.
- **Forecast unit: raion** (119 units: the raions of 23 oblasts plus Kyiv City, which has no raions and is a single unit). Luhansk and Crimea are excluded. The model is trained and evaluated on raion series, not on oblast aggregates.
- **Everything is in UTC, including day boundaries** (a day is 00:00–24:00 UTC, i.e. 02:00/03:00 Kyiv time). Every day is 24 h, so DST plays no role.
- **Reference hierarchy:** `data/reference/regions.json`, with English names transliterated from `regions_uk.json` by `scripts/translate_regions.py`.
- **A hromada alert counts as the whole raion being under alert.** This is accepted even though it puts Kharkivskyi and Chuhuivskyi raions at 24 h/day from mid-2024 to mid-2025.
- **No duration-based filtering of alerts.** See the findings below.
- **Data end: 2026-09-06** (the last complete day). Alerts still running at 2026-09-07 00:00 UTC are cut off there.
- **Data start: 2022-03-16** (the first complete day). The dataset starts at 2022-03-15 16:10 UTC.
- **Alert type (drones vs missiles) is not used.** Neither public dataset records it. Both store only the alert start and end.
- **The private 48h UkraineAlarm log is not used.**
- **Source: official dataset only** (`official_data_en.csv`). No other sources. It ends on 2026-09-07, so "the latest period" for evaluation means the weeks just before that date.
- **Oblasts: 24** (23 oblasts plus Kyiv City). Luhansk has no rows in 2026 because its alert is permanent, so it is excluded. Crimea is not in the dataset.

## Data findings that affect the target (2026-10-12)
- Official and volunteer data agree well in 2022 (daily hours corr 0.94, MAE 0.17 h) but drift apart from 2024 on.
- Since December 2025, official alerts are issued almost only at raion level. Volunteer daily hours stay flat at about 3.1 h, while the official union across all levels rises to about 8.9 h. The volunteer series likely undercounts in recent periods.
- Official data ends on 2026-09-07.
- A hromada alert does not imply a raion alert: only 45% of hromada alerts start inside a raion- or oblast-level alert.
- Since December 2025, oblast-level alerts exist **only for Kyiv City**. The other 23 oblasts get raion-level alerts only.
- Before December 2025, oblast and raion alerts mostly ran **at the same time**. A common pattern is an oblast alert ending at the exact minute alerts start in all of its raions, which looks like the same alert being re-issued per raion.
- **38% of rows (113,845 of 299,275) are exact duplicates.** They must be removed before counting alerts. Hours computed from merged intervals are not affected.
- **There are no stuck alerts in the data.** After de-duplication, 896 alerts are longer than 24 h, but none of them has another alert for the same unit starting inside it, so the records are internally consistent. 95% are in Donetska and Dnipropetrovska oblasts (Nikopol area), where near-continuous alerts are plausible.
- Two hromada alerts are extremely long: Lyptsi (Kharkivskyi raion, 604 days from 2024-05-12) and Vovchansk (Chuhuivskyi raion, 438 days from 2024-05-20). Both start during the May 2024 Kharkiv offensive, so they look like permanent frontline alerts, not errors.
- About 5.4% of alert time falls in 21:00–22:00 UTC, the hour whose day assignment changes between summer and winter time if days follow the Kyiv clock.

## Hourly findings (2026-10-12)
Measured on the hourly labels before modelling, to pick the baseline and features.
- In the test weeks (2026-07-13 → 2026-09-06), raions are under alert **34% of all hours**. The share barely depends on the hour of day (30–38%).
- Baselines on the test weeks (Brier, origins every 3 h): persistence 0.108 at +1 h but 0.183 on average; 28-day rate 0.107 on average; hour-of-day rate 0.109. Alerts switch on and off often, so persistence is only useful for the first hour or so.
- Alert start times (2025–2026) line up better between winter and summer on the **Kyiv clock** (correlation 0.81) than on the UTC clock (0.73). This is why calendar features use Kyiv time.
- Raions by share of time under alert in the 12 weeks before the test start: 69 under 10%, 14 at 10–20%, 17 at 20–60%, 19 over 60%. Most raions are quiet, which is why skill is reported per band.
