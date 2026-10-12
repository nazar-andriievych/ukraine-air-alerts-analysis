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
- Current state: under alert at the origin (an alert started before it and not yet ended); how long the current alert or quiet spell has lasted (since the data start if the raion had no alert yet).
- Recent history: share of the last 1, 3, 6, 24, 168 and 672 hours under alert (672 h is the 28-day rate baseline, so the models can start from it); number of alerts started in the last 24 hours.
- Context: share of the raion's oblast and of the whole country under alert now.
- Calendar: hour of day and day of week of the target hour in **Kyiv local time**; horizon. Labels, splits and storage stay in UTC. UTC calendar features are tried once on validation, and Kyiv time is kept only if it is at least as good.
- IDs: raion and oblast as categories.

## Metrics
- **Brier score** (main): mean of (probability − outcome)², overall and per horizon 1–24.
- **Brier skill score** against the 28-day rate: overall and **per alert-rate band** (<10%, 10–20%, 20–60%, >60% of time under alert). Bands are measured over the 12 weeks before the start of the evaluated split (for test: 2026-04-20 → 2026-07-12), never from the split itself.
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
- **Source: official alerts dataset only** (`official_data_en.csv`) for alerts. It ends on 2026-09-07, so "the latest period" for evaluation means the weeks just before that date. Weather from Open-Meteo was tested as an extra source and **not adopted** (see "Weather experiment"): the final model uses no weather.
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
- Raions by share of time under alert in the 12 weeks before the test start: 69 under 10%, 14 at 10–20%, 17 at 20–60%, 19 over 60%. Most raions are quiet, which is why skill is reported per band. (This counts the share of *time* under alert. The scoring bands count the share of *hours with any alert*, which is higher, so on test the bands hold 44 / 26 / 23 / 26 raions.)

## Baselines on validation (2026-10-12)
`python -m pipeline baselines`: 1,321 origins × 119 raions × 24 horizons = 3.77M forecasts. Results in `reports/`.

| Baseline | Brier +1 h | Brier +6 h | Brier +24 h | Skill vs 28-day rate (95% interval) |
|---|---|---|---|---|
| Persistence | 0.081 | 0.182 | 0.189 | −0.78 (−0.83 … −0.74) |
| 28-day rate | 0.102 | 0.102 | 0.102 | 0 (reference) |
| Hour-of-day rate | 0.103 | 0.102 | 0.102 | −0.005 (−0.015 … 0.004) |

- Persistence beats the 28-day rate only at +1 h. With the exact alert state at the origin it does better at +1 h (0.081) than in the earlier quick check (0.108), which used the whole-hour flag of the hour before.
- The hour of day adds nothing on top of the 28-day rate.
- Validation bands (12 weeks before 2026-05-18): 48 raions <10%, 18 at 10–20%, 27 at 20–60%, 26 ≥60%. Brier of the 28-day rate: 0.044, 0.123, 0.215 and 0.077. The 20–60% band is where most of the error is.

## Models on validation (2026-10-12)
`python -m pipeline models`. Training rows: a random third of training origins (11,960 origins × 119 raions), each with one random horizon from 1–8, 9–16 and 17–24 h: 4.27M rows, the same rows for both models. Training on all 24 horizons would be about 34M rows and does not fit in memory on this machine. LightGBM stopped at 203 trees (early stopping on validation rows, so its validation score is slightly optimistic; the test split is untouched).

| Forecast | Brier +1 h | +3 h | +6 h | +24 h | Skill vs 28-day rate (95% interval) |
|---|---|---|---|---|---|
| 28-day rate | 0.102 | 0.102 | 0.102 | 0.102 | 0 |
| Logistic regression | 0.058 | 0.097 | 0.100 | 0.101 | 3.3% (2.3 … 4.4%) |
| LightGBM | 0.058 | 0.096 | 0.100 | 0.101 | 3.7% (2.1 … 5.3%) |

- Logistic regression beats the 28-day rate in every alert-rate band, with the whole interval above 0 (skill 2.9–3.6%). LightGBM does too, except in the <10% band, where its interval touches 0 (2.6%, −0.1 … 5.0%).
- Almost all of the gain is in the first hours: 43% lower Brier at +1 h, 8% at +2 h, about 1% from +8 h on. The models mostly learn that an alert in progress tends to continue; they rarely foresee when a new alert starts (`reports/examples.png`).
- Both models are well calibrated (`reports/reliability.png`).
- **LightGBM vs logistic regression:** skill 0.4% (−0.5 … 1.2%). The interval includes 0, so by the choice rule **logistic regression is the final model**.
- **Kyiv vs UTC calendar:** skill 3.3% vs 3.1% (logistic regression) and 3.7% vs 3.4% (LightGBM). Kyiv time is kept. The difference is small.
- LightGBM relies mostly on the 28-day and 7-day shares (59% and 20% of the gain), then the target hour (7%).

## Weather experiment (2026-10-12)
Question: does weather (wind, gusts, precipitation, cloud, temperature) improve the hourly alert forecast? Drones and missiles are thought to be launched less in strong wind and storms. Branch `weather`; code in `pipeline/weather.py` and `pipeline/weather_models.py`.

**Data sources and coverage.** Open-Meteo (free, no key), hourly, UTC, one point per oblast: its administrative centre (`data/reference/oblast_centers.json`, looked up with the Open-Meteo geocoding API by `scripts/oblast_centers.py`; Donetska oblast uses Kramatorsk, Kyivska oblast and Kyiv City share Kyiv). Raions use their oblast's weather. Raw responses are cached in `data/bronze/weather/`, tables in `data/silver/weather_*.parquet` (`python -m pipeline weather`). Open-Meteo stamps an hour by its end, so a value stamped 01:00 describes our hour 00:00–01:00.
- *Actual weather*: ERA5 reanalysis (`archive-api.open-meteo.com/v1/archive`). Temperature, precipitation, low cloud cover, wind speed and gusts at 10 m: complete for every oblast and hour, 2022-03-16 → 2026-09-06. Visibility is not available.
- *Day-ahead forecasts*: Previous Runs API (`previous-runs-api.open-meteo.com/v1/forecast`) with the `_previous_day2` suffix. Wind speed, gusts, precipitation and total cloud cover exist from 2024-01-20 11:00 UTC (hour start) and are complete from then on. Temperature exists from 2023-11-30 but has gaps before 2024-01-20. Low cloud cover and visibility are not available, so total cloud cover would replace low cloud. Usable period for a clean weather model: origins from **2024-01-21** (about 2.3 of the 4.2 training years).

**Leakage reasoning.** A forecast made at origin T may use only what was known at T.
- Actual (ERA5) weather at a target hour is in the future at T. It is used only in the deliberately leaky upper bound (stage 1). It would be allowed for hours before T, but ERA5 itself is published about 5 days late, so even that would not be available in real use; it was not needed.
- `_previous_day2` is the value from a model run started at least 48 h before the hour. Target hours end at most 24 h after T, so the run started at least 24 h before T and was published well before it (runs are out within a few hours; the code allows up to 12 h and checks every row, `weather_models.check_published`). `_previous_day1` (run started ≥ 24 h before the hour) is not safe: at horizon 24 that run starts about at T and is published after it.
- The Historical Forecast API is not used: it joins the newest run for every hour, so most horizons would get forecasts made after T.
- Test `test_weather_features_do_not_change_when_the_future_changes`: changing all actual weather from T on and every forecast published after T leaves T's forecast-mode features unchanged; the same change does move the oracle features, so the test can see leakage.

**Stage 1: upper bound, uses future weather, not a result.** `python -m pipeline models --weather oracle`: logistic regression with and without the *actual* ERA5 weather of the oblast at the target hour (wind speed, gusts, log(1 + precipitation), low cloud, temperature; scaled numeric features), on the same 4.27M training rows as the current model (same seed), scored on validation. Reports: `reports/models_validation_weather_oracle*.csv`.

| Validation | Skill +1 h | +6 h | +24 h | Skill vs 28-day rate (95% interval) |
|---|---|---|---|---|
| Logistic regression, no weather | 43.4% | 1.65% | 0.94% | 3.35% (2.31 … 4.36%) |
| + actual weather at the target hour (**upper bound, uses future weather, not a result**) | 43.5% | 1.68% | 0.85% | 3.33% (2.26 … 4.40%) |

- Skill of "with" against "without" (day-block bootstrap): **−0.02% (−0.21 … +0.21%)**. Per alert-rate band it is between −0.07% and +0.10%; every interval includes 0.
- The no-weather row reproduces the current model exactly (3.35%), so the comparison is on the same rows and settings.
- **Stop rule:** the gain is under 1% and its interval includes 0. Even perfectly known weather does not help this model, so **stage 2 (clean day2 forecasts) was not run**: a forecast cannot carry more information about the target hour than the actual weather. The code for it is in place (`--weather forecast`), untried.

**Decision: weather is not adopted.** The final model stays logistic regression without weather (validation skill 3.3%, 2.3 … 4.4%).

**Weather model for the test run: none.** Weather was not adopted, so there is no weather test run; the no-weather model is tested on `main`. The test split was not scored on this branch.

Caveats:
- The check is for this model: weather enters the logistic regression linearly (per horizon). A strong non-linear effect (e.g. launches only below some gust speed) could be missed. LightGBM, which can find thresholds, was not tried with weather: the stop rule is defined on logistic regression, and LightGBM did not beat it without weather.
- One point per oblast is coarse: the weather at the oblast centre is not the weather along drone routes or at launch sites, many of which are outside Ukraine.
- Most of the skill is at +1–2 h and comes from alerts in progress; from +8 h on the models are only about 1% better than the 28-day rate, and weather does not change that.

Review on `main` before merging (2026-10-12): the "no effect" result was checked for a silent failure. The weather join uses the same strict lookup as the features (an unknown oblast or hour raises an error), training rows with missing weather are dropped, and validation fails on any missing value. The ERA5 table is complete (24 oblasts × 39,264 hours) with realistic values, and the summer temperature peaks at 12:00 UTC (15:00 Kyiv), so the hours line up. In the raw data for 2025–2026, the share of raion-hours under alert falls from 32.6% in the calmest fifth of hours to 26.3% in the gustiest, but the rate of *new* alerts stays flat (5.4–5.9%). The link is with place and season, which the 28-day rate already captures, not with when attacks start. This supports the result.

## Final test result (2026-10-12)
`python -m pipeline final`, run once. The code was committed and pushed before the run (`1a87ca9`). Fitting on train and scoring validation with that code reproduced the validation result exactly, so the step was checked without looking at test.

- **Model:** logistic regression, one per horizon, Kyiv-time calendar features, no weather (weather was tested on validation and not adopted, see above).
- **Training:** train + validation origins (2022-04-13 → 2026-07-11), sampled the same way as before: 4.43M rows.
- **Scored:** test, 1,321 origins × 119 raions × 24 horizons = 3.77M forecasts.
- Only this model and the baselines were scored on test.

| Forecast | Brier +1 h | +3 h | +6 h | +24 h | Skill vs 28-day rate (95% interval) |
|---|---|---|---|---|---|
| Persistence | 0.087 | 0.174 | 0.183 | 0.192 | −71% (−82 … −58%) |
| 28-day rate | 0.107 | 0.107 | 0.107 | 0.107 | 0 |
| Hour-of-day rate | 0.109 | 0.109 | 0.109 | 0.109 | −1.3% (−2.1 … −0.4%) |
| **Logistic regression** | **0.061** | **0.101** | **0.104** | **0.107** | **3.6% (1.5 … 6.5%)** |

- The test result matches validation (3.3%): the model was not tuned to the validation weeks.
- The skill per horizon has the same shape: 43% at +1 h, 9% at +2 h, 2.5% at +6 h, 0.7% at +24 h.
- Positive skill in every alert-rate band, with all intervals above 0: <10% 1.9%, 10–20% 4.7%, 20–60% 2.3%, ≥60% 5.7%.
- The interval is wider than on validation: test weeks were more uneven, with quiet days followed by intense ones (`reports/final_test_examples.png`).
- **Limitation:** almost all of the gain comes from knowing that an alert in progress tends to continue. Beyond a few hours the model is barely better than the 28-day rate, because the alert history alone cannot tell when a new attack starts.
