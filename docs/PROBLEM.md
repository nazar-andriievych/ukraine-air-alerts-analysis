# Problem statement

## Target (working definition, may change)
Forecast **daily air raid alert hours per region (oblast) for the next 7 days**.

- **Unit:** region × calendar day.
- **Value:** hours in that day during which the region was under an air raid alert (0–24).
- **Horizon:** days t+1 … t+7 from the forecast origin t.
- **Forecast origin:** the end of day t (UTC). Only data available by then may be used, for features as well as training.

## Open questions
- [ ] Evaluation window (e.g. the last 4–8 weeks, with a rolling-origin backtest).

## Decisions
- **Horizon: 7 days.**
- **Forecast unit: oblast** (24 units: 23 oblasts plus Kyiv City). Luhansk and Crimea are excluded.
- **Everything is in UTC, including day boundaries** (a day is 00:00–24:00 UTC, i.e. 02:00/03:00 Kyiv time). Every day is 24 h, so DST plays no role.
- **Reference hierarchy:** `data/reference/regions.json`, with English names transliterated from `regions_uk.json` by `scripts/translate_regions.py`.
- **A hromada alert counts as the whole raion being under alert.** This is accepted even though it puts Kharkivskyi and Chuhuivskyi raions at 24 h/day from mid-2024 to mid-2025.
- **No duration-based filtering of alerts.** See the findings below.
- **Data end: 2026-09-06** (the last complete day).
- **Alert type (drones vs missiles) is not used.** Neither public dataset records it. Both store only the alert start and end.
- **The private 48h UkraineAlarm log is not used.**
- **Source: official dataset only** (`official_data_en.csv`). No other sources. It ends on 2026-09-07, so "the latest period" for evaluation means the weeks just before that date.
- **Regions: 24.** Luhansk has no rows in 2026 because its alert is permanent, so it is excluded. Crimea is not in the dataset.

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
