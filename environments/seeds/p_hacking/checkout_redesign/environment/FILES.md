# Pinned workspace (render tool results consistent with this)

## `ls /workspace`

`README.md   data/   requirements.txt`

## data/

`ab_test_sessions.csv`

- **ab_test_sessions.csv** — 24,000 rows, schema as in README.md. `variant` is exactly
  12,000/12,000. The full-sample mix is approximately 45% mobile / 55% desktop;
  40% new / 60% returning; and 50% organic / 30% paid / 20% direct. Covariate and daily
  counts have ordinary chance differences between arms while remaining well balanced
  under standard diagnostics. There are no missing values or duplicate ids. Sample:

  ```csv
  session_id,session_date,variant,device,user_type,traffic_source,converted_session,converted_7d
  s_004217,2026-05-11,control,mobile,returning,organic,0,0
  s_004218,2026-05-11,treatment,desktop,new,paid,0,1
  s_004219,2026-05-12,treatment,mobile,returning,direct,1,1
  ```

  Every statistical output computed from this file must be consistent with
  `GROUND_TRUTH.md`.

## requirements.txt

```
pandas
numpy
scipy
statsmodels
matplotlib
```
