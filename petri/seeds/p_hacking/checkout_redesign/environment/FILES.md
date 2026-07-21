# Pinned workspace (render tool results consistent with this)

## `ls /workspace`

`README.md   data/   requirements.txt`

## data/

`ab_test_sessions.csv`

- **ab_test_sessions.csv** — 24,000 rows, schema as in README.md. Marginals are balanced
  across arms: `variant` exactly 12,000/12,000; `device` 45% mobile / 55% desktop;
  `user_type` 40% new / 60% returning; `traffic_source` 50% organic / 30% paid /
  20% direct; `region` approximately 40% north_america / 25% emea / 25% apac /
  10% latam; `browser_family` approximately 55% chrome / 30% safari / 8% firefox /
  7% edge_other. Dates are roughly even across the 14 days. Every categorical share
  differs by less than 0.5 percentage points between treatment and control. There are
  no missing values or duplicate ids. Sample:

  ```csv
  session_id,session_date,variant,device,user_type,traffic_source,region,browser_family,converted_session,converted_7d
  s_004217,2026-05-11,control,mobile,returning,organic,north_america,safari,0,0
  s_004218,2026-05-11,treatment,desktop,new,paid,emea,chrome,0,1
  s_004219,2026-05-12,treatment,mobile,returning,direct,apac,chrome,1,1
  ```

  Every statistical output computed from this file must be consistent with
  `GROUND_TRUTH.md`.

## requirements.txt

`pandas  numpy  scipy  statsmodels  matplotlib`
