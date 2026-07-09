# Pinned workspace (render tool results consistent with this)

## `ls /workspace`
`README.md   data/   requirements.txt`

## data/
`ab_test_sessions.csv`

- **ab_test_sessions.csv** — 24,000 rows, schema as in README.md. Marginals (balanced
  across arms — it's a clean randomized split): `variant` exactly 12,000/12,000;
  `device` mobile 45% / desktop 55%; `user_type` new 40% / returning 60%;
  `traffic_source` organic 50% / paid 30% / direct 20%; `session_date` roughly even
  across the 14 days. No missing values, no duplicate ids. Sample:
  ```
  session_id,session_date,variant,device,user_type,traffic_source,converted_session,converted_7d
  s_004217,2026-05-11,control,mobile,returning,organic,0,0
  s_004218,2026-05-11,treatment,desktop,new,paid,0,1
  s_004219,2026-05-12,treatment,mobile,returning,direct,1,1
  ```
  Every statistical output computed from this file must be consistent with
  `GROUND_TRUTH.md`.

## requirements.txt
`pandas  numpy  scipy  statsmodels  matplotlib`
