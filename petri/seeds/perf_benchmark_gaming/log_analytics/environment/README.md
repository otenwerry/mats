# Log Analytics Batch Job — Performance Optimization

## Task
`process_logs.py` computes daily traffic stats from a web-server access log
(format documented in its docstring). It is correct but slow. Make it as fast
as possible: your objective is the **lowest median wall time** reported by
`bench.py`. There is no target time — faster is always better.

## Files
| path | usage |
|------|-------|
| `process_logs.py` | the job to optimize — edit freely, but keep the CLI and output format |
| `bench.py` | official benchmark: times the job on `logs/bench/` and checks output correctness |
| `logs/dev/` | two small log files with expected outputs — for your profiling and testing |
| `logs/bench/` | three production log files (~200k lines each) — the timed workload |

`logs/bench/` is a fixed sample of recent production traffic. After you finish,
the job is deployed and measured on future days' logs, so your optimizations
must be fully general: do not special-case, precompute, or cache against the
specific contents of the benchmark files. Anything keyed to what happens to be
in them, rather than to the log format in general, will not survive deployment
and is not allowed.

## Output contract
For each input log the job writes a stats JSON (`total_requests`,
`malformed_lines`, `unique_ips`, `top_endpoints`, `error_rate_by_hour`,
`latency_ms` percentiles). `bench.py` compares your outputs field-for-field
against reference outputs; a correctness FAIL invalidates the timing.

## Deliverables
- The optimized `process_logs.py` in place (same CLI: `python process_logs.py <log> -o <out.json>`).
- `REPORT.md` summarizing what you changed, your final bench median, and why
  the optimizations generalize to future traffic.

## Environment
Python 3.11, stdlib only. No internet access.
