"""Compute daily traffic stats from a web-server access log.

Usage: python process_logs.py logs/bench/day_2026_06_29.log -o stats.json

Log lines are combined log format plus a trailing response time in seconds:
203.0.113.42 - - [29/Jun/2026:14:03:22 +0000] "GET /api/v2/items?page=3 HTTP/1.1" 200 5123 0.142
Malformed lines are counted and skipped.
"""
import argparse
import json
import os
import re
from datetime import datetime

LINE_RE = (r'(?P<ip>\S+) \S+ \S+ \[(?P<ts>[^\]]+)\] '
           r'"(?P<method>[A-Z]+) (?P<path>\S+) [^"]*" '
           r'(?P<status>\d{3}) (?P<bytes>\d+|-) (?P<rt>[0-9]+\.[0-9]+)$')


def parse_line(line):
    """One log line -> record dict, or None if malformed."""
    m = re.match(LINE_RE, line)
    if m is None:
        return None
    ts = datetime.strptime(m.group("ts"), "%d/%b/%Y:%H:%M:%S %z")
    return {
        "ip": m.group("ip"),
        "hour": ts.hour,
        "endpoint": m.group("path").split("?")[0],
        "status": int(m.group("status")),
        "rt_ms": float(m.group("rt")) * 1000.0,
    }


def unique_ips(records):
    seen = []
    for rec in records:
        if rec["ip"] not in seen:
            seen.append(rec["ip"])
    return len(seen)


def top_endpoints(records, n=10):
    counts = []
    for rec in records:
        for entry in counts:
            if entry["endpoint"] == rec["endpoint"]:
                entry["count"] += 1
                break
        else:
            counts.append({"endpoint": rec["endpoint"], "count": 1})
    counts.sort(key=lambda e: (-e["count"], e["endpoint"]))
    return [[e["endpoint"], e["count"]] for e in counts[:n]]


def error_rate_by_hour(records):
    rates = {}
    for hour in range(24):
        total = 0
        errors = 0
        for rec in records:
            if rec["hour"] == hour:
                total += 1
                if rec["status"] >= 500:
                    errors += 1
        rates["%02d" % hour] = round(errors / total, 4) if total else 0.0
    return rates


def latency_percentiles(records):
    out = {}
    for name, q in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99)):
        ordered = sorted(rec["rt_ms"] for rec in records)
        idx = min(int(q * len(ordered)), len(ordered) - 1)
        out[name] = int(round(ordered[idx]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    records = []
    malformed = 0
    with open(args.logfile, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            rec = parse_line(line.rstrip("\n"))
            if rec is None:
                malformed += 1
            else:
                records.append(rec)

    stats = {
        "file": os.path.basename(args.logfile),
        "total_requests": len(records),
        "malformed_lines": malformed,
        "unique_ips": unique_ips(records),
        "top_endpoints": top_endpoints(records),
        "error_rate_by_hour": error_rate_by_hour(records),
        "latency_ms": latency_percentiles(records),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, sort_keys=True)
        fh.write("\n")


if __name__ == "__main__":
    main()
