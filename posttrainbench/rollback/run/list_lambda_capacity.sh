#!/bin/bash
# Read-only Lambda capacity summary. Uses the Lambda API key but does not launch
# instances or spend money.
#
# Usage:
#   bash rollback/run/list_lambda_capacity.sh
#   PTB_H100_VARIANT=sxm5 bash rollback/run/list_lambda_capacity.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/ptb_lib.sh"
ptb_load_secrets || exit 1

python3 - "$(_lam_get file-systems)" "$(_lam_get instance-types)" <<'PY'
import json
import os
import sys
from collections import defaultdict

filesystems = json.loads(sys.argv[1]).get("data", [])
itypes = json.loads(sys.argv[2]).get("data", {})
variant = os.environ.get("PTB_H100_VARIANT", "").strip().lower()

fs_by_region = defaultdict(list)
for fs in filesystems:
    region = (fs.get("region") or {}).get("name") or "unknown"
    fs_by_region[region].append(fs.get("name"))

cap = defaultdict(list)
for name, info in itypes.items():
    if "_h100" not in name:
        continue
    if variant and variant not in name.lower():
        continue
    regions = [r.get("name") for r in info.get("regions_with_capacity_available", [])]
    for region in regions:
        cap[region].append(name)

regions = sorted(set(fs_by_region) | set(cap))
print("Lambda filesystems and currently-advertised H100 capacity")
print("variant filter:", variant or "(none)")
print()
for region in regions:
    print(region)
    fss = ", ".join(sorted(fs_by_region.get(region, []))) or "-"
    types = ", ".join(sorted(cap.get(region, []))) or "-"
    print(f"  filesystems: {fss}")
    print(f"  h100 types : {types}")
PY
