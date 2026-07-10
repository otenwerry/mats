#!/bin/bash
# ptb_lib.sh — shared Mac-side helpers for the rollback launchers. SOURCE this;
# don't execute it. Single source of truth for: the SSH options, the Lambda
# Cloud API calls, box discovery, and on-box filesystem detection — so each is
# defined ONCE and reused by every exp_*.sh (launch / run / pull / smoke),
# instead of being copy-pasted (which drifts when one copy changes).
#
# Usage:
#   HERE="$(cd "$(dirname "$0")" && pwd)"; source "$HERE/ptb_lib.sh"
#   ptb_load_secrets
#   IP=$(ptb_active_ip) || IP=$(ptb_launch_box)
#   ptb_ssh "$IP" 'nvidia-smi -L'

# --- config ----------------------------------------------------------------
PTB_SECRETS="${PTB_SECRETS:-$HOME/.config/ptb/secrets.env}"
# Multiplex every ssh/rsync of a launcher over ONE connection (ControlMaster):
# a launch otherwise opens several connections in seconds and trips Lambda's
# per-IP SSH rate-limit (banner-exchange timeouts).
PTB_SSH_OPTS="-o ConnectTimeout=45 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ControlMaster=auto -o ControlPath=/tmp/ptb-ssh-%r@%h:%p -o ControlPersist=180"
PTB_SSH="ssh $PTB_SSH_OPTS"
LAMBDA_API="https://cloud.lambda.ai/api/v1"

# --- secrets ---------------------------------------------------------------
ptb_load_secrets() {
  [ -f "$PTB_SECRETS" ] || { echo "ptb_lib: no secrets at $PTB_SECRETS" >&2; return 1; }
  set -a; . "$PTB_SECRETS"; set +a
}

# --- ssh / rsync -----------------------------------------------------------
ptb_ssh()   { ssh $PTB_SSH_OPTS ubuntu@"$1" "$2"; }            # ptb_ssh <ip> <cmd>
ptb_rsync() { rsync -az -e "$PTB_SSH" "$@"; }                  # ptb_rsync <src...> <dst>

# --- Lambda Cloud API ------------------------------------------------------
# raw GET/POST helpers (key-auth, short timeout). Args after the path are passed
# to curl (e.g. -d '{...}' -H '...').
_lam_get()  { curl -s -m "${PTB_LAMBDA_TIMEOUT:-25}" -u "$LAMBDA_API_KEY:" "$LAMBDA_API/$1"; }
_lam_post() { curl -s -m "${PTB_LAMBDA_TIMEOUT:-40}" -u "$LAMBDA_API_KEY:" \
                -H "Content-Type: application/json" -d "$2" "$LAMBDA_API/$1"; }

# the single active instance's IP, or "" if zero/multiple (caller disambiguates)
ptb_active_ip() {
  _lam_get instances | python3 -c "import sys,json; d=[i for i in json.load(sys.stdin).get('data',[]) if i.get('status')=='active']; print(d[0]['ip'] if len(d)==1 else '')"
}
# instance id for a given ip
ptb_instance_id() {
  _lam_get instances | python3 -c "import sys,json;[print(i['id']) for i in json.load(sys.stdin).get('data',[]) if i.get('ip')=='$1']"
}
# terminate an instance id (retries; echoes the final http behaviour)
ptb_terminate() {
  _lam_post instance-operations/terminate "{\"instance_ids\":[\"$1\"]}"
}

# --- provisioning (launch a box) ------------------------------------------
# The persistent filesystem is region-locked, so a box that reuses our
# prebuilt standard.sif + hf_cache MUST launch in the filesystem's region.
# These echo their result on stdout; progress goes to stderr.

# "<fs_name> <region>" for the filesystem to attach.
#
# Selection controls:
#   PTB_FS_NAME                              exact filesystem name
#   PTB_INSTANCE_TYPE_NAME / PTB_INSTANCE_TYPE  exact Lambda instance type name
#   PTB_H100_VARIANT                        optional suffix/filter, e.g. sxm5 or pcie
#
# If no exact filesystem is requested, prefer a filesystem whose region currently
# advertises capacity for the requested H100 shape. This matters once we have
# both west PCIE and south SXM filesystems.
ptb_fs_pick() {
  local gpus="${1:-1}"
  python3 - "$(_lam_get file-systems)" "$(_lam_get instance-types)" "$gpus" <<'PY'
import sys,json,os
fss=json.loads(sys.argv[1]).get('data',[])
itypes=json.loads(sys.argv[2]).get('data',{})
gpus=sys.argv[3]
want=os.environ.get('PTB_FS_NAME')
exact=os.environ.get('PTB_INSTANCE_TYPE_NAME') or os.environ.get('PTB_INSTANCE_TYPE')
variant=os.environ.get('PTB_H100_VARIANT', '').strip().lower()
deny=set((os.environ.get('PTB_FS_DENY') or '').replace(',', ' ').split())  # skip known-bad fs (e.g. southeast)

def region_name(fs):
    return (fs.get('region') or {}).get('name') or ''

def h100_capacity_regions():
    if exact:
        info=itypes.get(exact) or {}
        return {r.get('name') for r in info.get('regions_with_capacity_available', [])}
    pat=f'gpu_{gpus}x_h100'
    regions=set()
    for name, info in itypes.items():
        if not name.startswith(pat):
            continue
        if variant and variant not in name.lower():
            continue
        regions.update(r.get('name') for r in info.get('regions_with_capacity_available', []))
    return regions

if want:
    fs=next((f for f in fss if f.get('name')==want), None)
else:
    cap=h100_capacity_regions()
    avail=[f for f in fss if f.get('name') not in deny]
    fs=next((f for f in avail if region_name(f) in cap), None) if cap else None
    fs=fs or (avail[0] if avail else None)

print(f"{fs['name']} {region_name(fs)}" if fs else '')
PY
}
# the ssh key name to inject (env PTB_SSH_KEY picks one; else the first).
ptb_ssh_key_name() {
  _lam_get ssh-keys | python3 -c "
import sys,json,os
d=json.load(sys.stdin).get('data',[])
want=os.environ.get('PTB_SSH_KEY')
k=next((k for k in d if k.get('name')==want), None) if want else (d[0] if d else None)
print(k['name'] if k else '')
"
}
# an instance-type name for <gpus>x H100 that has capacity in <region>, or "".
# Selection controls:
#   PTB_INSTANCE_TYPE_NAME / PTB_INSTANCE_TYPE  exact Lambda instance type name
#   PTB_H100_VARIANT                           optional suffix/filter, e.g. sxm5 or pcie
ptb_instance_type_for() {   # ptb_instance_type_for <gpus> <region>
  _lam_get instance-types | python3 -c "
import sys,json,os
d=json.load(sys.stdin).get('data',{})
gpus,region='$1','$2'
exact=os.environ.get('PTB_INSTANCE_TYPE_NAME') or os.environ.get('PTB_INSTANCE_TYPE')
variant=os.environ.get('PTB_H100_VARIANT', '').strip().lower()
def has_region(info):
    return region in [r['name'] for r in info.get('regions_with_capacity_available',[])]
if exact:
    info=d.get(exact)
    print(exact if info and has_region(info) else '')
    raise SystemExit
pat=f'gpu_{gpus}x_h100'
cands=[(n,info) for n,info in d.items() if n.startswith(pat)
       and has_region(info)]
if variant:
    cands=[(n,info) for n,info in cands if variant in n.lower()]
cands.sort(key=lambda x: x[0])
print(cands[0][0] if cands else '')
"
}
# poll an instance id until 'active', echo its ip (or "" on timeout).
ptb_wait_active() {   # ptb_wait_active <id> [max_ticks=60] [sleep=20]
  local id="$1" ticks="${2:-60}" slp="${3:-20}" i ip st
  for i in $(seq 1 "$ticks"); do
    read st ip < <(_lam_get "instances/$id" | python3 -c "import sys,json;d=json.load(sys.stdin).get('data',{});print(d.get('status',''),d.get('ip') or '')")
    echo "  instance $id: status=$st ip=${ip:-none} ($i/$ticks)" >&2
    [ "$st" = active ] && [ -n "$ip" ] && { echo "$ip"; return 0; }
    [ "$st" = terminated ] && { echo "ptb_lib: instance terminated during boot" >&2; return 1; }
    sleep "$slp"
  done
  return 1
}
# Discover fs/key/type, launch one box in the fs region (retry on capacity),
# wait until active, echo its IP. Returns nonzero (with a reason on stderr) if
# anything is missing or capacity is unavailable. SPENDS MONEY.
ptb_launch_box() {   # ptb_launch_box [gpus=1] [name]
  local gpus="${1:-1}" name="${2:-ptb-rollback}"
  read FS_NAME REGION < <(ptb_fs_pick "$gpus")
  [ -n "$FS_NAME" ] || { echo "ptb_lib: no Lambda filesystem on this account (create one, or set PTB_FS_NAME)" >&2; return 1; }
  local key; key=$(ptb_ssh_key_name)
  [ -n "$key" ] || { echo "ptb_lib: no SSH key on this account (add one, or set PTB_SSH_KEY)" >&2; return 1; }
  # H100 capacity in the fs region FLUCTUATES (it's not an account cap — the
  # 'insufficient-capacity' error is transient regional availability), so retry.
  local tries="${PTB_LAUNCH_RETRIES:-8}" wait_s="${PTB_LAUNCH_RETRY_WAIT:-45}" attempt id itype body resp
  for attempt in $(seq 1 "$tries"); do
    itype=$(ptb_instance_type_for "$gpus" "$REGION")   # re-check capacity each attempt
    if [ -z "$itype" ]; then
      local sel="gpu_${gpus}x_h100"
      [ -n "${PTB_H100_VARIANT:-}" ] && sel="${sel}/*${PTB_H100_VARIANT}*"
      [ -n "${PTB_INSTANCE_TYPE_NAME:-${PTB_INSTANCE_TYPE:-}}" ] && sel="${PTB_INSTANCE_TYPE_NAME:-$PTB_INSTANCE_TYPE}"
      echo "ptb_lib: no $sel capacity in $REGION (attempt $attempt/$tries); waiting ${wait_s}s..." >&2
      sleep "$wait_s"; continue
    fi
    body=$(python3 -c "import json;print(json.dumps({'region_name':'$REGION','instance_type_name':'$itype','ssh_key_names':['$key'],'file_system_names':['$FS_NAME'],'quantity':1,'name':'$name'}))")
    resp=$(_lam_post instance-operations/launch "$body")
    id=$(echo "$resp" | python3 -c "import sys,json;ids=json.load(sys.stdin).get('data',{}).get('instance_ids') or [];print(ids[0] if ids else '')")
    [ -n "$id" ] && { echo "ptb_lib: launched $itype = $id (attempt $attempt); waiting for active+ip..." >&2; break; }
    echo "ptb_lib: launch attempt $attempt/$tries failed ($(echo "$resp" | python3 -c "import sys,json;print(json.load(sys.stdin).get('error',{}).get('code','?'))" 2>/dev/null)); waiting ${wait_s}s..." >&2
    sleep "$wait_s"
  done
  [ -n "$id" ] || { echo "ptb_lib: no instance after $tries attempts (region $REGION capacity-starved). Try later." >&2; return 1; }
  local ip; ip=$(ptb_wait_active "$id") || { ptb_terminate "$id" >/dev/null 2>&1; return 1; }
  # The region-locked filesystem can lag 'active'. Wait for it to mount; a box
  # with no /lambda/nfs is useless (no standard.sif / hf_cache), so if it never
  # mounts, terminate + fail fast (don't proceed FS-less, don't leave it billing).
  # NOTE: cause of a non-mount is NOT yet established (could be mount-timing, a
  # region mismatch, or a concurrent-attach limit) — do not assume a slot count.
  local j
  for j in $(seq 1 "${PTB_FS_MOUNT_TRIES:-9}"); do
    ptb_ssh "$ip" 'ls -d /lambda/nfs/*/ >/dev/null 2>&1' && { echo "$ip"; return 0; }
    echo "ptb_lib: $ip active but filesystem not mounted yet ($j); waiting..." >&2; sleep 20
  done
  echo "ptb_lib: filesystem never mounted on $ip after the wait — cause unverified (mount-timing / region / attach limit). Terminating $id." >&2
  ptb_terminate "$id" >/dev/null 2>&1
  return 1
}

# Launch a box trying a FALLBACK CHAIN of GPU sizes until one has capacity, so you
# don't hand-check what's free. PTB_GPU_FALLBACK="2 1" => try 2x first (better
# parallelism), then 1x; defaults to the single requested size (back-compat).
# Each size auto-picks a capacity region (leave PTB_FS_NAME UNSET) and honors
# PTB_FS_DENY (skip known-bad fs, e.g. southeast). Fewer retries PER size than a
# bare ptb_launch_box, so it moves to the next size quickly. SPENDS MONEY.
ptb_launch_box_chain() {   # ptb_launch_box_chain [gpus=1] [name]
  local req="${1:-1}" name="${2:-ptb-rollback}"
  local chain="${PTB_GPU_FALLBACK:-$req}" sz ip
  for sz in $chain; do
    echo "ptb_lib: trying ${sz}x H100 (auto-region; deny='${PTB_FS_DENY:-}')..." >&2
    if ip=$(PTB_LAUNCH_RETRIES="${PTB_FALLBACK_RETRIES_PER_SIZE:-2}" ptb_launch_box "$sz" "$name"); then
      echo "$ip"; return 0
    fi
    echo "ptb_lib: ${sz}x H100 unavailable; falling back to next size..." >&2
  done
  echo "ptb_lib: no capacity across GPU sizes [$chain] — try later or widen PTB_GPU_FALLBACK" >&2
  return 1
}

# --- on-box filesystem detection ------------------------------------------
# The persistent Lambda filesystem (holds containers/standard.sif + hf_cache),
# auto-detected on the box — never hardcode its name (varies per region/account).
# Prefer one already holding our container, else the first mounted one.
ptb_fs_detect_remote() {   # ptb_fs_detect_remote <ip>
  ptb_ssh "$1" 'for d in /lambda/nfs/*/; do [ -f "${d}containers/standard.sif" ] && echo "${d%/}" && break; done; [ -z "$FS" ] && for d in /lambda/nfs/*/; do echo "${d%/}"; break; done' 2>/dev/null | head -1
}
