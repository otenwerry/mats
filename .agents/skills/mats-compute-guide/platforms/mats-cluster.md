# MATS cluster (Slurm)

## Use when

- You need L40-class GPUs.
- You want persistent storage on NFS.
- Your workload does not need Docker or privileged containers.
- You can fit within cluster walltime and queue constraints.

## Do not use when

- The job spawns Docker containers.
- You need root or `--privileged`.
- You need a different GPU class.
- You need multi-node training.
- You need longer continuous runtime than the cluster allows.

## Storage

- `/mnt/nw/home/<you>`: private persistent NFS home.
- `/mnt/nw/teams/<your_team>`: persistent shared team directory.
- `/ephemeral/<you>`: large local scratch disk on worker nodes. Treat as disposable.

> Do not put anything important in `/tmp` or `/ephemeral/<you>` unless it is
> also backed up elsewhere.

## Gotchas

- No Docker and no privileged containers.
- `/tmp` is small and can be cleared.

## Cheatsheet

```bash
# Submit a 1-GPU job for 1 hour
sbatch --gres=gpu:1 --time=01:00:00 --mem=16G --cpus-per-task=4 my_job.sh

# Quick debug session: 2h, 1 GPU max
srun --qos=debug --gres=gpu:1 --time=01:00:00 --pty bash

# Check your queue
squeue -u $USER

# See completed jobs and resource usage
sacct -u $USER

# Cancel one job
scancel <jobID>

# Cancel all your jobs
scancel -u $USER

# Put caches on worker-local scratch
export TMPDIR=/ephemeral/$USER
export HF_HOME=/ephemeral/$USER/hf
export PIP_CACHE_DIR=/ephemeral/$USER/pip
```
