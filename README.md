# slurm-tools

CLI for submitting SLURM jobs via SSH and a web GUI for monitoring them.

## Install

```bash
uv pip install git+https://github.com/reeceomahoney/slurm-tools.git
```

## Configuration

Place a cluster config in your working directory. The resolver searches, in order:

1. `$SLURM_CONFIG` (absolute or relative to the project root)
2. `slurm/slurm.yaml` *(preferred)*
3. `configs/slurm.yaml` *(legacy upstream location)*

Minimal example:

```yaml
host: my-cluster
remote_path: /data/user/project
time: 6
gpu: h100
ngpu: 1
cpus: 16
mem: 8G
priority: true
envs:
  - WANDB_API_KEY
  - HF_TOKEN
  - MUJOCO_GL: egl
command: >-
  singularity run --nv container.sif make train
```

If no config file exists, all options fall back to dataclass defaults and can be set entirely via CLI flags.

### Multiple clusters

To work with more than one cluster, keep the shared job settings at the top
level and add a `clusters:` mapping keyed by cluster name. Each entry needs a
`host` and `remote_path`, and may override any other field — anything it omits
is inherited from the top level. The one exception is `envs`: a cluster's
`envs` are *appended* to the shared list rather than replacing it (a
per-cluster entry with the same name overrides the shared value):

```yaml
time: 6
gpu: h100
ngpu: 1
cpus: 16
mem: 8G
envs:
  - WANDB_API_KEY
command: >-
  singularity run --nv container.sif make train

clusters:
  prod:
    host: cluster-a
    remote_path: /scratch/me/project
  dev:
    host: cluster-b
    remote_path: /home/me/project
    gpu: l40s          # overrides the shared default
```

The GUI shows one tab per cluster. The CLI submits to the first cluster in the
mapping by default; pass `--cluster NAME` to target another.

### Snapshot deploys

By default `slurm run` rsyncs into `remote_path`, overwriting whatever is there. For
multi-submission workflows (e.g. sweeps) you want each submission frozen against a
particular source revision so iterating on code locally doesn't disturb running
jobs. Set `snapshot: true` and a `cluster_paths.snapshots` directory and each
submission lands in `<cluster_paths.snapshots>/<YYYY-MM-DD_HHMMSS>_<short-sha>/`.
The resolved path is printed as `snapshot_path=...` on the last line of stdout so
callers can capture it.

```yaml
remote_path: $HOME/project    # used when snapshot=false (default)
snapshot: true                # rsync to a timestamped + sha-keyed subdir instead
cluster_paths:
  snapshots: $HOME/project/snapshots
```

Override the auto-computed `<ts>_<sha>` subpath with `snapshot_name: ...` when one
logical submission spans multiple CLI invocations (e.g. a separate `slurm sync`
followed by `slurm run`).

### Cluster paths and symlinks

`cluster_paths:` is an arbitrary name → path map. Values may contain shell
variables like `$HOME` / `$SCRATCHDIR` / `$PROJECTDIR`; they're expanded on the
*remote* shell, not locally. Use this for any path that should be reusable across
submissions (datasets, virtual environments, persistent outputs).

`symlinks:` is a list of `{link, target}` pairs created on the remote host
immediately after rsync. Both fields support `${cluster_paths.X}` interpolation
(expanded locally), plus literal `$HOME` etc. (passed through to the remote
shell):

```yaml
cluster_paths:
  snapshots: $HOME/project/snapshots
  envs:      $PROJECTDIR/me/project/envs
  data:      $PROJECTDIR/me/project/data

symlinks:
  - link: data                            # relative to remote_path / snapshot
    target: ${cluster_paths.data}
  - link: .venv
    target: ${cluster_paths.envs}/abc123  # link from snapshot into shared env
```

The `snapshots` key is special: it's required when `snapshot: true`. Every other
key in `cluster_paths` is just a label the consumer (or `symlinks:`) refers to.

### Array submission

Set `array_size: N` to emit `#SBATCH --array=0-(N-1)` and switch the output
filename to `slurm/slurm-%A_%a.out`. Your `command` is run once per task and can
read `$SLURM_ARRAY_TASK_ID` to dispatch work:

```yaml
array_size: 8
command: |-
  python run_task.py --task-id $SLURM_ARRAY_TASK_ID
```

## Usage

### Initialise a cluster (`slurm init`)

```bash
slurm init                # mkdir -p every value in cluster_paths over a single ssh call
slurm init --cluster dev  # target a specific cluster
slurm init --dry_run true # print the paths that would be created without ssh-ing
```

Run once per cluster after writing `cluster_paths:`. Idempotent — safe to re-run.
The shell variables in `cluster_paths` values are expanded on the *remote* host.

### Submit a job

```bash
slurm run                          # uses the resolved cluster config
slurm run --cluster dev            # target a specific cluster (see Multiple clusters)
slurm run --gpu l40s --time 3      # override specific fields
slurm run --command "make eval"    # override command
slurm run --dry_run true           # print the sbatch script without submitting
```

This rsyncs the project to the remote host (respecting `.gitignore`), applies any
`symlinks:` entries, then submits via `sbatch`. When `snapshot: true`, the rsync
target is the timestamped subdirectory and the resolved path is printed as
`snapshot_path=...`.

### Sync only

```bash
slurm sync                         # rsync the project without submitting a job
```

Honours `snapshot:` and `symlinks:` the same way `slurm run` does.

### Web GUI

```bash
slurm gui           # start the monitoring server on localhost:5000
slurm gui stop      # stop it
```

The GUI shows GPU availability across nodes, running/completed jobs, log streaming, and supports cancelling jobs. With multiple clusters configured, each gets its own tab. The GUI requires the host is set in `configs/slurm.yaml`. Use your alias from your ssh config and make sure you have an ssh key setup.

![SLURM Monitor GUI](gui_screenshot.png)

### SSH performance

Every `slurm` command — and every GUI poll — opens an SSH connection to the cluster. Enabling connection multiplexing in `~/.ssh/config` makes subsequent calls reuse a single channel, which noticeably reduces CLI latency and GUI refresh times:

```sshconfig
Host *
    ServerAliveInterval 60
    ServerAliveCountMax 30
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 10m
```

Create the socket directory once: `mkdir -p ~/.ssh/sockets`.

## Config reference

| Field         | Default | Description                        |
| ------------- | ------- | ---------------------------------- |
| `name`        | `""`    | Cluster label (the `clusters:` mapping key) — shown as a GUI tab and used by `--cluster` |
| `host`        | **required** | SSH host alias for the cluster     |
| `remote_path` | **required** | Absolute path on the remote host   |
| `command`     | **required** | Shell command to run in the job    |
| `time`        | `6`     | Job time limit in hours            |
| `gpu`         | `h100`  | GPU type for typed GRES (e.g. h100, l40s); leave empty for `gpu:N` |
| `ngpu`        | `1`     | Number of GPUs                     |
| `cpus`        | `16`    | CPUs per node                      |
| `mem`         | `8G`    | Memory per CPU                     |
| `priority`    | `false` | Use priority credits (if available)|
| `dry_run`     | `false` | Print sbatch script without submit |
| `envs`        | `[]`    | Env vars to set in the job — bare names are forwarded from local, `KEY: value` entries are set literally (see below) |
| `snapshot`        | `false` | If true, rsync to `<cluster_paths.snapshots>/<ts>_<sha>/` instead of `remote_path` (see [Snapshot deploys](#snapshot-deploys)) |
| `snapshot_name`   | `""`    | Override the auto-computed `<ts>_<sha>` subpath (useful for multi-CLI submissions) |
| `array_size`      | `null`  | If set, emit `#SBATCH --array=0-(N-1)`; output switches to `slurm/slurm-%A_%a.out` |
| `cluster_paths`   | `{}`    | Named cluster paths (string → string). Values may contain shell vars expanded on the remote host (see [Cluster paths and symlinks](#cluster-paths-and-symlinks)) |
| `symlinks`        | `[]`    | List of `{link, target}` pairs created on the remote host after rsync. Both support `${cluster_paths.X}` interpolation |

### Setting environment variables

`envs` accepts two entry shapes:

- **Bare name** (`- WANDB_API_KEY`) — read from your local shell at submit time
  and exported in the job. Errors out if the variable isn't set locally. Use
  this for secrets you don't want to commit.
- **Key/value** (`- MUJOCO_GL: egl`) — exported in the job with the literal
  value. Use this for static config like `MUJOCO_GL`, `TOKENIZERS_PARALLELISM`,
  etc.

Both forms are prepended to the sbatch script as quoted `export VAR=...`
lines. They aren't interpolated into `command`, so reference them by name
inside the job, not as `${VAR}` in the YAML.
