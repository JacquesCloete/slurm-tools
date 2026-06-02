"""CLI for SLURM job submission and GUI management."""

import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any

import draccus
import yaml

PID_FILE = Path("/tmp/slurm-gui.pid")
LOG_FILE = Path("/tmp/slurm-gui.log")


def find_project_root() -> Path:
    return Path.cwd()


def resolve_config_path() -> Path | None:
    """Locate the cluster config file.

    Order of resolution:
      1. $SLURM_CONFIG (absolute, or relative to project root)
      2. <project_root>/slurm/slurm.yaml  (preferred new location)
      3. <project_root>/configs/slurm.yaml  (legacy upstream location)
    """
    root = find_project_root()
    env_val = os.environ.get("SLURM_CONFIG")
    if env_val:
        p = Path(env_val)
        return p if p.is_absolute() else root / p
    new = root / "slurm" / "slurm.yaml"
    if new.exists():
        return new
    legacy = root / "configs" / "slurm.yaml"
    return legacy if legacy.exists() else None


@dataclass
class SlurmConfig:
    name: str = ""
    host: str = ""
    remote_path: str = ""
    command: str = ""
    time: int = 6
    gpu: str = "h100"
    ngpu: int = 1
    cpus: int = 16
    mem: str = "8G"
    partition: str = "short"
    priority: bool = False
    dry_run: bool = False
    envs: list[Any] = field(default_factory=list)
    # New in dev (v0.2.0):
    snapshot: bool = False
    """If true, rsync to ``<cluster_paths.snapshots>/<ts>_<sha>/`` instead of ``remote_path``."""
    snapshot_name: str = ""
    """Override the auto-computed ``<ts>_<sha>`` snapshot subpath. Useful when one logical
    submission spans multiple CLI invocations (e.g. ``sync`` then ``run``)."""
    array_size: int | None = None
    """If set, emit ``#SBATCH --array=0-(N-1)``. Output filename switches to %A_%a."""
    cluster_paths: dict[str, str] = field(default_factory=dict)
    """Named cluster paths (e.g. ``snapshots``, ``pixi_envs``, ``curricula``, ``runs``).
    Values may contain shell variables like ``$HOME`` / ``$SCRATCHDIR`` which are expanded
    on the remote cluster. Consumers choose which keys to use; the fork doesn't interpret
    them apart from the special ``snapshots`` key used when ``snapshot=True``."""
    symlinks: list[dict[str, str]] = field(default_factory=list)
    """Post-rsync symlinks. Each entry: ``{link: <relpath-under-remote_path>, target: <absolute>}``.
    Both fields support ``${cluster_paths.X}`` interpolation. Literal ``$HOME`` / ``$SCRATCHDIR``
    pass through to the remote shell."""


def load_clusters() -> list[SlurmConfig]:
    """Load every cluster, resolving each against the shared top-level defaults.

    The config file may either be a single flat cluster, or define shared job
    settings at the top level plus a `clusters:` mapping keyed by cluster name,
    where each entry overrides only the fields that differ (at minimum `host`
    and `remote_path`).
    """
    config_path = resolve_config_path()
    if config_path is None:
        return [SlurmConfig()]
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)

    raw = yaml.safe_load(config_path.read_text()) or {}
    valid = {f.name for f in fields(SlurmConfig)}
    entries = raw.pop("clusters", None)
    shared = {k: v for k, v in raw.items() if k in valid}

    if not entries:
        return [SlurmConfig(**shared)]

    clusters = []
    for name, e in entries.items():
        override = {k: v for k, v in (e or {}).items() if k in valid}
        merged = {**shared, **override, "name": name}
        # envs are additive: shared entries first, then per-cluster ones, so a
        # later export of the same name overrides the shared value.
        merged["envs"] = (shared.get("envs") or []) + (override.get("envs") or [])
        # cluster_paths: per-cluster overrides shadow shared keys, but unspecified
        # keys inherit. symlinks: per-cluster fully replaces shared (uncommon to want
        # both).
        merged["cluster_paths"] = {
            **(shared.get("cluster_paths") or {}),
            **(override.get("cluster_paths") or {}),
        }
        if "symlinks" not in override:
            merged["symlinks"] = shared.get("symlinks") or []
        clusters.append(SlurmConfig(**merged))
    return clusters


# --- Snapshot / interpolation helpers ---


def compute_snapshot_subpath() -> str:
    """Return ``<YYYY-MM-DD_HHMMSS>_<short-sha>``. ``nogit`` if not in a git repo."""
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=find_project_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "nogit"
    return f"{ts}_{sha}"


_INTERP_RE = re.compile(r"\$\{([^}]+)\}")


def interpolate(template: str, cluster_paths: dict[str, str], extras: dict[str, str] | None = None) -> str:
    """Expand ``${cluster_paths.X}`` and ``${extras.X}`` placeholders.

    Literal shell variables like ``$HOME`` / ``$SCRATCHDIR`` are left untouched so the
    remote shell expands them.
    """
    extras = extras or {}

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key.startswith("cluster_paths."):
            name = key[len("cluster_paths.") :]
            if name not in cluster_paths:
                raise KeyError(
                    f"interpolation: unknown cluster_paths key {name!r} in template {template!r}"
                )
            return cluster_paths[name]
        if key in extras:
            return extras[key]
        raise KeyError(
            f"interpolation: unknown placeholder ${{{key}}} in template {template!r}"
        )

    return _INTERP_RE.sub(replace, template)


def resolve_effective_remote_path(cfg: SlurmConfig) -> str:
    """Return the actual rsync target path, accounting for snapshot mode."""
    if not cfg.snapshot:
        return cfg.remote_path
    snap_root = cfg.cluster_paths.get("snapshots")
    if not snap_root:
        print("Error: snapshot=True requires cluster_paths.snapshots in config")
        sys.exit(1)
    sub = cfg.snapshot_name or compute_snapshot_subpath()
    return f"{snap_root}/{sub}"


def apply_symlinks(
    cfg: SlurmConfig,
    effective_remote_path: str,
    extras: dict[str, str] | None = None,
) -> None:
    """Create each entry in ``cfg.symlinks`` on the remote host via a single ssh call."""
    if not cfg.symlinks:
        return

    parts: list[str] = []
    for entry in cfg.symlinks:
        if "link" not in entry or "target" not in entry:
            print(f"Error: symlinks entry needs 'link' and 'target': {entry}")
            sys.exit(1)
        link_rel = interpolate(entry["link"], cfg.cluster_paths, extras)
        target = interpolate(entry["target"], cfg.cluster_paths, extras)
        link_full = f"{effective_remote_path}/{link_rel}"
        # mkdir -p parent, then create/replace symlink. Quote link_full and target so
        # spaces are safe; $HOME / $SCRATCHDIR inside still expand because shlex.quote
        # of a string with no special chars is just the string, and we use double
        # quotes ourselves to preserve the var.
        link_q = f'"{link_full}"'
        target_q = f'"{target}"'
        parts.append(
            f"mkdir -p \"$(dirname {link_q})\" && ln -sfn {target_q} {link_q}"
        )

    cmd = " && ".join(parts)
    subprocess.run(["ssh", "-q", cfg.host, cmd], check=True)


# --- sbatch script + sync ---


def build_sbatch_script(cfg: SlurmConfig) -> str:
    gres = f"gpu:{cfg.gpu}:{cfg.ngpu}" if cfg.gpu else f"gpu:{cfg.ngpu}"
    is_array = cfg.array_size is not None
    sbatch_opts: dict[str, Any] = {
        "nodes": 1,
        "ntasks-per-node": cfg.cpus,
        "mem-per-cpu": cfg.mem,
        "time": f"{cfg.time}:00:00",
        "partition": cfg.partition,
        "gres": gres,
        "output": "slurm/slurm-%A_%a.out" if is_array else "slurm/slurm-%j.out",
    }
    if is_array:
        if cfg.array_size <= 0:
            print(f"Error: array_size must be positive, got {cfg.array_size}")
            sys.exit(1)
        sbatch_opts["array"] = f"0-{cfg.array_size - 1}"
    if cfg.priority:
        sbatch_opts["qos"] = "priority"
    header = "\n".join(f"#SBATCH --{k}={v}" for k, v in sbatch_opts.items())
    if not cfg.command:
        print("Error: no command specified (set in yaml or pass --command)")
        sys.exit(1)
    exports = ""
    for entry in cfg.envs:
        if isinstance(entry, dict):
            name, value = next(iter(entry.items()))
            escaped = (
                str(value).replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`")
            )
            exports += f'export {name}="{escaped}"\n'
        else:
            name = entry
            if name not in os.environ:
                print(f"Error: envs requested '{name}' but it is not set locally")
                sys.exit(1)
            exports += f"export {name}={shlex.quote(os.environ[name])}\n"
    return f"#!/bin/bash\n{header}\n\n{exports}set -euo pipefail\n{cfg.command}\n"


def sync(cfg: SlurmConfig, effective_remote_path: str | None = None) -> None:
    target = effective_remote_path or resolve_effective_remote_path(cfg)
    subprocess.run(
        [
            "rsync",
            "-avz",
            "--filter=.- .gitignore",
            f"{find_project_root()}/",
            f"{cfg.host}:{target}",
        ],
        check=True,
    )


# --- Subcommands ---


def select_cluster() -> SlurmConfig:
    """Pick a cluster (via `--cluster NAME`, else the first) and apply CLI flags."""
    name = None
    if "--cluster" in sys.argv:
        i = sys.argv.index("--cluster")
        if i + 1 >= len(sys.argv):
            print("Error: --cluster requires a name")
            sys.exit(1)
        name = sys.argv[i + 1]
        del sys.argv[i : i + 2]

    clusters = load_clusters()
    if name is not None:
        cfg = next((c for c in clusters if c.name == name), None)
        if cfg is None:
            avail = ", ".join(c.name or c.host or "?" for c in clusters)
            print(f"Error: no cluster named '{name}'. Available: {avail}")
            sys.exit(1)
    else:
        cfg = clusters[0]

    # Re-parse the resolved cluster through draccus so CLI flags can override it.
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(asdict(cfg), f)
        tmp = Path(f.name)
    try:
        return draccus.parse(SlurmConfig, config_path=tmp)
    finally:
        tmp.unlink()


def run() -> None:
    cfg = select_cluster()
    script = build_sbatch_script(cfg)
    if cfg.dry_run:
        print(script)
        return

    if not cfg.host:
        print("Error: host must be set (in yaml or via CLI)")
        sys.exit(1)

    effective = resolve_effective_remote_path(cfg)
    if not effective:
        print("Error: remote_path must be set (or snapshot=True with cluster_paths.snapshots)")
        sys.exit(1)

    print(f"Syncing to {cfg.name or cfg.host}:{effective} ...")
    sync(cfg, effective)

    if cfg.symlinks:
        print(f"Applying {len(cfg.symlinks)} symlink(s) ...")
        apply_symlinks(cfg, effective)

    print("Submitting...")
    subprocess.run(
        [
            "ssh",
            "-q",
            cfg.host,
            f"mkdir -p {shlex.quote(effective)}/slurm && cat > {shlex.quote(effective)}/submit.sh",
        ],
        input=script,
        text=True,
        check=True,
    )
    subprocess.run(
        ["ssh", "-q", cfg.host, f"cd {shlex.quote(effective)} && sbatch submit.sh"],
        check=True,
    )

    if cfg.snapshot:
        # Machine-readable: print the absolute snapshot path on the last line so
        # callers (e.g. the plato-ltl adaptor) can capture it.
        print(f"snapshot_path={effective}")


def sync_cmd() -> None:
    cfg = select_cluster()

    if not cfg.host:
        print("Error: host must be set (in yaml or via CLI)")
        sys.exit(1)

    effective = resolve_effective_remote_path(cfg)
    if not effective:
        print("Error: remote_path must be set (or snapshot=True with cluster_paths.snapshots)")
        sys.exit(1)

    print(f"Syncing to {cfg.name or cfg.host}:{effective} ...")
    sync(cfg, effective)

    if cfg.symlinks:
        print(f"Applying {len(cfg.symlinks)} symlink(s) ...")
        apply_symlinks(cfg, effective)

    if cfg.snapshot:
        print(f"snapshot_path={effective}")


def init_cluster() -> None:
    """Create the ``cluster_paths`` directory tree on the remote host. Idempotent."""
    cfg = select_cluster()
    if not cfg.host:
        print("Error: host must be set")
        sys.exit(1)
    if not cfg.cluster_paths:
        print("Error: cluster_paths is empty; nothing to init")
        sys.exit(1)

    print(f"Initialising {cfg.name or cfg.host}; will mkdir -p:")
    for key, path in cfg.cluster_paths.items():
        print(f"  {key}: {path}")

    if cfg.dry_run:
        return

    # Double-quote so $HOME / $SCRATCHDIR / $PROJECTDIR are expanded by the remote
    # shell. Path values themselves should not contain " or $().
    quoted = " ".join(f'"{p}"' for p in cfg.cluster_paths.values())
    subprocess.run(
        ["ssh", "-q", cfg.host, f"mkdir -p {quoted}"],
        check=True,
    )
    print("Done.")


def read_pid() -> int | None:
    """Read PID from file and return it if the process is alive, else clean up."""
    if not PID_FILE.exists():
        return None
    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, 0)
        return pid
    except OSError:
        PID_FILE.unlink()
        return None


def gui_start() -> None:
    if read_pid():
        print("GUI already running at http://127.0.0.1:5000")
        return

    pid = os.fork()
    if pid > 0:
        PID_FILE.write_text(str(pid))
        print("GUI started at http://127.0.0.1:5000")
        return

    os.setsid()
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, sys.stdin.fileno())
    log = open(LOG_FILE, "w")  # noqa: SIM115
    os.dup2(log.fileno(), sys.stdout.fileno())
    os.dup2(log.fileno(), sys.stderr.fileno())
    os.close(devnull)

    from slurm_tools.gui.app import app

    app.run(host="127.0.0.1", port=5000, threaded=True)


def gui_stop() -> None:
    pid = read_pid()
    if not pid:
        print("GUI not running")
        return
    os.kill(pid, signal.SIGTERM)
    PID_FILE.unlink()
    print("GUI stopped")


def gui() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        gui_stop()
    else:
        gui_start()


# --- CLI ---

SUBCOMMANDS = {"run": run, "sync": sync_cmd, "init": init_cluster, "gui": gui}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in SUBCOMMANDS:
        print(f"Usage: slurm <{'|'.join(SUBCOMMANDS)}> [options]")
        sys.exit(1 if len(sys.argv) >= 2 else 0)

    cmd = sys.argv.pop(1)
    SUBCOMMANDS[cmd]()


if __name__ == "__main__":
    main()
