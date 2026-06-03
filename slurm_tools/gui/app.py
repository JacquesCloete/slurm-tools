"""SLURM experiment monitor — lightweight web GUI."""

import fnmatch
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Flask, Response, render_template, request

from slurm_tools.slurm import SlurmConfig, interpolate, load_clusters

CLUSTERS: list[tuple[str, SlurmConfig]] = [
    (c.name or c.host or f"cluster-{i + 1}", c)
    for i, c in enumerate(load_clusters())
]
CLUSTER_MAP = dict(CLUSTERS)
CLUSTER_NAMES = [name for name, _ in CLUSTERS]

GPU_MEM_RE = re.compile(r"gpu_mem:(\d+)GB", re.IGNORECASE)

dir = Path(__file__).parent
app = Flask(
    __name__,
    template_folder=str(dir),
    static_folder=str(dir),
    static_url_path="/static",
)


def ssh(cluster: SlurmConfig, cmd: str, *, timeout: int = 10) -> str:
    result = subprocess.run(
        ["ssh", "-q", cluster.host, cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout


def ssh_many(cluster: SlurmConfig, *cmds: str, timeout: int = 10) -> list[str]:
    """Run several SSH commands against one cluster concurrently."""
    with ThreadPoolExecutor(max_workers=len(cmds)) as pool:
        return list(pool.map(lambda c: ssh(cluster, c, timeout=timeout), cmds))


def current_cluster() -> SlurmConfig:
    """Resolve the cluster for this request from the `?cluster=` query param."""
    name = request.args.get("cluster")
    if name and name in CLUSTER_MAP:
        return CLUSTER_MAP[name]
    return CLUSTERS[0][1]


def parse_table(raw: str) -> tuple[list[str], list[list[str]]]:
    lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
    if not lines:
        return [], []
    return lines[0].split(), [ln.split() for ln in lines[1:]]


# -- Routes ----------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html", clusters=CLUSTER_NAMES)


GRES_RE = re.compile(r"gpu:(?:\(null\)|[^():\s]+)?:?(\d+)", re.IGNORECASE)
GRES_TYPE_RE = re.compile(r"gpu:([^:(]+):(\d+)", re.IGNORECASE)
FEATURE_GPU_RE = re.compile(r"NVIDIA_(\w+)", re.IGNORECASE)
UNAVAILABLE = ("down", "drain", "maint", "fail", "invalid", "reserved", "unknown")


def is_unavailable(state: str) -> bool:
    s = state.lower()
    return any(tok in s for tok in UNAVAILABLE)


def gpu_count(gres: str) -> int:
    m = GRES_RE.search(gres)
    return int(m.group(1)) if m else 0


def gpu_type_and_count(gres: str, features: str = "") -> tuple[str, int] | None:
    m = GRES_TYPE_RE.search(gres)
    if m:
        return (m.group(1).upper(), int(m.group(2)))
    n = gpu_count(gres)
    if not n:
        return None
    feat = FEATURE_GPU_RE.search(features)
    return (feat.group(1).upper() if feat else "GPU", n)


@app.route("/nodes")
def nodes():
    cluster = current_cluster()
    raw_nodes = ssh(
        cluster,
        "sinfo -N -h -O 'NodeHost:25,StateLong:20,Gres:40,GresUsed:60,Features:250'",
    )

    # Aggregate by (gpu_type, vram_gb) — same type can have different VRAM sizes
    seen: set[str] = set()
    totals: dict[tuple, int] = {}
    free_map: dict[tuple, int] = {}
    for line in raw_nodes.strip().splitlines():
        parts = line.split(maxsplit=4)
        if len(parts) < 4 or parts[0] in seen:
            continue
        seen.add(parts[0])
        _, state, gres, gres_used = parts[:4]
        features = parts[4] if len(parts) > 4 else ""

        parsed = gpu_type_and_count(gres, features)
        if not parsed:
            continue
        gpu_type, total = parsed
        m = GPU_MEM_RE.search(features)
        vram_gb = int(m.group(1)) if m else 0
        key = (gpu_type, vram_gb)

        allocated = gpu_count(gres_used)
        node_free = 0 if is_unavailable(state) else max(0, total - allocated)

        totals[key] = totals.get(key, 0) + total
        free_map[key] = free_map.get(key, 0) + node_free

    gpus = []
    for key in sorted(totals):
        gpu_type, vram_per_gpu = key
        t, f = totals[key], free_map[key]
        vram_total = t * vram_per_gpu
        vram_used = (t - f) * vram_per_gpu
        vram_free = f * vram_per_gpu
        vram_pct = round(vram_used / vram_total * 100) if vram_total else 0
        gpus.append(
            {
                "type": gpu_type,
                "total": t,
                "used": t - f,
                "free": f,
                "pct": round((t - f) / t * 100) if t else 0,
                "vram_per_gpu": vram_per_gpu,
                "vram_total": vram_total,
                "vram_used": vram_used,
                "vram_free": vram_free,
                "vram_pct": vram_pct,
            }
        )
    return render_template("nodes.html", gpus=gpus)


@app.route("/jobs")
def jobs():
    cluster = current_cluster()
    cmds = [
        "squeue -u $USER -o '%.12i %.30j %.8T %.10M %.20b'",
        "sacct -u $USER -S now-7days --noheader --parsable2 -X "
        "-o 'JobID,JobName,State,Elapsed,AllocTRES' "
        "| grep -vE 'RUNNING|PENDING'",
    ]
    # When a log_glob is configured, also list every log file that still exists
    # on disk (concurrently — it doesn't depend on the sacct result) so we can
    # drop closed jobs whose logs have been deleted. This keeps the closed-jobs
    # panel a reflection of what's actually on the cluster rather than a flat
    # 7-day sacct window. Active (squeue) jobs are never filtered.
    listing_glob = log_listing_glob(cluster)
    if listing_glob is not None:
        cmds.append(f"ls -1 {listing_glob} 2>/dev/null")
    results = ssh_many(cluster, *cmds)
    raw, raw_closed = results[0], results[1]
    existing_log_basenames = (
        {Path(ln).name for ln in results[2].splitlines() if ln.strip()}
        if listing_glob is not None
        else None
    )
    headers, rows = parse_table(raw)
    # Rename ugly TRES_PER_NODE header
    headers = ["GPU" if "TRES" in h else h for h in headers]
    # Format GPU column: "gpu:l40s:1" -> "l40s"
    gpu_idx = headers.index("GPU") if "GPU" in headers else None
    if gpu_idx is not None:
        for row in rows:
            if gpu_idx < len(row):
                m = GRES_TYPE_RE.search(row[gpu_idx])
                row[gpu_idx] = (
                    f"{m.group(1).upper()}:{m.group(2)}" if m else row[gpu_idx]
                )
    # Recent completed/failed/cancelled jobs from the last 7 days
    closed_lines = [ln for ln in raw_closed.strip().splitlines() if ln.strip()]
    closed_rows = [ln.split("|") for ln in reversed(closed_lines)]
    for row in closed_rows:
        # Normalise e.g. "CANCELLED by 12345" or "CANCELLED+" to "CANCELLED"
        if len(row) > 2:
            row[2] = row[2].split()[0].rstrip("+")
        tres = row[4] if len(row) > 4 else ""
        gpu = ""
        for part in tres.split(","):
            if "gres/gpu:" in part:
                rest = part.split("gres/gpu:")[1]
                name, _, count = rest.partition("=")
                gpu = f"{name.upper()}:{count}" if count else name.upper()
                break
        row[4:] = [gpu]
    # Drop closed jobs whose log files no longer exist on disk (deleted runs).
    if existing_log_basenames is not None:
        closed_rows = [
            row
            for row in closed_rows
            if row and job_has_log(cluster, row[0], existing_log_basenames)
        ]
    closed_headers = ["JOBID", "NAME", "STATE", "ELAPSED", "GPU"] if closed_rows else []
    return render_template(
        "jobs.html",
        headers=headers,
        rows=rows,
        closed_headers=closed_headers,
        closed_rows=closed_rows,
    )


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel(job_id):
    # Accept both ``<A>`` (whole array) and ``<A>_<a>`` (single array task).
    if parse_job_id(job_id) is None:
        return "Bad job id", 400
    ssh(current_cluster(), f"scancel {job_id}")
    return "", 204



JOB_ID_RE = re.compile(r"^(\d+)(?:_(\d+))?$")


def parse_job_id(job_id: str) -> tuple[str, str] | None:
    """Split ``<A>`` or ``<A>_<a>`` into ``(base, arrayidx)`` or return ``None``.

    For non-array jobs ``arrayidx`` is the empty string. Returns ``None`` for
    anything that isn't ``\\d+`` or ``\\d+_\\d+`` so route handlers can reject
    obvious garbage before shelling out.
    """
    m = JOB_ID_RE.match(job_id)
    if m is None:
        return None
    return m.group(1), m.group(2) or ""


def resolve_log_paths(cluster: SlurmConfig, job_id: str) -> str:
    """Return a shell-ready path or glob locating the log file(s) for ``job_id``.

    ``job_id`` may be a bare job ID (``\\d+``) or an array-task ID
    (``\\d+_\\d+``). The ``{jobid}`` template variable always resolves to the
    base job ID. ``{arrayidx}`` resolves to the array task index when the user
    clicked a specific task, and to ``*`` (shell wildcard, matches any task)
    when the user clicked the bare base ID — so a template like
    ``slurm-{jobid}_{arrayidx}.out`` shows a single task's log on a per-task
    click and every task's log when the base ID is clicked.

    If ``cluster.log_glob`` is set, expand its ``{jobid}`` and ``{arrayidx}``
    placeholders and any ``${cluster_paths.X}`` references; the resulting
    pattern (which may contain shell wildcards) is interpreted as relative to
    ``remote_path`` unless it begins with ``/`` or ``$`` (an unexpanded shell
    var the remote shell will handle). Otherwise fall back to the legacy
    single-file path ``<remote_path>/slurm/slurm-<jobid>.out``.
    """
    parsed = parse_job_id(job_id)
    base = parsed[0] if parsed else job_id
    # Default arrayidx to "*" so templates like `slurm-{jobid}_{arrayidx}.out`
    # still match every task when the user clicks the base array job ID.
    arrayidx = parsed[1] if parsed and parsed[1] else "*"

    if not cluster.log_glob:
        return f"{cluster.remote_path}/slurm/slurm-{base}.out"
    pattern = cluster.log_glob.replace("{jobid}", base).replace("{arrayidx}", arrayidx)
    pattern = interpolate(pattern, cluster.cluster_paths)
    if pattern.startswith(("/", "$")):
        return pattern
    return f"{cluster.remote_path}/{pattern}"


def log_listing_glob(cluster: SlurmConfig) -> str | None:
    """Shell glob matching EVERY job's log file, or ``None`` if not filterable.

    Replaces both ``{jobid}`` and ``{arrayidx}`` with ``*`` so a single remote
    ``ls`` enumerates all logs that currently exist on disk. Returns ``None``
    when no ``log_glob`` is configured (nothing to filter against)."""
    if not cluster.log_glob:
        return None
    pattern = cluster.log_glob.replace("{jobid}", "*").replace("{arrayidx}", "*")
    pattern = interpolate(pattern, cluster.cluster_paths)
    if not pattern.startswith(("/", "$")):
        pattern = f"{cluster.remote_path}/{pattern}"
    return pattern


def job_has_log(cluster: SlurmConfig, job_id: str, existing_basenames: set[str]) -> bool:
    """True if some on-disk log file matches ``job_id``'s log pattern.

    Compares on basename only: the resolved pattern carries an unexpanded shell
    var (e.g. ``$SCRATCHDIR``) while the listed paths are already expanded, so
    the directory parts won't compare equal — the filename component does."""
    pat = Path(resolve_log_paths(cluster, job_id)).name
    return any(fnmatch.fnmatch(name, pat) for name in existing_basenames)


@app.route("/logs/<job_id>/history")
def logs_history(job_id):
    """Stream the existing log file as chunked plain text (fast initial load)."""
    if parse_job_id(job_id) is None:
        return "Bad job id", 400

    cluster = current_cluster()
    log_paths = resolve_log_paths(cluster, job_id)
    # 2>/dev/null suppresses "no such file" if the glob matches nothing.
    cmd = f"cat {log_paths} 2>/dev/null"

    def stream():
        proc = subprocess.Popen(
            ["ssh", "-q", cluster.host, cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        try:
            assert proc.stdout is not None
            while chunk := proc.stdout.read(65536):
                yield chunk
        finally:
            proc.terminate()
            proc.wait()

    return Response(stream(), mimetype="text/plain; charset=utf-8")


@app.route("/logs/<job_id>")
def logs(job_id):
    """SSE stream of newly-appended log lines only (use /history for backlog)."""
    if parse_job_id(job_id) is None:
        return "Bad job id", 400

    cluster = current_cluster()
    log_paths = resolve_log_paths(cluster, job_id)
    # tail -f over a glob follows whichever files exist at command start; new
    # array tasks that begin writing later won't be picked up. Acceptable for V1
    # of snapshot-aware logs — the user can refresh the page to re-glob.
    #
    # Wrap the remote `tail -f` in a stdin-EOF watchdog so it can't outlive this
    # SSE connection. Without a PTY, killing the local ssh client does NOT signal
    # the remote command, so a bare `tail -f` would be orphaned on the login node
    # (holding the log file open) every time a log pane closes. Instead: run tail
    # in the background and block on `cat` reading the ssh channel's stdin. When
    # this connection ends — we close proc.stdin below, or the ssh dies and the
    # channel tears down — the remote `cat` sees EOF and kills the tail. No PTY,
    # so the SSE output path is unchanged (a PTY would inject CRLF and corrupt the
    # `data: …\n\n` framing).
    cmd = (
        f"tail -n 0 -f {log_paths} 2>/dev/null & tailpid=$!; "
        f'cat >/dev/null; kill "$tailpid" 2>/dev/null'
    )

    def stream():
        proc = subprocess.Popen(
            ["ssh", "-q", cluster.host, cmd],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            for line in proc.stdout or []:
                yield f"data: {line.rstrip()}\n\n"
        finally:
            # Closing stdin forwards EOF through the live ssh to the remote `cat`,
            # which then kills the tail and lets the whole chain unwind cleanly.
            if proc.stdin:
                proc.stdin.close()
            proc.terminate()
            proc.wait()

    return Response(stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000, threaded=True)
