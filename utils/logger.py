# utils/run_logger.py
from __future__ import annotations

import json
import platform
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe(obj: Any) -> Any:
    """Best-effort JSON serialization."""
    try:
        json.dumps(obj)
        return obj
    except Exception:
        try:
            return json.loads(json.dumps(obj, default=str))
        except Exception:
            return str(obj)


def _git_cmd(args: list[str], cwd: Path) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
    except FileNotFoundError:
        return 127, "", "git not found"
    except Exception as e:
        return 1, "", f"{type(e).__name__}: {e}"


def capture_git_state(repo_dir: Path = ROOT) -> Dict[str, Any]:
    """Collects git metadata & diff. Gracefully degrades if not a git repo."""
    state: Dict[str, Any] = {"is_git_repo": False}

    rc, head, err = _git_cmd(["rev-parse", "HEAD"], repo_dir)
    if rc != 0:
        state["error"] = err or "Not a git repo?"
        return state

    state["is_git_repo"] = True
    state["commit"] = head

    rc, branch, _ = _git_cmd(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir)
    state["branch"] = branch if rc == 0 else None

    rc, porcelain, _ = _git_cmd(["status", "--porcelain"], repo_dir)
    state["dirty"] = bool(porcelain) if rc == 0 else None
    state["status_porcelain"] = porcelain if rc == 0 else None

    rc, remotes, _ = _git_cmd(["remote", "-v"], repo_dir)
    state["remotes"] = remotes if rc == 0 else None

    # Full diff (staged + unstaged)
    rc, diff, _ = _git_cmd(["diff"], repo_dir)
    state["diff"] = diff if rc == 0 else None

    # Include staged diff as well
    rc, diff_cached, _ = _git_cmd(["diff", "--cached"], repo_dir)
    state["diff_cached"] = diff_cached if rc == 0 else None

    return state


def _sanitize_ts(ts: str) -> str:
    """Make a filesystem-friendly timestamp slug."""
    # 2025-09-24T12:34:56.789012+00:00 -> 2025-09-24T12-34-56Z
    return (
        ts.replace(":", "-")
        .replace("+00:00", "Z")
        .replace("+", "p")
        .replace("-", "-")
        .replace("/", "_")
    )


def log_run(
    *,
    logdir: Path,
    config: Dict[str, Any],
    args: Dict[str, Any],
    results: Optional[Dict[str, Any]],
    git_state: Optional[Dict[str, Any]] = None,
    status: str = "success",
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Writes a JSON bundle with config, git state, results and runtime info.
    Returns paths of created artifacts.
    """
    logdir.mkdir(parents=True, exist_ok=True)
    ts = _sanitize_ts(_now_iso())
    run_id = uuid.uuid4().hex[:8]
    stem = f"run_{ts}_{run_id}"

    payload = {
        "run_id": run_id,
        "timestamp_utc": ts,
        "status": status,
        "cli_args": _safe(args),
        "config": _safe(config),
        "results": _safe(results),
        "git": _safe(git_state or {}),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "cwd": str(Path.cwd()),
        },
        "error": _safe(error) if error else None,
    }

    json_path = logdir / f"{stem}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Also save raw config & git patches as companions for convenience
    cfg_path = logdir / f"{stem}.config.yaml"
    try:
        import yaml  # local dep already present
        cfg_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    except Exception:
        # Fallback to JSON if yaml fails
        cfg_path = logdir / f"{stem}.config.json"
        cfg_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    patch_path = None
    if (git_state or {}).get("is_git_repo"):
        patch_path = logdir / f"{stem}.diff.patch"
        diff_all = (git_state or {}).get("diff") or ""
        diff_cached = (git_state or {}).get("diff_cached") or ""
        patch_path.write_text(
            f"### unstaged diff ###\n{diff_all}\n\n### staged diff (cached) ###\n{diff_cached}\n",
            encoding="utf-8",
        )

    return {"json": str(json_path), "config": str(cfg_path), "patch": str(patch_path) if patch_path else None}
