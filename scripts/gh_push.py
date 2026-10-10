#!/usr/bin/env python3
"""ConvexForge GitHub publisher with three-level graceful degradation.

Strategy (each level tried only if the previous one fails):
  L1  git + gh       : `git` CLI commit + `gh` push / repo create (preferred).
  L2  Git Data API   : `gh api` blobs -> tree -> commit -> ref (no git remote).
  L3  Contents API   : `gh api` repos/<r>/contents/<path> create/update per file.

Usage:
    python scripts/gh_push.py [--repo CJX0712/convexforge] [--tag v0.1.0] [--no-release]

The file list is discovered from the working tree (not hard-coded), so no source
file is silently left behind. Author: 晨星.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys

REPO_DEFAULT = "CJX0712/convexforge"
TAG_DEFAULT = "v0.1.0"
BRANCH = "main"

DESCRIPTION = (
    "World-class KKT-certified convex optimization suite: penalty / ADMM / "
    "Chambolle-Pock / Mehrotra interior point + ConvFuse regime router. Author: 晨星."
)

# Directories never published.
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "build",
    "dist",
    ".workbuddy",
    "node_modules",
}
# File suffixes never published (binary / secrets / editor junk).
EXCLUDE_SUFFIX = {
    ".pyc",
    ".pyo",
    ".so",
    ".dll",
    ".exe",
    ".npz",
    ".npy",
    ".key",
    ".pem",
    ".p12",
    ".pfx",
    ".env",
    ".bak",
    ".tmp",
    ".swp",
}
# Explicitly excluded names.
EXCLUDE_NAMES = {".DS_Store", "Thumbs.db", ".env", "id_rsa", "credentials.json"}


def discover(root: str) -> list[str]:
    """Every publishable file, as repo-relative POSIX paths, sorted."""
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if fn in EXCLUDE_NAMES:
                continue
            if os.path.splitext(fn)[1] in EXCLUDE_SUFFIX:
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), root).replace(os.sep, "/")
            out.append(rel)
    return sorted(out)


def _run(cmd: list[str], check: bool = True):
    print("  $ " + " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if check and res.returncode != 0:
        raise RuntimeError(
            f"command failed ({res.returncode}): {' '.join(cmd)}\n"
            f"stdout: {res.stdout}\nstderr: {res.stderr}"
        )
    return res


def _gh_api(method: str, path: str, data: dict | None = None) -> dict:
    # Critical: under Git-Bash/MSYS a leading "/" is rewritten to a Windows
    # path (e.g. ".../PortableGit/repos/..."), producing 400/409. Always pass
    # the endpoint WITHOUT a leading slash.
    path = path.lstrip("/")
    cmd = ["gh", "api", "--method", method, path]
    payload = None
    if data is not None:
        cmd += ["--input", "-"]
        payload = json.dumps(data).encode("utf-8")

    # Retry with exponential backoff: the GitHub API gateway occasionally
    # returns 502/503 under load; these are transient and heal on retry.
    last_exc: Exception | None = None
    for attempt in range(6):
        res = subprocess.run(cmd, input=payload, capture_output=True)
        if res.returncode == 0:
            if not res.stdout.strip():
                return {}
            return json.loads(res.stdout)
        stderr = res.stderr.decode("utf-8", "replace")
        if "Bad Gateway" in stderr or "Service Unavailable" in stderr or "502" in stderr or "503" in stderr:
            import time

            time.sleep(1.5 ** attempt)
            last_exc = RuntimeError(f"gh api {method} {path}: {stderr.strip()}")
            continue
        raise RuntimeError(
            f"gh api {method} {path} failed\nstdout: {res.stdout}\nstderr: {stderr}"
        )
    raise last_exc if last_exc else RuntimeError(f"gh api {method} {path} failed after retries")


def _release(repo: str, tag: str, make_release: bool, notes: str) -> None:
    if not make_release:
        return
    exists = _run(["gh", "release", "view", tag, "--repo", repo], check=False)
    if exists.returncode == 0:
        print(f"  (release {tag} already exists)")
        return
    _run(
        [
            "gh",
            "release",
            "create",
            tag,
            "--repo",
            repo,
            "--title",
            f"ConvexForge {tag}",
            "--notes",
            notes,
            "--latest",
        ],
        check=False,
    )


# ---------------------------------------------------------------------------
# L1: git + gh
# ---------------------------------------------------------------------------
def push_l1(repo: str, tag: str, make_release: bool, files: list[str]) -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    if not os.path.isdir(".git"):
        _run(["git", "init", "-q", "-b", BRANCH])
    _run(["git", "config", "user.name", "晨星"])
    _run(["git", "config", "user.email", "CJX0712@users.noreply.github.com"])

    for f in files:
        if os.path.exists(f):
            _run(["git", "add", f])

    res = _run(
        ["git", "commit", "-q", "-m", f"ConvexForge {tag} — KKT-certified convex optimization (author: 晨星)"],
        check=False,
    )
    if res.returncode != 0:
        print("  (nothing new to commit or commit skipped)")

    _run(["gh", "auth", "setup-git"])

    view = _run(["gh", "repo", "view", repo], check=False)
    if view.returncode != 0:
        _run(
            [
                "gh",
                "repo",
                "create",
                repo,
                "--public",
                "--description",
                DESCRIPTION,
                "--source",
                ".",
                "--push",
            ]
        )
    else:
        if "origin" not in _run(["git", "remote"], check=False).stdout:
            _run(["git", "remote", "add", "origin", f"https://github.com/{repo}.git"])
        # Must raise on failure so the caller degrades to L2/L3.
        _run(["git", "push", "-u", "origin", BRANCH])

    _run(["git", "tag", "-f", tag], check=False)
    _run(["git", "push", "-f", "origin", tag], check=False)
    _release(repo, tag, make_release, "Initial release. See README.md and docs/model_card.md. Author: 晨星.")


# ---------------------------------------------------------------------------
# L2: Git Data API
# ---------------------------------------------------------------------------
def push_l2(repo: str, tag: str, make_release: bool, files: list[str]) -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_sha = None
    try:
        info = _gh_api("GET", f"repos/{repo}")
        default_branch = info.get("default_branch", BRANCH)
        ref = _gh_api("GET", f"repos/{repo}/git/refs/heads/{default_branch}")
        base_sha = ref["object"]["sha"]
    except RuntimeError:
        base_sha = None  # brand-new empty repo

    blobs = []
    for f in files:
        p = os.path.join(root, f)
        if not os.path.exists(p):
            continue
        with open(p, "rb") as fh:
            content = base64.b64encode(fh.read()).decode("ascii")
        blob = _gh_api(
            "POST", f"repos/{repo}/git/blobs", {"content": content, "encoding": "base64"}
        )
        blobs.append({"path": f, "mode": "100644", "type": "blob", "sha": blob["sha"]})

    tree_payload: dict = {"tree": blobs}
    if base_sha:
        tree_payload["base_tree"] = base_sha
    tree = _gh_api("POST", f"repos/{repo}/git/trees", tree_payload)

    commit = _gh_api(
        "POST",
        f"repos/{repo}/git/commits",
        {
            "message": f"ConvexForge {tag} — author: 晨星",
            "tree": tree["sha"],
            "parents": [base_sha] if base_sha else [],
        },
    )

    if base_sha:
        _gh_api("PATCH", f"repos/{repo}/git/refs/heads/{BRANCH}", {"sha": commit["sha"]})
    else:
        try:
            _gh_api(
                "POST",
                f"repos/{repo}/git/refs",
                {"ref": f"refs/heads/{BRANCH}", "sha": commit["sha"]},
            )
        except RuntimeError:
            _gh_api("PATCH", f"repos/{repo}/git/refs/heads/{BRANCH}", {"sha": commit["sha"]})

    # Tag object pointing at the commit.
    try:
        tag_obj = _gh_api(
            "POST",
            f"repos/{repo}/git/tags",
            {
                "tag": tag,
                "message": f"ConvexForge {tag}",
                "object": commit["sha"],
                "type": "commit",
                "tagger": {"name": "晨星", "email": "CJX0712@users.noreply.github.com"},
            },
        )
        _gh_api(
            "POST",
            f"repos/{repo}/git/refs",
            {"ref": f"refs/tags/{tag}", "sha": tag_obj["sha"]},
        )
    except RuntimeError as exc:
        print(f"  (tag creation skipped: {exc})")

    _release(repo, tag, make_release, "Published via Git Data API. Author: 晨星.")


# ---------------------------------------------------------------------------
# L3: Contents API (per-file create/update)
# ---------------------------------------------------------------------------
def push_l3(repo: str, tag: str, make_release: bool, files: list[str]) -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for f in files:
        p = os.path.join(root, f)
        if not os.path.exists(p):
            continue
        with open(p, "rb") as fh:
            content = base64.b64encode(fh.read()).decode("ascii")
        sha = None
        res = subprocess.run(
            ["gh", "api", f"repos/{repo}/contents/{f}"], capture_output=True, text=True
        )
        if res.returncode == 0:
            try:
                sha = json.loads(res.stdout).get("sha")
            except json.JSONDecodeError:
                sha = None
        payload = {
            "message": f"ConvexForge {tag}: {f} (author: 晨星)",
            "content": content,
            "branch": BRANCH,
        }
        if sha:
            payload["sha"] = sha
        res2 = subprocess.run(
            ["gh", "api", "--method", "PUT", f"repos/{repo}/contents/{f}", "--input", "-"],
            input=json.dumps(payload).encode("utf-8"),
            capture_output=True,
            text=True,
        )
        if res2.returncode != 0:
            raise RuntimeError(f"L3 failed for {f}: {res2.stderr}")
    _release(repo, tag, make_release, "Published via Contents API fallback. Author: 晨星.")


def main() -> int:
    ap = argparse.ArgumentParser(description="ConvexForge GitHub publisher")
    ap.add_argument("--repo", default=REPO_DEFAULT)
    ap.add_argument("--tag", default=TAG_DEFAULT)
    ap.add_argument("--no-release", action="store_true")
    args = ap.parse_args()
    make_release = not args.no_release

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = discover(root)
    print(f"[push] discovered {len(files)} files")
    for f in files:
        print(f"       {f}")

    levels = [
        ("L1 git+gh", push_l1),
        ("L2 Git Data API", push_l2),
        ("L3 Contents API", push_l3),
    ]
    last_exc: Exception | None = None
    for name, fn in levels:
        try:
            print(f"[push] trying {name} ...")
            fn(args.repo, args.tag, make_release, files)
            print(f"[push] SUCCESS via {name}")
            return 0
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"[push] {name} failed: {exc}")
            print("[push] degrading to next level ...")
    print(f"[push] ALL LEVELS FAILED: {last_exc}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
