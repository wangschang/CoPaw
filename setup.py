# -*- coding: utf-8 -*-
import os
import shutil
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py


def _build_console():
    """Build the console frontend and copy to src/copaw/console/.

    Runs automatically during ``pip install -e .`` (or any source build).
    Skipped gracefully when:
    - the destination already contains an ``index.html`` (already built), or
    - ``console/package.json`` is absent (not a full checkout), or
    - ``npm`` is not on PATH (warns the user instead of failing).
    """
    repo_root = os.path.dirname(os.path.abspath(__file__))
    console_dir = os.path.join(repo_root, "console")
    console_dest = os.path.join(repo_root, "src", "copaw", "console")
    dist_dir = os.path.join(console_dir, "dist")

    # Already copied — nothing to do.
    if os.path.isfile(os.path.join(console_dest, "index.html")):
        return

    # Pre-built dist/ present (e.g. developer already ran npm build) — just copy.
    if os.path.isfile(os.path.join(dist_dir, "index.html")):
        print("[setup] Copying pre-built console assets to src/copaw/console/ ...")
        os.makedirs(console_dest, exist_ok=True)
        for name in os.listdir(dist_dir):
            src = os.path.join(dist_dir, name)
            dst = os.path.join(console_dest, name)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        return

    # No pre-built assets — need npm.
    if not os.path.isfile(os.path.join(console_dir, "package.json")):
        print("[setup] console/package.json not found; skipping frontend build.")
        return

    npm = shutil.which("npm")
    if npm is None:
        print(
            "[setup] WARNING: npm not found — skipping console frontend build.\n"
            "        Install Node.js (https://nodejs.org/) then run:\n"
            "          cd console && npm ci && npm run build\n"
            "          mkdir -p src/copaw/console\n"
            "          cp -R console/dist/. src/copaw/console/\n"
            "        The web UI will be unavailable without this step."
        )
        return

    print("[setup] Building console frontend (npm ci && npm run build) ...")
    try:
        subprocess.check_call([npm, "ci"], cwd=console_dir)
        subprocess.check_call([npm, "run", "build"], cwd=console_dir)
    except subprocess.CalledProcessError as exc:
        print(
            f"[setup] WARNING: console build failed ({exc}); "
            "the web UI won't be available."
        )
        return

    if not os.path.isfile(os.path.join(dist_dir, "index.html")):
        print(
            "[setup] WARNING: console build produced no index.html; "
            "the web UI won't be available."
        )
        return

    print("[setup] Copying console/dist/* -> src/copaw/console/ ...")
    os.makedirs(console_dest, exist_ok=True)
    for name in os.listdir(dist_dir):
        src = os.path.join(dist_dir, name)
        dst = os.path.join(console_dest, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    print("[setup] Console frontend ready.")


class BuildPy(build_py):
    """Extend the standard build_py step to include the console frontend."""

    def run(self):
        _build_console()
        super().run()


setup(cmdclass={"build_py": BuildPy})
