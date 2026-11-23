import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def main(job: str) -> None:
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=no", "--", "src", "pyproject.toml"):
        sys.exit("uncommitted changes in src/ or pyproject.toml: commit and push first")
    if not git("branch", "-r", "--contains", commit):
        sys.exit(f"commit {commit[:8]} is not on GitHub yet: push first")

    src = Path(__file__).parent / job
    with tempfile.TemporaryDirectory() as tmp:
        for f in src.iterdir():
            shutil.copy(f, tmp)
        run = Path(tmp) / "run.py"
        run.write_text(
            run.read_text(encoding="utf-8").replace("__COMMIT__", commit), encoding="utf-8"
        )
        kaggle = Path(sys.executable).with_name("kaggle")
        subprocess.run([str(kaggle), "kernels", "push", "-p", tmp], check=True)
    print(f"pushed {job} at {commit[:8]}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "smoke")
