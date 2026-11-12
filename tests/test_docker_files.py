"""Checks on the Docker files themselves: they run without Docker and guard the mistakes that matter."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _pins(path: Path) -> dict[str, str]:
    """name -> version for every requirement line; fails if a line is not pinned with ==."""
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.\-+]+)", line)
        assert match, f"{path.name}: '{line}' is not pinned with =="
        pins[match.group(1).lower()] = match.group(2)
    return pins


def test_dockerignore_keeps_private_and_heavy_files_out_of_the_image():
    ignored = {line.strip() for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()}
    for name in (".git", ".venv", "docs", "CLAUDE.md", "PLAN.md", "data", "models"):
        assert name in ignored, f".dockerignore must exclude {name}"


def test_dockerfile_follows_the_basic_safety_rules():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^FROM python:3\.\d+-slim$", text, re.MULTILINE), "pin the base image to a version, never :latest"
    assert re.search(r"^USER app$", text, re.MULTILINE), "the container must not run as root"
    assert "HEALTHCHECK" in text
    assert '"--host", "0.0.0.0"' in text, "listening on 127.0.0.1 makes the server unreachable from outside the container"
    assert "download.pytorch.org/whl/cpu" in text, "install the CPU-only torch wheel (the default one is several GB)"
    instructions = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "chown -R" not in instructions, "use COPY --chown: a recursive chown duplicates the files in a new layer"
    assert text.index("USER app") < text.index("RUN python -m src.data"), "data and model must be created by the non-root user"
    assert text.index("COPY requirements-api.txt") < text.index("COPY --chown=app:app src/"), "dependencies must come before the code (layer cache)"
    assert text.index("python -m src.train_final") < text.index("COPY --chown=app:app api/"), "the page must not invalidate the trained-model layer"


def test_api_requirements_are_a_pinned_subset_of_the_project_requirements():
    api, full = _pins(ROOT / "requirements-api.txt"), _pins(ROOT / "requirements.txt")
    assert {"torch", "fastapi", "uvicorn", "pandas", "numpy", "pydantic", "httpx"} <= set(api)
    for name, version in api.items():
        assert full.get(name) == version, f"{name}: requirements-api.txt has {version}, requirements.txt has {full.get(name)}"


def test_the_api_does_not_import_packages_missing_from_the_slim_requirements():
    """Importing the app must not need the experiment-only libraries that the image leaves out."""
    code = ("import sys, api.main; "
            "left_out = [m for m in ('sklearn', 'statsmodels', 'matplotlib', 'IPython', 'pytest') if m in sys.modules]; "
            "print(left_out)")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]", f"the API imports packages the image does not install: {result.stdout.strip()}"


def test_compose_publishes_the_api_port_and_builds_from_the_dockerfile():
    text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "build: ." in text and ":8000" in text
