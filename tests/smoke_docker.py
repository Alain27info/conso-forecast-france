"""Smoke test of a RUNNING service (typically the Docker container): checks that the whole chain works.

Usage (not collected by pytest, it needs a live server):
    python tests/smoke_docker.py [base_url]        # default http://127.0.0.1:8000
Exit code 0 = every check passed, 1 = at least one failed.
"""
import math
import sys

import httpx

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"
results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> bool:
    results.append((name, condition, detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}{' : ' + detail if detail else ''}")
    return condition


def main() -> int:
    client = httpx.Client(base_url=BASE, timeout=60)
    health = client.get("/health")
    check("GET /health answers 200 with status ok", health.status_code == 200 and health.json().get("status") == "ok", health.text[:120])
    page = client.get("/")
    check("GET / serves the web page", page.status_code == 200 and "conso-forecast" in page.text)
    check("static files are served", client.get("/static/app.js").status_code == 200 and client.get("/static/results.json").status_code == 200)
    docs = client.get("/openapi.json").json()
    check("the API contract exposes /health and /predict", {"/health", "/predict"} <= set(docs["paths"]), str(sorted(docs["paths"])))

    span = client.get("/demo/range")
    if not check("GET /demo/range answers 200 (the dataset is inside the image)", span.status_code == 200, span.text[:150]):
        return finish()
    last_date = span.json()["last_date"]
    demo = client.get("/demo/day", params={"date": last_date}).json()
    response = client.post("/predict", json=demo["request"])
    values = [p["consumption_mw"] for p in response.json().get("predictions", [])] if response.status_code == 200 else []
    check(f"POST /predict for {last_date} answers 200 with 48 half-hours", response.status_code == 200 and len(values) == 48, f"HTTP {response.status_code}, {len(values)} values")
    check("the forecast values are finite and plausible (20 000 to 110 000 MW)", bool(values) and all(math.isfinite(v) and 20_000 < v < 110_000 for v in values),
          f"min {min(values):.0f}, max {max(values):.0f}" if values else "no values")

    forbidden = {**demo["request"], "date": demo["request"]["history"][-1]["timestamp"][:10]}
    refused = client.post("/predict", json=forbidden)
    check("a request containing data from the target day is refused with 422", refused.status_code == 422, refused.text[:160])
    check("a malformed request is refused with 422", client.post("/predict", json={"date": "oops"}).status_code == 422)
    return finish()


def finish() -> int:
    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)} / {len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
