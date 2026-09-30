"""Build the static site for GitHub Pages: the page plus pre-calculated data for the example team.

Runs the backend in-process (no server needed) and writes everything to `site/`:
    python3 scripts/build_static.py

Environment variables:
    XPOINTS_API_URL   the calculation server for teams other than the example (e.g. the Render URL)
    EXAMPLE_TEAM_ID   the example team (default 408324)
    SITE_DOMAIN       optional custom domain, written to site/CNAME
"""
import json
import os
import shutil
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")
SITE = os.path.join(ROOT, "site")
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

os.environ.setdefault("WARM_EXAMPLE", "0")
import app as backend  # noqa: E402

EXAMPLE = int(os.environ.get("EXAMPLE_TEAM_ID", "408324"))
API = os.environ.get("XPOINTS_API_URL", "").rstrip("/")


def fetch(client, path):
    response = client.get(path)
    data = response.get_json()
    if response.status_code != 200 or "error" in data:
        raise SystemExit(f"{path} failed: {data.get('error', response.status_code)}")
    return data


def main():
    shutil.rmtree(SITE, ignore_errors=True)
    os.makedirs(os.path.join(SITE, "data"))
    client = backend.app.test_client()

    files = {
        "config": {"example_team_id": EXAMPLE},
        "analysis-example": fetch(client, f"/analysis?team_id={EXAMPLE}"),
        "chips-example": fetch(client, f"/chips?team_id={EXAMPLE}"),
        "accuracy-example": fetch(client, f"/accuracy?team_id={EXAMPLE}"),
        "price-changes": fetch(client, "/price-changes"),
    }
    for name, data in files.items():
        with open(os.path.join(SITE, "data", f"{name}.json"), "w") as f:
            json.dump(data, f)

    shutil.copy(os.path.join(ROOT, "frontend", "index.html"), os.path.join(SITE, "index.html"))
    config = {"static": True, "api": API, "exampleTeamId": EXAMPLE,
              "generated": datetime.now(timezone.utc).isoformat()}
    with open(os.path.join(SITE, "config.js"), "w") as f:
        f.write(f"window.XPOINTS_CONFIG = {json.dumps(config)};\n")
    open(os.path.join(SITE, ".nojekyll"), "w").close()  # serve files as-is
    domain = os.environ.get("SITE_DOMAIN", "").strip()
    if domain:
        with open(os.path.join(SITE, "CNAME"), "w") as f:
            f.write(domain + "\n")
    print(f"Built {SITE} (GW{files['analysis-example']['gameweek']}, API: {API or 'none'})")


if __name__ == "__main__":
    main()
