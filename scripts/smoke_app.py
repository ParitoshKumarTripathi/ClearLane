"""Headless Streamlit smoke test used before deployment."""
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
if app.exception:
    for exception in app.exception:
        print(exception.value)
    raise SystemExit(1)
expected_tabs = ["Operations", "Model performance", "Methodology"]
if [tab.label for tab in app.tabs] != expected_tabs:
    raise SystemExit("Unexpected Streamlit tab structure")
if len(app.metric) < 4:
    raise SystemExit("Dashboard metrics did not render")
print("ClearLane Streamlit smoke test passed")
