PYTHON ?= python
DATA ?= data/raw/police_violations.csv

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

pipeline:
	PYTHONPATH=src $(PYTHON) -m clearlane.cli pipeline --input "$(DATA)"

train:
	PYTHONPATH=src $(PYTHON) -m clearlane.cli train --input "$(DATA)"

app:
	streamlit run app.py

test:
	PYTHONPATH=src pytest -q
	PYTHONPATH=src $(PYTHON) scripts/smoke_app.py

lint:
	ruff check .
