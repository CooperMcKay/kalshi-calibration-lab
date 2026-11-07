PY ?= python

.PHONY: test analyze dashboard

test:
	$(PY) -m pytest -q

analyze:
	$(PY) -m kalshi_lab.calibration
	$(PY) -m kalshi_lab.model
	$(PY) -m kalshi_lab.paper_compare

# Builds dashboard/dashboard.html from YOUR local data/ (gitignored) and opens it.
dashboard:
	$(PY) -m kalshi_lab.dashboard
	open dashboard/dashboard.html
