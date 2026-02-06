SHELL := /bin/bash

.PHONY: help setup dev backend ui stop status clean-pids ingest propagate grid validate validate-science orbit-preview-gif full-pipeline full-pipeline-offline docker-up docker-down public-audit

BACKEND_HOST ?= 127.0.0.1
BACKEND_PORT ?= 8000
UI_HOST ?= 127.0.0.1
UI_PORT ?= 5173

PID_DIR := .pids
BACKEND_PID := $(PID_DIR)/backend.pid
UI_PID := $(PID_DIR)/ui.pid

help:
	@echo "Targets:"
	@echo "  make setup      - Installe dependances Python + UI"
	@echo "  make dev        - Lance backend + UI (Ctrl-C pour tout arreter)"
	@echo "  make backend    - Lance seulement le backend FastAPI"
	@echo "  make ui         - Lance seulement le frontend Vite"
	@echo "  make ingest     - Ingestion TLE (Space-Track/CelesTrak)"
	@echo "  make propagate  - Propagation SGP4 offline -> samples"
	@echo "  make grid       - Build des features par cellule"
	@echo "  make validate   - Checks features de base"
	@echo "  make validate-science - Rapport scientifique markdown"
	@echo "  make orbit-preview-gif - Regenere le GIF orbit preview du README"
	@echo "  make full-pipeline    - ingest + propagate + grid + validations"
	@echo "  make full-pipeline-offline - pipeline sans ingest (reproductible sur latest.txt)"
	@echo "  make stop       - Stoppe backend + UI lances via make"
	@echo "  make status     - Affiche l'etat (PIDs) si lances via make"
	@echo "  make docker-up  - Lance backend + UI via docker compose"
	@echo "  make docker-down - Stoppe docker compose"
	@echo "  make public-audit - Pre-check avant publication GitHub"
	@echo ""
	@echo "Options:"
	@echo "  BACKEND_PORT=8001 UI_PORT=5174 make dev"

$(PID_DIR):
	@mkdir -p $(PID_DIR)

status:
	@set -euo pipefail; \
	for f in "$(BACKEND_PID)" "$(UI_PID)"; do \
	  if [[ -f "$$f" ]]; then \
	    pid="$$(cat "$$f" || true)"; \
	    if [[ -n "$$pid" ]] && kill -0 "$$pid" 2>/dev/null; then \
	      echo "$$f -> RUNNING (pid=$$pid)"; \
	    else \
	      echo "$$f -> NOT RUNNING"; \
	    fi; \
	  else \
	    echo "$$f -> (missing)"; \
	  fi; \
	done

stop:
	@set -euo pipefail; \
	stop_one() { \
	  local f="$$1"; \
	  if [[ ! -f "$$f" ]]; then return 0; fi; \
	  local pid="$$(cat "$$f" || true)"; \
	  if [[ -z "$$pid" ]]; then rm -f "$$f"; return 0; fi; \
	  if kill -0 "$$pid" 2>/dev/null; then \
	    echo "Stopping pid $$pid ($$f)"; \
	    pkill -TERM -P "$$pid" 2>/dev/null || true; \
	    kill -TERM "$$pid" 2>/dev/null || true; \
	    for _ in 1 2 3 4 5 6 7 8 9 10; do \
	      if kill -0 "$$pid" 2>/dev/null; then sleep 0.15; else break; fi; \
	    done; \
	    if kill -0 "$$pid" 2>/dev/null; then \
	      pkill -KILL -P "$$pid" 2>/dev/null || true; \
	      kill -KILL "$$pid" 2>/dev/null || true; \
	    fi; \
	  fi; \
	  rm -f "$$f"; \
	}; \
	stop_one "$(UI_PID)"; \
	stop_one "$(BACKEND_PID)"; \
	rmdir "$(PID_DIR)" 2>/dev/null || true

clean-pids:
	@rm -rf $(PID_DIR)

setup:
	@set -euo pipefail; \
	echo "Installing Python dependencies..."; \
	python3 -m pip install -r requirements.txt; \
	echo "Installing UI dependencies..."; \
	cd ui && npm ci

ingest:
	@python3 src/ingest.py

propagate:
	@python3 src/propagate.py

grid:
	@python3 src/grid_builder.py

validate:
	@python3 src/validate_features.py

validate-science:
	@python3 src/validate_science.py --output docs/validation_reports/latest.md

orbit-preview-gif:
	@python3 scripts/generate_orbit_preview_gif.py --ui-url http://$(UI_HOST):$(UI_PORT) --norad 25544 --gif-out docs/assets/ui-orbit-preview.gif --png-out docs/assets/ui-orbit-preview.png

full-pipeline:
	@set -euo pipefail; \
	python3 src/ingest.py; \
	python3 src/propagate.py; \
	python3 src/grid_builder.py; \
	python3 src/validate_features.py; \
	python3 src/validate_science.py --output docs/validation_reports/latest.md

full-pipeline-offline:
	@set -euo pipefail; \
	test -f data/tle/latest.txt; \
	python3 src/propagate.py; \
	python3 src/grid_builder.py; \
	python3 src/validate_features.py; \
	python3 src/validate_science.py --output docs/validation_reports/latest.md

docker-up:
	@docker compose up --build

docker-down:
	@docker compose down

public-audit:
	@./scripts/public_audit.sh

backend: | $(PID_DIR)
	@set -euo pipefail; \
	if lsof -nP -iTCP:$(BACKEND_PORT) -sTCP:LISTEN >/dev/null 2>&1; then \
	  echo "Port $(BACKEND_PORT) deja utilise. Change BACKEND_PORT=... ou stoppe le process."; \
	  exit 1; \
	fi; \
	echo "Backend: http://$(BACKEND_HOST):$(BACKEND_PORT)"; \
	python3 -m uvicorn src.api:app --reload --host $(BACKEND_HOST) --port $(BACKEND_PORT) & \
	echo $$! > "$(BACKEND_PID)"; \
	wait $$(cat "$(BACKEND_PID)")

ui: | $(PID_DIR)
	@set -euo pipefail; \
	if lsof -nP -iTCP:$(UI_PORT) -sTCP:LISTEN >/dev/null 2>&1; then \
	  echo "Port $(UI_PORT) deja utilise. Change UI_PORT=... ou stoppe le process."; \
	  exit 1; \
	fi; \
	echo "UI: http://$(UI_HOST):$(UI_PORT)"; \
	( cd ui && npm run dev -- --host $(UI_HOST) --port $(UI_PORT) ) & \
	echo $$! > "$(UI_PID)"; \
	wait $$(cat "$(UI_PID)")

dev: | $(PID_DIR)
	@set -euo pipefail; \
	trap '$(MAKE) stop' INT TERM EXIT; \
	if lsof -nP -iTCP:$(BACKEND_PORT) -sTCP:LISTEN >/dev/null 2>&1; then \
	  echo "Port $(BACKEND_PORT) deja utilise. Change BACKEND_PORT=... ou stoppe le process."; \
	  exit 1; \
	fi; \
	if lsof -nP -iTCP:$(UI_PORT) -sTCP:LISTEN >/dev/null 2>&1; then \
	  echo "Port $(UI_PORT) deja utilise. Change UI_PORT=... ou stoppe le process."; \
	  exit 1; \
	fi; \
	echo "Backend: http://$(BACKEND_HOST):$(BACKEND_PORT)"; \
	python3 -m uvicorn src.api:app --reload --host $(BACKEND_HOST) --port $(BACKEND_PORT) & \
	echo $$! > "$(BACKEND_PID)"; \
	sleep 0.6; \
	echo "UI: http://$(UI_HOST):$(UI_PORT)"; \
	( cd ui && npm run dev -- --host $(UI_HOST) --port $(UI_PORT) ) & \
	echo $$! > "$(UI_PID)"; \
	wait
