PY := ./.venv/bin/python
RUN := PYTHONPATH=. $(PY)

.DEFAULT_GOAL := help
.PHONY: help pc classes deep fluid hjb bp routing alignment test figures experiments lint all

help:
	@echo ""
	@echo "  make pc          predictive coding on MNIST, 10 classes, no backprop   (~2 min)"
	@echo "  make classes     the same, swept over 2 / 3 / 5 / 10 classes           (~4 min)"
	@echo "  make deep        strict vs fixed predictions at 6 hidden layers        (~5 min)"
	@echo "  make fluid       + Navier-Stokes transport, EMNIST-Letters, 26 classes (~6 min)"
	@echo "  make hjb         the same, + Hamilton-Jacobi-Bellman regularisation    (~7 min)"
	@echo "  make bp          backprop baseline, best of a learning-rate sweep      (~1 min)"
	@echo ""
	@echo "  make alignment   the headline finding: PC vs the true BP gradient      (~3 min)"
	@echo "  make routing     the paper's routing task, with baselines it lacks     (~1 min)"
	@echo ""
	@echo "  make test        the full test suite (19 tests)"
	@echo "  make figures     rebuild figures/ from results/"
	@echo "  make lint        ruff"
	@echo ""

# -- training runs -------------------------------------------------------------

pc:
	$(PY) train.py --dataset mnist --epochs 10 --train-subset 0 --test-subset 0 \
	  --hidden 256 128 --weight-lr 0.1 --track-alignment

classes:
	@for k in 2 3 5 10; do \
	  $(PY) train.py --dataset mnist --num-classes $$k --epochs 8 \
	    --hidden 256 128 --weight-lr 0.1 --track-alignment; \
	done

deep:
	@for mode in strict fixed; do \
	  $(PY) train.py --dataset mnist --prediction-mode $$mode --epochs 8 \
	    --hidden 128 128 128 128 128 128 --weight-lr 0.1 --track-alignment; \
	done

fluid:
	$(PY) train.py --dataset emnist_letters --fluid --epochs 4 \
	  --weight-lr 0.1 --fluid-lr 0.01

hjb:
	$(PY) train.py --dataset emnist_letters --fluid --hjb --epochs 4 \
	  --weight-lr 0.1 --fluid-lr 0.01

bp:
	$(PY) train.py --learner bp --dataset mnist --epochs 8 --hidden 256 128

# -- studies -------------------------------------------------------------------

alignment:
	$(RUN) scripts/alignment_study.py

routing:
	$(RUN) scripts/routing_task.py

# -- housekeeping --------------------------------------------------------------

test:
	$(PY) -m pytest -q

figures:
	$(RUN) scripts/make_figures.py

experiments:
	$(RUN) scripts/run_experiments.py

lint:
	$(PY) -m ruff check .

all: test alignment routing experiments figures
