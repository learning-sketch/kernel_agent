# Local pre-ship checks. There is no CI in this repository: run `make preship` before pushing.
#
#   make test       suite on the default interpreter
#   make test-310   suite on Python 3.10, the minimum version declared in pyproject.toml
#   make preship    both of the above (what must be green before a push)
#   make tox        all supported interpreters via tox (needs `pip install tox`)

PY      ?= python3
PY310   ?= $(shell command -v python3.10 2>/dev/null)
UV      ?= $(shell command -v uv 2>/dev/null)
PKGS     = kopt_agent ops tests
PYTEST   = -m pytest -q $(PYTEST_ARGS)

.PHONY: test test-310 preship tox clean

test:
	$(PY) -m compileall -q $(PKGS)
	$(PY) $(PYTEST)

# uv provisions 3.10 + numpy + pytest on demand (`uv python install 3.10` once); without uv a
# python3.10 with numpy and pytest installed is used directly.
test-310:
ifneq ($(UV),)
	$(UV) run --python 3.10 --no-project --with "pytest>=8.0" --with "numpy>=1.26" python -m compileall -q $(PKGS)
	$(UV) run --python 3.10 --no-project --with "pytest>=8.0" --with "numpy>=1.26" python $(PYTEST)
else ifneq ($(PY310),)
	$(PY310) -m compileall -q $(PKGS)
	$(PY310) $(PYTEST)
else
	@echo "need python3.10 (or uv: 'uv python install 3.10') to run the minimum-version check"; exit 1
endif

preship: test-310 test
	@echo "pre-ship checks passed on 3.10 and $(shell $(PY) -c 'import sys; print("%d.%d" % sys.version_info[:2])')"

tox:
	tox -q

clean:
	rm -rf .pytest_cache .tox results core core.* 
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
