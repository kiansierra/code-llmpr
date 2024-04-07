style:
	isort src scripts
	ruff --fix src scripts
	black src scripts

check:
	black --check src scripts
	ruff src scripts
	pylint -j 8 --fail-under=8 src

test:
	rm -r build
	rm -r src/llm_prompt.egg-info
	pip install -e .
	pytest tests