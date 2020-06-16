# markov_blankets
Discovering Markov blankets in static and sequence data

# How to use
* Install python 3.7 or later 
```bash
python -version
```
* Create a virtual environment 
```bash
python -m venv .venv/
```
* Install dependencies
```bash
pip install -r requirements.txt
```
* Install the project module
```bash
pip install -e .
```
* Use the library
```python
from markov_blankets.utilities.tools import make_single_column
```

# How to develop
* Follow steps [above](#how-to-use)
* Install dev dependencies
```bash
pip install -r requirements-dev.txt
```

## How to run the tests
* Use `pytest`, which is installed as part of `requirements-dev.txt`
* Run all tests
```bash
pytest
```
* or run all tests within a module
```bash
pytest tests/utilities/tools.py
```
* or run a specific test
```bash
pytest tests/utilities/tools.py::test_choose_no_overflow
```


## How to add a dependency
* Use `pip-compile` utility, which is installed as part of `requirements-dev.txt`
* Add required library to either `requirements.in` or `requirements-dev.in`
* Resolve latest package dependencies.
```bash
pip-compile requirements.in
pip-compile requirements-dev.in
```

## How to format code
* Use `black` utility, which is installed as part of `requirements-dev.txt`
* Reformat all python files under current folder and subfolders 
```bash
black .
```
* Check if format is acceptable 
```bash
black --check .
```
