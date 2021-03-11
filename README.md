# markov_blankets
Discovering Markov blankets in static and sequence data

# How to use
* Install python. Currently supported versions: >=3.7.1 <3.9
```bash
python --version
```
* [optionally] Create a virtual environment and activate it
```bash
python -m venv venv/
source venv/bin/activate
```
* Install [poetry](https://python-poetry.org/docs/#installation)
* Install project along with dependencies
```bash
poetry install
```

## How to run the tests
* Use `pytest` to run all tests
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
* Use poetry utility, to add a new dependency: `poetry add xxx`

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
