# explora
Explora discovers Markov blankets and approximate functional dependencies for supervised feature selection tasks.

## What is it?
**explora** is a Python package that provides fast and flexible algorithms for 
greedily maximizing dependence measures such as Mutual Information.

[comment]: <> (#TODO add theory background information)


## How to use
In order to use `explora` python library needs to be installed. Follow the [installation
instructions](#installation-from-sources). After `explora` has been installed
one can simply import it and use its functions. For example:
```python
import pandas as pd
from explora.algorithms.stochastic_markov_blanket_discovery import stochastic_markov_blanket_discovery
from explora.information_theory.estimators import fraction_of_information_permutation

data = pd.DataFrame(...)
result, score = stochastic_markov_blanket_discovery(fraction_of_information_permutation, data)
print(f'selected attributes with permutation {result} with score {score}')
```

For more examples, see [examples](examples) folder


## Installation from sources
To install explora from source you need Python 3.7.1 or later and earlier than 3.9.
If such version is not installed, [pyenv](https://github.com/pyenv/pyenv) is recommended to install and manage multiple versions.

We recommend using a virtual environment in order to isolate the dependencies of this project.
The virtual environment folder is usually in the same folder as the project, but any path would work.
To create a virtual environment execute: 
```bash
python -m venv explora_venv/
source explora_venv/bin/activate
```
Read more about [python venv module](https://docs.python.org/3/library/venv.html)

We use poetry tool to manage the dependencies of explora.
To install poetry in osx / linux one can run
```bash
curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
```
For more information on the installation follow the instructions [here](https://python-poetry.org/docs/#installation)


Install explora package along with the dependencies. 
In the `explora` directory, where `pyproject.toml` is (same one where you found this file after cloning the git repo), 
execute:
```bash
poetry install --no-dev
```
The `--no-dev` flag will exclude the dependencies used only for explora development, like tests.


## Development
[comment]: <> (#TODO add more details and verify examples)
### How to run the tests
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

### How to add a dependency
* Use poetry utility, to add a new dependency: `poetry add xxx`

### How to format code
* Use `black` utility, which is installed as part of `requirements-dev.txt`
* Reformat all python files under current folder and subfolders 
```bash
black .
```
* Check if format is acceptable 
```bash
black --check .
```
