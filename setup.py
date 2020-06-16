from setuptools import setup

with open('requirements.txt') as req_file:
    required = req_file.read().splitlines()

setup(
    name='markov_blankets',
    version='0.1',
    packages=['markov_blankets'],
    url='https://github.com/pmandros/markov_blankets',
    install_requires=required
)
