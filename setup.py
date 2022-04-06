from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='genetic_selector',
    version='1.0',
    description='Feature-Selector Genetic Algorithm created to choose the best subset of features from a original dataset.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/BiDAlab/GeneticAlgorithm',
    author='BiDA Lab',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=['scikit-learn>=0.23', 'numpy>=1.18', 'pandas>=1.0.1'],
)
