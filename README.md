# Histogram-Filter

Implementation of a simple Histogram Filter for robot localization in a one-dimensional discrete grid world.

## Background

The task is seen as a typical filtering problem of a HMM where:

- hidden states: position of the agent on the 1D grid world
- evidence: sensor measurements
  and the goal is to predict the categorical probability distribution over the states at a point t given all past evidence
  from 1:t and an initial prior distribution.

## Reference
See Artificial Intelligence: A Modern Approach - 3rd Edition (Russel, Norvig) Chapter 15.2.1 Filtering and prediction.

## Installation
The file `requirements.txt` contains all the libraries needed.
In addition, the library `termplotlib` used for plotting graphs in the console requires `gnuplot`. This link refers to the installation on https://wiki.ubuntuusers.de/Gnuplot/.

## Development
We use black for formatting, mypy for static typing and flake for linting. The development dependencies are listed under the `requirements.txt` file.
In order to enforce these, we are using https://pre-commit.com/ for which one installs the git-hooks with `pre-commit install`.
