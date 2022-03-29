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
