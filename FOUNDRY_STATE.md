# Foundry State

## Objective
Build an open decision-analysis toolkit for choices under uncertainty using probabilities, expected value, downside, reversibility, sensitivity, and value-of-information.

## Public boundary
Generic decision mathematics only. Do not add proprietary venture-selection heuristics, private financial data, or private cross-project ranking logic.

## V0 milestone
- Decision tree schema with chance and choice nodes
- Expected-value evaluation
- Downside / worst-case summaries
- Sensitivity analysis over uncertain probabilities and payoffs
- Value-of-information calculation for one additional observation
- Deterministic tests and worked examples

## Acceptance
A stranger can encode a real decision in a small JSON/Python example and receive the preferred option plus the assumptions most capable of flipping that choice.

## Next move
Implement the tree data model, EV solver, and one sensitivity example before any UI.

Status: ACTIVE / NOT YET PROVEN
