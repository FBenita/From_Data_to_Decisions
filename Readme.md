This repository contains the dataset and the code necessary to reproduce the results in the paper:

**Benita, F. (2026).** *From Data to Decisions: A Need-Weighted Optimization Framework for Evidence-Based Urban Planning*. 

## Project Structure
- `notebooks/`: Jupyter Notebooks containing the robustness tests, knapsack optimization, and multi-objective Pareto analysis.
- `data/`: Intermediate data (Bayesian coefficients and standard errors) required to run the optimization.

## Data Privacy & Reproducibility
Due to Institutional Review Board restrictions and participant privacy agreements, the raw, respondent-level survey data (`all_responses_coded.csv`) is not publicly available.

**To ensure full reproducibility**, the optimization scripts include hardcoded baseline population statistics (means) from the primary study. This allows researchers to reproduce all optimal portfolios, sensitivity analyses, and Pareto fronts presented in the paper without requiring access to the restricted primary dataset.

## How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
