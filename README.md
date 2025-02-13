# Open-Source-Funding-Allocation-Predictor

## Overview

This repository contains a machine learning solution for predicting the relative funding share between pairs of open-source projects. The problem is formulated as a supervised regression task where, given historical funding data and associated repository metrics (extracted via the GitHub API), the model predicts the expected funding share for a given repository pair.

## Model Description

The solution utilizes an ensemble approach based on tree-based models with a meta-learning stage:

1. **Feature Engineering:**

   - GitHub metrics such as stars, forks, watchers, open issues, and last update (in days) are extracted for each repository.
   - Additional features include organization/repository name lengths and binary technology flags (for TypeScript, blockchain, and testing).
   - Engineered features (differences, sums, and ratios) are computed between the two repositories in a pair.
   - Extra contextual features (year, quarter, and funding source as one-hot encoded variables) are appended.

2. **Base Model (LightGBM):**

   - A LightGBM regressor is trained using 5-fold cross-validation.
   - The model is tuned to minimize mean squared error (MSE) and generate robust predictions.

3. **Meta-Learning:**
   - Base model predictions are transformed using a degree‑2 polynomial expansion.
   - A Ridge regression meta‑learner is trained on these features to refine the predictions.
   - Final predictions are clipped between 0 and 1.

## Installation

Ensure that you have Python 3.7 or higher installed. Then, install the required packages:

```bash
pip install pandas numpy scikit-learn tqdm requests lightgbm xgboost catboost tensorflow
```
