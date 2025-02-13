# Open Source Funding Allocation Predictor

## Overview
This repository contains a machine learning solution for predicting the relative funding share between pairs of open-source repositories based on historical funding data and GitHub metrics. Our approach involves extracting key features via the GitHub API, engineering relative features between repository pairs, and training a LightGBM model. The base predictions are further refined using a polynomial transformation and a Ridge regression meta‑learner.

## Model Architecture
1. **Feature Extraction:**  
   - Retrieve GitHub metrics (stars, forks, watchers, open issues, last update) using the GitHub API.
   - Extract additional features such as organization and repository name lengths and technology flags.
   - Compute engineered features (difference, sum, ratio) between repository pairs.
   - Include extra contextual features (year, quarter, funder).
2. **Base Model:**  
   - Train a LightGBM regressor with 5‑fold cross‑validation.
3. **Meta‑Learning:**  
   - Apply a degree‑2 polynomial expansion to base predictions.
   - Train a Ridge regression meta‑learner on the expanded features.
4. **Output:**  
   - Final predictions (clipped between 0 and 1) are saved to `submission.csv`.

## Installation
Install required packages using pip:
```bash
pip install pandas numpy scikit-learn tqdm requests lightgbm xgboost catboost tensorflow
