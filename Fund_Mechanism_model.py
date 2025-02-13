import pandas as pd
import numpy as np
import re
import requests
from datetime import datetime
from tqdm import tqdm

import lightgbm as lgb
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

# ------------------------------
# Global variables and caching
# ------------------------------
# Replace with your GitHub token if needed.
GITHUB_TOKEN = 'Token'
HEADERS = {'Authorization': f'token {GITHUB_TOKEN}'}
GITHUB_CACHE = {}  # In-memory cache for GitHub API responses

# Global variables for one-hot encoding and scaling.
FUNDER_COLUMNS = None
SCALER = None

# ------------------------------
# GitHub API functions with caching
# ------------------------------
def fetch_github_metrics(url):
    """
    Fetch GitHub repository metrics with basic error handling.
    Returns a dictionary with stars, forks, watchers, open issues, and days since last push.
    """
    try:
        repo_match = re.match(r'https://github\.com/([^/]+)/([^/]+)/?', url)
        if not repo_match:
            return {}
        owner, repo = repo_match.groups()
        api_url = f'https://api.github.com/repos/{owner}/{repo}'
        response = requests.get(api_url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'pushed_at' in data:
                last_update = (datetime.now() - datetime.strptime(
                    data['pushed_at'], '%Y-%m-%dT%H:%M:%SZ')).days
            else:
                last_update = 3650
            return {
                'stars': data.get('stargazers_count', 0),
                'forks': data.get('forks_count', 0),
                'watchers': data.get('subscribers_count', 0),
                'open_issues': data.get('open_issues_count', 0),
                'last_update': last_update,
            }
        else:
            return {}
    except Exception:
        return {}

def get_github_metrics(url):
    """
    Use an in-memory cache to avoid redundant API calls.
    """
    if url in GITHUB_CACHE:
        return GITHUB_CACHE[url]
    metrics = fetch_github_metrics(url)
    GITHUB_CACHE[url] = metrics
    return metrics

# ------------------------------
# Feature extraction functions
# ------------------------------
def extract_repo_features(url):
    """
    Extract repository features from the URL and GitHub API.
    Returns a dictionary of numeric features.
    """
    features = {}
    # Derive organization and repository names from URL.
    match = re.search(r'github\.com/([^/]+)/([^/]+)', url)
    org = match.group(1).lower() if match else 'unknown'
    repo = match.group(2).lower() if match else 'unknown'
    
    # Get GitHub API metrics with caching.
    gh_data = get_github_metrics(url)
    features['stars'] = int(gh_data.get('stars', 0))
    features['forks'] = int(gh_data.get('forks', 0))
    features['watchers'] = int(gh_data.get('watchers', 0))
    features['open_issues'] = int(gh_data.get('open_issues', 0))
    features['last_update'] = int(gh_data.get('last_update', 3650))
    features['org_length'] = len(org)
    features['repo_length'] = len(repo)
    
    # Technology flags based on URL keywords.
    tech_keywords = {
        'typescript': ['typescript', 'ts'],
        'blockchain': ['web3', 'ethereum', 'solidity'],
        'testing': ['jest', 'mocha', 'pytest']
    }
    for tech, terms in tech_keywords.items():
        features[f'tech_{tech}'] = int(any(term in url.lower() for term in terms))
    
    return features

def prepare_features_for_lgb(df, is_training=True):
    """
    Given a DataFrame (either training or test), this function:
      - Extracts features for project_a and project_b from GitHub.
      - Computes engineered features: differences, sums, and ratios between a and b.
      - Adds extra features: year, quarter, and one-hot encoded funder.
      - Fills missing values and scales the final feature set using RobustScaler.
    Returns the processed feature DataFrame.
    """
    global FUNDER_COLUMNS, SCALER

    # Extract repository features for project_a and project_b.
    a_features = pd.DataFrame([extract_repo_features(url) for url in tqdm(df['project_a'], desc="Processing project_a")])
    b_features = pd.DataFrame([extract_repo_features(url) for url in tqdm(df['project_b'], desc="Processing project_b")])
    
    # Engineered features:
    # 1. Difference between a and b.
    diff_features = a_features - b_features
    diff_features.columns = [f"diff_{col}" for col in diff_features.columns]
    # 2. Sum of a and b.
    sum_features = a_features + b_features
    sum_features.columns = [f"sum_{col}" for col in sum_features.columns]
    # 3. Ratio: a_features divided by (b_features + epsilon).
    eps = 1e-5
    ratio_features = a_features / (b_features.replace(0, eps))
    ratio_features.columns = [f"ratio_{col}" for col in ratio_features.columns]
    
    # Extra features: temporal features and one-hot encoding for funder.
    temporal_features = pd.DataFrame({
        'year': df['quarter'].str[:4].astype(int),
        'quarter': df['quarter'].str[-1].astype(int)
    })
    funder_dummies = pd.get_dummies(df['funder'], prefix='funder')
    if is_training:
        FUNDER_COLUMNS = funder_dummies.columns.tolist()
    else:
        if FUNDER_COLUMNS is not None:
            funder_dummies = funder_dummies.reindex(columns=FUNDER_COLUMNS, fill_value=0)
    
    extra_features = pd.concat([temporal_features, funder_dummies], axis=1)
    
    # Combine all engineered features.
    features = pd.concat([diff_features, sum_features, ratio_features, extra_features], axis=1)
    
    # Fill missing values.
    features.fillna(features.median(), inplace=True)
    
    # Scale features using RobustScaler.
    if is_training:
        SCALER = RobustScaler().fit(features)
        features_scaled = pd.DataFrame(SCALER.transform(features), columns=features.columns)
    else:
        if SCALER is None:
            raise ValueError("Scaler has not been fitted on training data!")
        features_scaled = pd.DataFrame(SCALER.transform(features), columns=features.columns)
    
    return features_scaled

def main():
    try:
        # Load training and test datasets.
        train_df = pd.read_csv('dataset.csv')
        test_df = pd.read_csv('test.csv')
        
        print("Preparing training features...")
        X_train = prepare_features_for_lgb(train_df, is_training=True)
        y_train = train_df['weight_a'].values  # Target funding share for project A
        
        print("Preparing test features...")
        X_test = prepare_features_for_lgb(test_df, is_training=False)
        
        # Set up 5-fold cross-validation.
        folds = 5
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)
        base_oof = np.zeros(X_train.shape[0])
        base_test = np.zeros(X_test.shape[0])
        
        # Train a LightGBM model as the base model.
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            print(f"\nStarting fold {fold+1}...")
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            train_data = lgb.Dataset(X_tr, label=y_tr)
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            params = {
                'objective': 'regression',
                'metric': 'mse',
                'boosting': 'gbdt',
                'learning_rate': 0.005,
                'num_leaves': 31,
                'max_depth': 7,
                'verbose': -1,
                'seed': 42,
            }
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=10000,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
            )
            
            base_oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
            base_test += model.predict(X_test, num_iteration=model.best_iteration) / folds
        
        # Clip base model predictions to [0, 1]
        base_oof = np.clip(base_oof, 0, 1)
        base_test = np.clip(base_test, 0, 1)
        
        # Transform base predictions with PolynomialFeatures (degree 2)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        meta_train = poly.fit_transform(base_oof.reshape(-1, 1))
        meta_test = poly.transform(base_test.reshape(-1, 1))
        
        # Train a Ridge regression meta-learner on the transformed meta-features.
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(meta_train, y_train)
        final_preds = meta_model.predict(meta_test)
        final_preds = np.clip(final_preds, 0, 1)
        
        # Save submission file.
        submission = pd.DataFrame({
            'id': test_df['id'],
            'pred': final_preds
        })
        submission.to_csv('submission.csv', index=False)
        print("\nSubmission file saved as submission.csv")
        
    except Exception as e:
        print(f"Runtime Error: {str(e)}")

if __name__ == "__main__":
    main()
