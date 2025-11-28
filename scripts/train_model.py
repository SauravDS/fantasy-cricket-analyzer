"""
Machine learning model trainer for fantasy points prediction.
Train and evaluate multiple regression models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import load_dataset
from src.fantasy.points_calculator import FantasyPointsCalculator


def prepare_training_data(csv_path: str = "all_matches.csv"):
    """
    Prepare training dataset with features and target variable.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names, training_df)
    """
    print("Loading dataset...")
    df, loader = load_dataset(csv_path)
    
    print("Calculating fantasy points...")
    calculator = FantasyPointsCalculator(df)
    training_df = calculator.create_training_dataset()
    
    # Now we need to add features to this training data
    # OPTIMIZATION: Limit to recent matches for faster training
    unique_dates = sorted(training_df['date'].unique())
    if len(unique_dates) > 50:
        cutoff_date = unique_dates[-50]
        print(f"Limiting training to last 50 matches (since {cutoff_date}) for performance...")
        training_df = training_df[training_df['date'] >= cutoff_date]
        
    print(f"Engineering features for {len(training_df)} player-match records...")
    
    from src.features.player_features import (extract_batting_features, extract_bowling_features,
                                          extract_form_features, extract_consistency_features)
    from src.features.contextual_features import extract_ground_features, extract_opposition_features
    
    # For each player-match record, extract features based on PRIOR data
    # This is important to avoid data leakage
    feature_rows = []
    
    for idx, row in training_df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"Processing {idx + 1}/{len(training_df)} records...")
        
        player = row['player']
        match_date = row['date']
        venue = row['venue']
        opposition = row['opposition']
        
        # Get data BEFORE this match (to avoid leakage)
        prior_df = df[df['start_date'] < match_date]
        
        if len(prior_df) == 0:
            # Skip if no prior data
            continue
        
        # Extract features from prior data
        features = {
            'player': player,
            'match_id': row['match_id'],
            'fantasy_points': row['fantasy_points']  # Target variable
        }
        
        features.update(extract_batting_features(player, prior_df))
        features.update(extract_bowling_features(player, prior_df))
        features.update(extract_form_features(player, prior_df))
        features.update(extract_consistency_features(player, prior_df))
        features.update(extract_ground_features(player, venue, prior_df))
        features.update(extract_opposition_features(player, opposition, prior_df))
        
        feature_rows.append(features)
    
    feature_df = pd.DataFrame(feature_rows)
    print(f"\nCreated feature matrix with {len(feature_df)} samples")
    
    # Prepare X and y
    feature_columns = [col for col in feature_df.columns 
                      if col not in ['player', 'match_id', 'fantasy_points']]
    
    X = feature_df[feature_columns].fillna(0)
    y = feature_df['fantasy_points']
    
    # Train-test split (temporal split - last 20% as test)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, feature_columns, feature_df


def train_random_forest(X_train, y_train):
    """Train Random Forest model."""
    print("\nTraining Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    """Train XGBoost model."""
    print("\nTraining XGBoost...")
    model = XGBRegressor(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting model."""
    print("\nTraining Gradient Boosting...")
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R²:   {r2:.3f}")
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'predictions': y_pred}


def select_best_model(models, X_test, y_test):
    """Select best model based on R² score."""
    best_model = None
    best_r2 = -float('inf')
    best_name = None
    
    for name, model in models.items():
        results = evaluate_model(model, X_test, y_test, name)
        if results['r2'] > best_r2:
            best_r2 = results['r2']
            best_model = model
            best_name = name
    
    print(f"\n{'='*50}")
    print(f"Best Model: {best_name} (R² = {best_r2:.3f})")
    print(f"{'='*50}")
    
    return best_model, best_name


def save_model(model, filepath):
    """Save model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"\nModel saved to: {filepath}")


def main():
    """Main training pipeline."""
    print("="*60)
    print("FANTASY CRICKET ML MODEL TRAINING")
    print("="*60)
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_names, feature_df = prepare_training_data()
    
    # Train multiple models
    models = {
        'Random Forest': train_random_forest(X_train, y_train),
        'XGBoost': train_xgboost(X_train, y_train),
        'Gradient Boosting': train_gradient_boosting(X_train, y_train)
    }
    
    # Evaluate and select best model
    best_model, best_name = select_best_model(models, X_test, y_test)
    
    # Save best model and feature names
    save_model(best_model, 'models/fantasy_predictor.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    # Save feature importance if available
    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))
        
        importance_df.to_csv('models/feature_importance.csv', index=False)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
