"""
In-app ML model training module with Streamlit integration.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os
from datetime import datetime

from src.data.data_loader import DataLoader
from src.fantasy.points_calculator import FantasyPointsCalculator
from src.features.player_features import (extract_batting_features, extract_bowling_features,
                                         extract_form_features, extract_consistency_features)
from src.features.contextual_features import extract_ground_features, extract_opposition_features


def train_model_from_dataframe(df: pd.DataFrame, progress_callback=None, league_name="Cricket", max_matches=None):
    """
    Train fantasy cricket model from uploaded DataFrame.
    
    Args:
        df: Ball-by-ball DataFrame
        progress_callback: Function to call with progress updates (message, percent)
        league_name: Name of the league for display
        max_matches: Maximum number of recent matches to use (None = use all matches)
        
    Returns:
        Tuple of (model, feature_names, model_info_dict)
    """
    
    def update_progress(message, percent=None):
        if progress_callback:
            progress_callback(message, percent)
    
    try:
        # Validate DataFrame
        update_progress("Validating dataset...", 5)
        required_cols = ['match_id', 'batting_team', 'bowling_team', 'striker', 
                        'bowler', 'total_runs', 'venue']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Calculate fantasy points
        update_progress("Calculating fantasy points...", 10)
        calculator = FantasyPointsCalculator(df)
        match_player_points = calculator.create_training_dataset()
        
        if len(match_player_points) == 0:
            raise ValueError("No player-match records found in dataset")
        
        # Optionally limit to recent matches
        if max_matches:
            update_progress(f"Preparing training data (Last {max_matches} matches)...", 15)
            unique_matches = sorted(df['match_id'].unique())
            if len(unique_matches) > max_matches:
                recent_matches = unique_matches[-max_matches:]
                match_player_points = match_player_points[
                    match_player_points['match_id'].isin(recent_matches)
                ]
        else:
            update_progress(f"Preparing training data (All {df['match_id'].nunique()} matches)...", 15)
        
        total_records = len(match_player_points)
        update_progress(f"Engineering features for {total_records} records...", 20)
        
        # Feature engineering
        features_list = []
        for idx, row in enumerate(match_player_points.iterrows()):
            i, player_match = row
            
            if idx % 100 == 0:
                percent = 20 + int((idx / total_records) * 30)
                update_progress(f"Processing {idx}/{total_records} records...", percent)
            
            player = player_match['player']
            match_id = player_match['match_id']
            
            # Get historical data (excluding current match)
            hist_df = df[df['match_id'] < match_id]
            
            if len(hist_df) == 0:
                continue
            
            # Extract features
            features = {}
            features.update(extract_batting_features(player, hist_df))
            features.update(extract_bowling_features(player, hist_df))
            features.update(extract_form_features(player, hist_df))
            features.update(extract_consistency_features(player, hist_df))
            
            # Get venue and opposition
            match_df = df[df['match_id'] == match_id]
            venue = match_df['venue'].iloc[0] if len(match_df) > 0 else ''
            
            # Determine player's team and opposition
            batting_teams = match_df['batting_team'].unique()
            player_team = None
            for team in batting_teams:
                team_players = match_df[match_df['batting_team'] == team]['striker'].unique()
                if player in team_players:
                    player_team = team
                    break
            
            if player_team:
                opposition = [t for t in batting_teams if t != player_team]
                opposition = opposition[0] if opposition else ''
            else:
                opposition = ''
            
            features.update(extract_ground_features(player, venue, hist_df))
            features.update(extract_opposition_features(player, opposition, hist_df))
            
            features['fantasy_points'] = player_match['fantasy_points']
            features_list.append(features)
        
        if len(features_list) == 0:
            raise ValueError("No features could be extracted from dataset")
        
        # Create feature matrix
        update_progress("Building feature matrix...", 55)
        features_df = pd.DataFrame(features_list)
        features_df = features_df.fillna(0)
        
        # Separate features and target
        y = features_df['fantasy_points']
        X = features_df.drop('fantasy_points', axis=1)
        feature_names = X.columns.tolist()
        
        # Train/test split
        update_progress("Splitting data for training...", 60)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train models
        models = {}
        
        update_progress("Training Random Forest...", 65)
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        models['Random Forest'] = rf
        
        update_progress("Training XGBoost...", 75)
        xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        models['XGBoost'] = xgb_model
        
        update_progress("Training Gradient Boosting...", 85)
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb.fit(X_train, y_train)
        models['Gradient Boosting'] = gb
        
        # Evaluate models
        update_progress("Evaluating models...", 90)
        best_model_name = None
        best_r2 = -float('inf')
        model_scores = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            model_scores[name] = {'r2': r2, 'mae': mae, 'rmse': rmse}
            
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
        
        best_model = models[best_model_name]
        
        # Save model
        update_progress("Saving model...", 95)
        os.makedirs('models', exist_ok=True)
        joblib.dump(best_model, 'models/fantasy_predictor.pkl')
        joblib.dump(feature_names, 'models/feature_names.pkl')
        
        # Prepare model info
        model_info = {
            'league_name': league_name,
            'best_model': best_model_name,
            'model_scores': model_scores,
            'n_samples': len(X),
            'n_matches': df['match_id'].nunique(),
            'n_teams': df['batting_team'].nunique(),
            'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        update_progress("Training complete!", 100)
        
        return best_model, feature_names, model_info
        
    except Exception as e:
        raise Exception(f"Training failed: {str(e)}")
