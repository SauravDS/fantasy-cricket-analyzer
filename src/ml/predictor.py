"""
ML Predictor module for generating fantasy point predictions.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.player_features import (extract_batting_features, extract_bowling_features,
                                       extract_form_features, extract_consistency_features)
from src.features.contextual_features import extract_ground_features, extract_opposition_features


class FantasyPredictor:
    """Predict fantasy points using trained ML model."""
    
    def __init__(self, model_path: str = 'models/fantasy_predictor.pkl'):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to saved model file
        """
        self.model = joblib.load(model_path)
        self.feature_names = joblib.load('models/feature_names.pkl')
        print(f"Loaded model from {model_path}")
    
    def predict_fantasy_points(self, player: str, team: str, opposition: str, 
                              venue: str, df: pd.DataFrame) -> Tuple[float, Dict]:
        """
        Predict fantasy points for a single player.
        
        Args:
            player: Player name
            team: Player's team
            opposition: Opposition team
            venue: Match venue
            df: Historical ball-by-ball data
            
        Returns:
            Tuple of (predicted_points, feature_dict)
        """
        # Extract features
        features = {}
        features.update(extract_batting_features(player, df))
        features.update(extract_bowling_features(player, df))
        features.update(extract_form_features(player, df))
        features.update(extract_consistency_features(player, df))
        features.update(extract_ground_features(player, venue, df))
        features.update(extract_opposition_features(player, opposition, df))
        
        # Create feature vector in correct order
        feature_vector = [features.get(fname, 0) for fname in self.feature_names]
        
        # Create DataFrame with feature names to avoid warning
        feature_df = pd.DataFrame([feature_vector], columns=self.feature_names)
        
        # Predict
        prediction = self.model.predict(feature_df)[0]
        
        # Ensure non-negative prediction
        prediction = max(0, prediction)
        
        return prediction, features
    
    def predict_all_players(self, players: List[str], team1: str, team2: str,
                           venue: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fantasy points for all players in a match.
        
        Args:
            players: List of player names (22 players)
            team1: First team name
            team2: Second team name
            venue: Match venue
            df: Historical ball-by-ball data
            
        Returns:
            DataFrame with predictions for all players
        """
        predictions = []
        
        # Determine which team each player belongs to
        team1_players_data = df[df['batting_team'] == team1]['striker'].unique()
        team2_players_data = df[df['batting_team'] == team2]['striker'].unique()
        
        for player in players:
            # Determine player's team
            if player in team1_players_data:
                player_team = team1
                opp_team = team2
            elif player in team2_players_data:
                player_team = team2
                opp_team = team1
            else:
                # Try bowling
                team1_bowlers = df[df['bowling_team'] == team1]['bowler'].unique()
                if player in team1_bowlers:
                    player_team = team1
                    opp_team = team2
                else:
                    player_team = team2
                    opp_team = team1
            
            # Predict
            pred_points, features = self.predict_fantasy_points(
                player, player_team, opp_team, venue, df
            )
            
            predictions.append({
                'player': player,
                'team': player_team,
                'predicted_points': pred_points,
                'bat_avg_runs': features.get('bat_avg_runs', 0),
                'bowl_avg_wickets': features.get('bowl_avg_wickets', 0),
                'recent_form_score': features.get('recent_form_score', 0),
                'venue_performance': features.get('venue_performance', 0),
                'opp_performance': features.get('opp_performance', 0),
            })
        
        pred_df = pd.DataFrame(predictions)
        pred_df = pred_df.sort_values('predicted_points', ascending=False).reset_index(drop=True)
        
        return pred_df
    
    def rank_players(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank players by predicted points.
        
        Args:
            predictions_df: DataFrame with predictions
            
        Returns:
            Ranked DataFrame with rank column
        """
        ranked_df = predictions_df.copy()
        ranked_df['rank'] = range(1, len(ranked_df) + 1)
        return ranked_df
    
    def get_prediction_confidence(self, player: str, features: Dict) -> str:
        """
        Get confidence level for prediction based on data availability.
        
        Args:
            player: Player name
            features: Feature dictionary
            
        Returns:
            Confidence level string
        """
        matches_played = features.get('bat_matches', 0) + features.get('bowl_matches', 0)
        venue_matches = features.get('venue_matches', 0)
        opp_matches = features.get('opp_matches', 0)
        
        if matches_played >= 10 and venue_matches >= 2 and opp_matches >= 2:
            return "High"
        elif matches_played >= 5 and (venue_matches >= 1 or opp_matches >= 1):
            return "Medium"
        else:
            return "Low"


def load_predictor(model_path: str = 'models/fantasy_predictor.pkl') -> FantasyPredictor:
    """
    Convenience function to load predictor.
    
    Args:
        model_path: Path to model file
        
    Returns:
        FantasyPredictor instance
    """
    return FantasyPredictor(model_path)


if __name__ == "__main__":
    # Test the predictor
    from src.data.data_loader import load_dataset
    
    df, loader = load_dataset()
    predictor = load_predictor()
    
    # Test prediction for a player
    player = 'JN Malan'
    pred, features = predictor.predict_fantasy_points(
        player, 'Boland', 'Western Province', 'Boland Park, Paarl', df
    )
    
    print(f"\nPredicted fantasy points for {player}: {pred:.2f}")
    print(f"Confidence: {predictor.get_prediction_confidence(player, features)}")
