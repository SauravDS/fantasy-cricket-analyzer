"""
Contextual feature extraction for ground and opposition-specific performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from ..data.data_aggregator import calculate_batting_stats, calculate_bowling_stats


def extract_ground_features(player: str, venue: str, df: pd.DataFrame) -> Dict:
    """
    Extract player performance features at a specific ground.
    
    Args:
        player: Player name
        venue: Venue name
        df: Ball-by-ball DataFrame
        
    Returns:
        Dictionary of ground-specific features
    """
    # Filter matches at this venue
    venue_df = df[df['venue'] == venue]
    
    # Get player's matches at this venue
    player_venue_matches = venue_df[
        (venue_df['striker'] == player) | 
        (venue_df['bowler'] == player)
    ]['match_id'].unique()
    
    if len(player_venue_matches) == 0:
        return {
            'venue_matches': 0,
            'venue_bat_avg': 0,
            'venue_bowl_avg': 0,
            'venue_performance': 0,
        }
    
    # Calculate stats at this venue
    total_runs = 0
    total_wickets = 0
    
    for match_id in player_venue_matches:
        match_df = venue_df[venue_df['match_id'] == match_id]
        bat_stats = calculate_batting_stats(player, match_df)
        bowl_stats = calculate_bowling_stats(player, match_df)
        
        total_runs += bat_stats['bat_runs']
        total_wickets += bowl_stats['bowl_wickets']
    
    return {
        'venue_matches': len(player_venue_matches),
        'venue_bat_avg': total_runs / len(player_venue_matches),
        'venue_bowl_avg': total_wickets / len(player_venue_matches),
        'venue_performance': (total_runs + total_wickets * 20) / len(player_venue_matches),
    }


def extract_opposition_features(player: str, opposition_team: str, df: pd.DataFrame) -> Dict:
    """
    Extract player performance features against a specific opposition.
    
    Args:
        player: Player name
        opposition_team: Opposition team name
        df: Ball-by-ball DataFrame
        
    Returns:
        Dictionary of opposition-specific features
    """
    # Find matches where player faced this opposition
    # Player's team is batting and opposition is bowling, or vice versa
    opposition_matches = df[
        ((df['striker'] == player) & (df['bowling_team'] == opposition_team)) |
        ((df['bowler'] == player) & (df['batting_team'] == opposition_team))
    ]['match_id'].unique()
    
    if len(opposition_matches) == 0:
        return {
            'opp_matches': 0,
            'opp_bat_avg': 0,
            'opp_bowl_avg': 0,
            'opp_performance': 0,
        }
    
    # Calculate stats against this opposition
    total_runs = 0
    total_wickets = 0
    
    for match_id in opposition_matches:
        match_df = df[df['match_id'] == match_id]
        bat_stats = calculate_batting_stats(player, match_df)
        bowl_stats = calculate_bowling_stats(player, match_df)
        
        total_runs += bat_stats['bat_runs']
        total_wickets += bowl_stats['bowl_wickets']
    
    return {
        'opp_matches': len(opposition_matches),
        'opp_bat_avg': total_runs / len(opposition_matches),
        'opp_bowl_avg': total_wickets / len(opposition_matches),
        'opp_performance': (total_runs + total_wickets * 20) / len(opposition_matches),
    }


def extract_matchup_features(player: str, opposition_players: List[str], df: pd.DataFrame) -> Dict:
    """
    Extract head-to-head matchup features.
    
    Args:
        player: Player name
        opposition_players: List of opposition player names
        df: Ball-by-ball DataFrame
        
    Returns:
        Dictionary of matchup features
    """
    # This is a simplified version
    # In a real scenario, you'd want bowler vs batsman specific stats
    
    matchup_score = 0
    encounters = 0
    
    # If player is a batsman, look at performance against opposition bowlers
    for opp_player in opposition_players:
        # Balls where player faced opp_player
        matchup_balls = df[
            (df['striker'] == player) & 
            (df['bowler'] == opp_player)
        ]
        
        if len(matchup_balls) > 0:
            runs = matchup_balls['runs_off_bat'].sum()
            dismissed = player in matchup_balls['player_dismissed'].values
            encounters += 1
            matchup_score += runs - (50 if dismissed else 0)
    
    # If player is a bowler, look at performance against opposition batsmen
    for opp_player in opposition_players:
        matchup_balls = df[
            (df['bowler'] == player) & 
            (df['striker'] == opp_player)
        ]
        
        if len(matchup_balls) > 0:
            wickets = len(matchup_balls[matchup_balls['player_dismissed'] == opp_player])
            runs_conceded = matchup_balls['total_runs'].sum()
            encounters += 1
            matchup_score += (wickets * 25) - runs_conceded
    
    return {
        'matchup_encounters': encounters,
        'matchup_score': matchup_score / encounters if encounters > 0 else 0,
    }


def create_feature_matrix(players: List[str], team1: str, team2: str, 
                         venue: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Create complete feature matrix for all players.
    
    Args:
        players: List of player names
        team1: First team name
        team2: Second team name
        venue: Venue name
        df: Ball-by-ball DataFrame
        
    Returns:
        DataFrame with features for all players
    """
    from .player_features import (extract_batting_features, extract_bowling_features,
                                   extract_form_features, extract_consistency_features)
    
    feature_list = []
    
    for player in players:
        features = {'player': player}
        
        # Determine player's team and opposition
        team1_players = df[df['batting_team'] == team1]['striker'].unique()
        player_team = team1 if player in team1_players else team2
        opposition = team2 if player_team == team1 else team1
        
        features['team'] = player_team
        features['opposition'] = opposition
        features['venue'] = venue
        
        # Player features
        features.update(extract_batting_features(player, df))
        features.update(extract_bowling_features(player, df))
        features.update(extract_form_features(player, df))
        features.update(extract_consistency_features(player, df))
        
        # Contextual features
        features.update(extract_ground_features(player, venue, df))
        features.update(extract_opposition_features(player, opposition, df))
        
        feature_list.append(features)
    
    return pd.DataFrame(feature_list)
