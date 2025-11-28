"""
Feature engineering for player performance prediction.
Extracts features from historical data for ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from ..data.data_aggregator import calculate_batting_stats, calculate_bowling_stats


def extract_batting_features(player: str, df: pd.DataFrame, n_recent: int = 5) -> Dict:
    """
    Extract batting features for a player.
    
    Args:
        player: Player name
        df: Ball-by-ball DataFrame
        n_recent: Number of recent matches for form calculation
        
    Returns:
        Dictionary of batting features
    """
    # Get all matches where player batted
    player_batting_matches = df[df['striker'] == player]['match_id'].unique()
    
    if len(player_batting_matches) == 0:
        return {
            'bat_avg_runs': 0,
            'bat_avg_sr': 0,
            'bat_matches': 0,
            'bat_recent_form': 0,
            'bat_consistency': 0,
        }
    
    # Calculate per-match stats
    match_runs = []
    match_sr = []
    
    for match_id in player_batting_matches:
        match_df = df[df['match_id'] == match_id]
        stats = calculate_batting_stats(player, match_df)
        match_runs.append(stats['bat_runs'])
        match_sr.append(stats['bat_strike_rate'])
    
    # Overall averages
    avg_runs = np.mean(match_runs)
    avg_sr = np.mean(match_sr)
    
    # Recent form (last n matches)
    recent_runs = match_runs[-n_recent:] if len(match_runs) >= n_recent else match_runs
    recent_form = np.mean(recent_runs) if recent_runs else 0
    
    # Consistency (coefficient of variation)
    consistency = np.std(match_runs) / avg_runs if avg_runs > 0 else 0
    
    return {
        'bat_avg_runs': avg_runs,
        'bat_avg_sr': avg_sr,
        'bat_matches': len(player_batting_matches),
        'bat_recent_form': recent_form,
        'bat_consistency': consistency,
        'bat_max_score': max(match_runs) if match_runs else 0,
    }


def extract_bowling_features(player: str, df: pd.DataFrame, n_recent: int = 5) -> Dict:
    """
    Extract bowling features for a player.
    
    Args:
        player: Player name
        df: Ball-by-ball DataFrame
        n_recent: Number of recent matches for form calculation
        
    Returns:
        Dictionary of bowling features
    """
    # Get all matches where player bowled
    player_bowling_matches = df[df['bowler'] == player]['match_id'].unique()
    
    if len(player_bowling_matches) == 0:
        return {
            'bowl_avg_wickets': 0,
            'bowl_avg_economy': 0,
            'bowl_matches': 0,
            'bowl_recent_form': 0,
            'bowl_consistency': 0,
        }
    
    # Calculate per-match stats
    match_wickets = []
    match_economy = []
    
    for match_id in player_bowling_matches:
        match_df = df[df['match_id'] == match_id]
        stats = calculate_bowling_stats(player, match_df)
        match_wickets.append(stats['bowl_wickets'])
        match_economy.append(stats['bowl_economy'])
    
    # Overall averages
    avg_wickets = np.mean(match_wickets)
    avg_economy = np.mean(match_economy)
    
    # Recent form (last n matches wickets)
    recent_wickets = match_wickets[-n_recent:] if len(match_wickets) >= n_recent else match_wickets
    recent_form = np.mean(recent_wickets) if recent_wickets else 0
    
    # Consistency
    consistency = np.std(match_wickets) / avg_wickets if avg_wickets > 0 else 0
    
    return {
        'bowl_avg_wickets': avg_wickets,
        'bowl_avg_economy': avg_economy,
        'bowl_matches': len(player_bowling_matches),
        'bowl_recent_form': recent_form,
        'bowl_consistency': consistency,
        'bowl_max_wickets': max(match_wickets) if match_wickets else 0,
    }


def extract_form_features(player: str, df: pd.DataFrame, n_matches: int = 5) -> Dict:
    """
    Extract recent form features with weighted recency.
    
    Args:
        player: Player name
        df: Ball-by-ball DataFrame
        n_matches: Number of recent matches to consider
        
    Returns:
        Dictionary of form features
    """
    # Get player's matches sorted by date
    player_matches = df[
        (df['striker'] == player) | 
        (df['bowler'] == player)
    ].sort_values('start_date')['match_id'].unique()
    
    if len(player_matches) == 0:
        return {'recent_form_score': 0, 'matches_in_window': 0}
    
    # Get last n matches
    recent_matches = player_matches[-n_matches:]
    
    # Calculate weighted performance
    # More recent matches get higher weight
    weights = np.linspace(0.5, 1.0, len(recent_matches))
    weighted_score = 0
    
    for i, match_id in enumerate(recent_matches):
        match_df = df[df['match_id'] == match_id]
        
        bat_stats = calculate_batting_stats(player, match_df)
        bowl_stats = calculate_bowling_stats(player, match_df)
        
        # Simple performance score
        perf_score = bat_stats['bat_runs'] + (bowl_stats['bowl_wickets'] * 20)
        weighted_score += perf_score * weights[i]
    
    avg_weighted_score = weighted_score / len(recent_matches)
    
    return {
        'recent_form_score': avg_weighted_score,
        'matches_in_window': len(recent_matches)
    }


def extract_consistency_features(player: str, df: pd.DataFrame) -> Dict:
    """
    Extract consistency and variance features.
    
    Args:
        player: Player name
        df: Ball-by-ball DataFrame
        
    Returns:
        Dictionary of consistency metrics
    """
    player_matches = df[
        (df['striker'] == player) | 
        (df['bowler'] == player)
    ]['match_id'].unique()
    
    if len(player_matches) < 3:
        return {
            'performance_std': 0,
            'performance_cv': 0,
        }
    
    # Calculate variance in performance
    match_scores = []
    
    for match_id in player_matches:
        match_df = df[df['match_id'] == match_id]
        bat_stats = calculate_batting_stats(player, match_df)
        bowl_stats = calculate_bowling_stats(player, match_df)
        
        score = bat_stats['bat_runs'] + (bowl_stats['bowl_wickets'] * 20)
        match_scores.append(score)
    
    std_dev = np.std(match_scores)
    mean_score = np.mean(match_scores)
    cv = std_dev / mean_score if mean_score > 0 else 0
    
    return {
        'performance_std': std_dev,
        'performance_cv': cv,
    }
