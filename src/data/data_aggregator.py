"""
Data aggregator to create player-level statistics from ball-by-ball data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def aggregate_match_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ball-by-ball data into player-match level performance.
    
    Args:
        df: Ball-by-ball DataFrame
        
    Returns:
        DataFrame with player-match level statistics
    """
    all_stats = []
    
    for match_id in df['match_id'].unique():
        match_df = df[df['match_id'] == match_id]
        venue = match_df['venue'].iloc[0]
        date = match_df['start_date'].iloc[0]
        
        # Get all players in match
        players = set()
        players.update(match_df['striker'].unique())
        players.update(match_df['non_striker'].unique())
        players.update(match_df['bowler'].unique())
        players.discard('')
        
        for player in players:
            stats = calculate_player_match_stats(player, match_id, match_df)
            stats['match_id'] = match_id
            stats['venue'] = venue
            stats['date'] = date
            all_stats.append(stats)
    
    return pd.DataFrame(all_stats)


def calculate_player_match_stats(player: str, match_id: int, match_df: pd.DataFrame) -> Dict:
    """
    Calculate all statistics for a player in a specific match.
    
    Args:
        player: Player name
        match_id: Match ID
        match_df: DataFrame for the specific match
        
    Returns:
        Dictionary with comprehensive player statistics
    """
    stats = {'player': player}
    
    # Batting stats
    batting_stats = calculate_batting_stats(player, match_df)
    stats.update(batting_stats)
    
    # Bowling stats
    bowling_stats = calculate_bowling_stats(player, match_df)
    stats.update(bowling_stats)
    
    # Determine role
    stats['role'] = determine_role(batting_stats, bowling_stats)
    
    return stats


def calculate_batting_stats(player: str, match_df: pd.DataFrame) -> Dict:
    """Calculate batting statistics for a player."""
    batting_balls = match_df[match_df['striker'] == player].copy()
    
    if len(batting_balls) == 0:
        return {
            'bat_runs': 0,
            'bat_balls': 0,
            'bat_fours': 0,
            'bat_sixes': 0,
            'bat_strike_rate': 0,
            'bat_dismissed': False
        }
    
    runs = batting_balls['runs_off_bat'].sum()
    balls = len(batting_balls)
    fours = len(batting_balls[batting_balls['runs_off_bat'] == 4])
    sixes = len(batting_balls[batting_balls['runs_off_bat'] == 6])
    strike_rate = (runs / balls * 100) if balls > 0 else 0
    dismissed = player in batting_balls['player_dismissed'].values
    
    return {
        'bat_runs': runs,
        'bat_balls': balls,
        'bat_fours': fours,
        'bat_sixes': sixes,
        'bat_strike_rate': strike_rate,
        'bat_dismissed': dismissed
    }


def calculate_bowling_stats(player: str, match_df: pd.DataFrame) -> Dict:
    """Calculate bowling statistics for a player."""
    bowling_balls = match_df[match_df['bowler'] == player].copy()
    
    if len(bowling_balls) == 0:
        return {
            'bowl_wickets': 0,
            'bowl_runs': 0,
            'bowl_balls': 0,
            'bowl_economy': 0,
            'bowl_maidens': 0,
            'bowl_dots': 0
        }
    
    wickets = len(bowling_balls[bowling_balls['is_wicket'] == True])
    runs = bowling_balls['total_runs'].sum()
    balls = len(bowling_balls)
    overs = balls / 6
    economy = (runs / overs) if overs > 0 else 0
    dots = len(bowling_balls[bowling_balls['is_dot'] == True])
    
    # Calculate maidens
    bowling_balls['over'] = bowling_balls['ball'].apply(lambda x: int(x))
    maidens = 0
    for over_num in bowling_balls['over'].unique():
        over_balls = bowling_balls[bowling_balls['over'] == over_num]
        if over_balls['total_runs'].sum() == 0 and len(over_balls) >= 6:
            maidens += 1
    
    return {
        'bowl_wickets': wickets,
        'bowl_runs': runs,
        'bowl_balls': balls,
        'bowl_economy': economy,
        'bowl_maidens': maidens,
        'bowl_dots': dots
    }


def determine_role(batting_stats: Dict, bowling_stats: Dict) -> str:
    """
    Determine player role based on their stats.
    
    Returns:
        'Batsman', 'Bowler', 'All-rounder', or 'Unknown'
    """
    has_batted = batting_stats['bat_balls'] > 0
    has_bowled = bowling_stats['bowl_balls'] > 0
    
    if has_batted and has_bowled:
        # Check if substantial contribution in both
        if batting_stats['bat_balls'] >= 6 and bowling_stats['bowl_balls'] >= 12:
            return 'All-rounder'
        elif batting_stats['bat_balls'] > bowling_stats['bowl_balls']:
            return 'Batsman'
        else:
            return 'Bowler'
    elif has_batted:
        return 'Batsman'
    elif has_bowled:
        return 'Bowler'
    else:
        return 'Unknown'


def get_player_summary(player: str, df: pd.DataFrame) -> Dict:
    """
    Get comprehensive career summary for a player.
    
    Args:
        player: Player name
        df: Complete ball-by-ball DataFrame
        
    Returns:
        Dictionary with career statistics
    """
    # Get all matches where player participated
    player_matches = df[
        (df['striker'] == player) | 
        (df['non_striker'] == player) | 
        (df['bowler'] == player)
    ]['match_id'].unique()
    
    # Aggregate stats across all matches
    total_batting_runs = 0
    total_batting_balls = 0
    total_wickets = 0
    total_bowling_runs = 0
    total_bowling_balls = 0
    matches_played = len(player_matches)
    
    for match_id in player_matches:
        match_df = df[df['match_id'] == match_id]
        batting_stats = calculate_batting_stats(player, match_df)
        bowling_stats = calculate_bowling_stats(player, match_df)
        
        total_batting_runs += batting_stats['bat_runs']
        total_batting_balls += batting_stats['bat_balls']
        total_wickets += bowling_stats['bowl_wickets']
        total_bowling_runs += bowling_stats['bowl_runs']
        total_bowling_balls += bowling_stats['bowl_balls']
    
    batting_avg = total_batting_runs / matches_played if matches_played > 0 else 0
    batting_sr = (total_batting_runs / total_batting_balls * 100) if total_batting_balls > 0 else 0
    bowling_avg = (total_bowling_runs / total_wickets) if total_wickets > 0 else 0
    bowling_economy = (total_bowling_runs / (total_bowling_balls / 6)) if total_bowling_balls > 0 else 0
    
    return {
        'player': player,
        'matches_played': matches_played,
        'batting_runs': total_batting_runs,
        'batting_average': batting_avg,
        'batting_strike_rate': batting_sr,
        'bowling_wickets': total_wickets,
        'bowling_average': bowling_avg,
        'bowling_economy': bowling_economy
    }


if __name__ == "__main__":
    # Test
    from data.data_loader import load_dataset
    
    df, loader = load_dataset()
    
    # Get summary for a player
    player = 'JN Malan'
    summary = get_player_summary(player, df)
    print(f"\nCareer Summary for {player}:")
    for key, value in summary.items():
        print(f"{key}: {value}")
