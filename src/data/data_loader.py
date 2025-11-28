"""
Data loading and preprocessing module for CSA T20 Challenge dataset.
Handles CSV loading, data cleaning, and basic data access functions.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import os


class DataLoader:
    """Load and manage CSA T20 Challenge ball-by-ball data."""
    
    def __init__(self, csv_path: str = "all_matches.csv"):
        """
        Initialize DataLoader with CSV path.
        
        Args:
            csv_path: Path to the all_matches.csv file
        """
        self.csv_path = csv_path
        self.df = None
        self._teams = None
        self._venues = None
        self._players = None
        
    def load_dataset(self) -> pd.DataFrame:
        """
        Load and preprocess the CSV dataset.
        
        Returns:
            DataFrame with preprocessed match data
        """
        print(f"Loading dataset from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        
        print(f"Loaded {len(self.df)} ball records from {self.df['match_id'].nunique()} matches")
        
        # Create derived columns
        self.df['total_runs'] = self.df['runs_off_bat'].fillna(0) + self.df['extras'].fillna(0)
        
        # Convert date to datetime
        self.df['start_date'] = pd.to_datetime(self.df['start_date'])
        
        # Fill NaN values for wicket-related columns
        self.df['wicket_type'] = self.df['wicket_type'].fillna('none')
        self.df['player_dismissed'] = self.df['player_dismissed'].fillna('')
        
        # Create is_wicket flag
        self.df['is_wicket'] = self.df['wicket_type'] != 'none'
        
        # Create boundary flags
        self.df['is_four'] = self.df['runs_off_bat'] == 4
        self.df['is_six'] = self.df['runs_off_bat'] == 6
        self.df['is_boundary'] = self.df['is_four'] | self.df['is_six']
        
        # Create dot ball flag (no runs scored)
        self.df['is_dot'] = self.df['total_runs'] == 0
        
        print("Dataset preprocessing complete!")
        return self.df
    
    def get_teams(self) -> List[str]:
        """
        Get list of all unique teams in the dataset.
        
        Returns:
            Sorted list of team names
        """
        if self._teams is None:
            batting_teams = set(self.df['batting_team'].unique())
            bowling_teams = set(self.df['bowling_team'].unique())
            self._teams = sorted(batting_teams | bowling_teams)
        return self._teams
    
    def get_venues(self) -> List[str]:
        """
        Get list of all unique venues in the dataset.
        
        Returns:
            Sorted list of venue names
        """
        if self._venues is None:
            self._venues = sorted(self.df['venue'].unique())
        return self._venues
    
    def get_players(self, teams: Optional[List[str]] = None) -> Dict[str, set]:
        """
        Get all unique players, optionally filtered by teams.
        
        Args:
            teams: Optional list of team names to filter players
            
        Returns:
            Dictionary mapping each team to set of player names
        """
        team_players = {}
        
        teams_to_process = teams if teams else self.get_teams()
        
        for team in teams_to_process:
            players = set()
            
            # Get batsmen
            team_batting = self.df[self.df['batting_team'] == team]
            players.update(team_batting['striker'].unique())
            players.update(team_batting['non_striker'].unique())
            
            # Get bowlers
            team_bowling = self.df[self.df['bowling_team'] == team]
            players.update(team_bowling['bowler'].unique())
            
            # Remove empty strings
            players.discard('')
            
            team_players[team] = players
        
        return team_players
    
    def get_match_data(self, team1: str, team2: str, venue: Optional[str] = None) -> pd.DataFrame:
        """
        Get match data filtered by teams and optionally venue.
        
        Args:
            team1: First team name
            team2: Second team name
            venue: Optional venue name to filter
            
        Returns:
            Filtered DataFrame containing relevant matches
        """
        # Filter matches between the two teams
        team_filter = (
            ((self.df['batting_team'] == team1) & (self.df['bowling_team'] == team2)) |
            ((self.df['batting_team'] == team2) & (self.df['bowling_team'] == team1))
        )
        
        filtered_df = self.df[team_filter].copy()
        
        # Further filter by venue if specified
        if venue:
            filtered_df = filtered_df[filtered_df['venue'] == venue]
        
        return filtered_df
    
    def get_player_balls(self, player_name: str, role: str = 'both') -> pd.DataFrame:
        """
        Get all balls for a specific player.
        
        Args:
            player_name: Name of the player
            role: 'batting', 'bowling', or 'both'
            
        Returns:
            DataFrame with player's balls
        """
        if role == 'batting':
            return self.df[
                (self.df['striker'] == player_name) | 
                (self.df['non_striker'] == player_name)
            ].copy()
        elif role == 'bowling':
            return self.df[self.df['bowler'] == player_name].copy()
        else:  # both
            batting = (self.df['striker'] == player_name) | (self.df['non_striker'] == player_name)
            bowling = self.df['bowler'] == player_name
            return self.df[batting | bowling].copy()
    
    def get_team_matches(self, team: str) -> List[int]:
        """
        Get all match IDs where a team played.
        
        Args:
            team: Team name
            
        Returns:
            List of match IDs
        """
        team_matches = self.df[
            (self.df['batting_team'] == team) | 
            (self.df['bowling_team'] == team)
        ]['match_id'].unique()
        return sorted(team_matches)


# Convenience function for quick loading
def load_dataset(csv_path: str = "all_matches.csv") -> Tuple[pd.DataFrame, DataLoader]:
    """
    Quick function to load dataset.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Tuple of (DataFrame, DataLoader instance)
    """
    loader = DataLoader(csv_path)
    df = loader.load_dataset()
    return df, loader


if __name__ == "__main__":
    # Test the data loader
    df, loader = load_dataset()
    
    print(f"\nDataset shape: {df.shape}")
    print(f"\nTeams ({len(loader.get_teams())}): {', '.join(loader.get_teams())}")
    print(f"\nVenues ({len(loader.get_venues())}): {', '.join(loader.get_venues()[:5])}...")
    
    # Test getting players for specific teams
    team_players = loader.get_players(['Warriors', 'Titans'])
    for team, players in team_players.items():
        print(f"\n{team} has {len(players)} players")
