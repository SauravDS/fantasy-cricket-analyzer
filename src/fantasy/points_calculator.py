"""
Fantasy points calculator implementing Dream11 scoring system.
Calculates historical fantasy points from match data for ML training.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class FantasyPointsCalculator:
    """Calculate Dream11 fantasy points from ball-by-ball data."""
    
    # Dream11 Point System
    POINTS = {
        # Batting
        'run': 1,
        'boundary_4': 1,  # Bonus
        'boundary_6': 2,  # Bonus
        'milestone_30': 4,
        'milestone_50': 8,
        'milestone_100': 16,
        'duck': -2,  # For batsmen (0 runs dismissed)
        
        # Bowling
        'wicket': 25,
        'bonus_3w': 4,
        'bonus_4w': 8,
        'bonus_5w': 16,
        'maiden': 12,
        
        # Fielding
        'catch': 8,
        'stumping': 12,
        'run_out_direct': 12,
        'run_out_indirect': 6,
        
        # Economy Rate (for bowlers, min 2 overs)
        'economy_below_5': 6,
        'economy_5_to_5.99': 4,
        'economy_6_to_7': 2,
        'economy_10_to_11': -2,
        'economy_11_to_12': -4,
        'economy_above_12': -6,
        
        # Strike Rate (for batsmen, min 10 balls)
        'sr_above_170': 6,
        'sr_150_to_170': 4,
        'sr_130_to_150': 2,
        'sr_60_to_70': -2,
        'sr_50_to_60': -4,
        'sr_below_50': -6,
    }
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize calculator with match data.
        
        Args:
            df: Ball-by-ball match data
        """
        self.df = df
    
    def calculate_batting_points(self, player: str, match_id: int) -> Tuple[float, Dict]:
        """
        Calculate batting points for a player in a specific match.
        
        Args:
            player: Player name
            match_id: Match ID
            
        Returns:
            Tuple of (total_points, breakdown_dict)
        """
        # Get all balls where player was striker
        player_balls = self.df[
            (self.df['match_id'] == match_id) & 
            (self.df['striker'] == player)
        ].copy()
        
        if len(player_balls) == 0:
            return 0.0, {}
        
        breakdown = {}
        total_points = 0.0
        
        # Calculate runs
        runs = player_balls['runs_off_bat'].sum()
        breakdown['runs'] = runs
        total_points += runs * self.POINTS['run']
        
        # Boundary bonuses
        fours = len(player_balls[player_balls['runs_off_bat'] == 4])
        sixes = len(player_balls[player_balls['runs_off_bat'] == 6])
        breakdown['fours'] = fours
        breakdown['sixes'] = sixes
        total_points += fours * self.POINTS['boundary_4']
        total_points += sixes * self.POINTS['boundary_6']
        
        # Milestone bonuses
        if runs >= 100:
            breakdown['century'] = True
            total_points += self.POINTS['milestone_100']
        elif runs >= 50:
            breakdown['half_century'] = True
            total_points += self.POINTS['milestone_50']
        elif runs >= 30:
            breakdown['milestone_30'] = True
            total_points += self.POINTS['milestone_30']
        
        # Duck penalty (dismissed for 0)
        was_dismissed = len(player_balls[player_balls['player_dismissed'] == player]) > 0
        if was_dismissed and runs == 0:
            breakdown['duck'] = True
            total_points += self.POINTS['duck']
        
        # Strike rate bonus/penalty (min 10 balls)
        balls_faced = len(player_balls)
        breakdown['balls_faced'] = balls_faced
        
        if balls_faced >= 10:
            strike_rate = (runs / balls_faced) * 100
            breakdown['strike_rate'] = strike_rate
            
            if strike_rate > 170:
                total_points += self.POINTS['sr_above_170']
                breakdown['sr_bonus'] = self.POINTS['sr_above_170']
            elif strike_rate >= 150:
                total_points += self.POINTS['sr_150_to_170']
                breakdown['sr_bonus'] = self.POINTS['sr_150_to_170']
            elif strike_rate >= 130:
                total_points += self.POINTS['sr_130_to_150']
                breakdown['sr_bonus'] = self.POINTS['sr_130_to_150']
            elif strike_rate <= 50:
                total_points += self.POINTS['sr_below_50']
                breakdown['sr_penalty'] = self.POINTS['sr_below_50']
            elif strike_rate <= 60:
                total_points += self.POINTS['sr_50_to_60']
                breakdown['sr_penalty'] = self.POINTS['sr_50_to_60']
            elif strike_rate <= 70:
                total_points += self.POINTS['sr_60_to_70']
                breakdown['sr_penalty'] = self.POINTS['sr_60_to_70']
        
        return total_points, breakdown
    
    def calculate_bowling_points(self, player: str, match_id: int) -> Tuple[float, Dict]:
        """
        Calculate bowling points for a player in a specific match.
        
        Args:
            player: Player name
            match_id: Match ID
            
        Returns:
            Tuple of (total_points, breakdown_dict)
        """
        # Get all balls bowled by player
        player_balls = self.df[
            (self.df['match_id'] == match_id) & 
            (self.df['bowler'] == player)
        ].copy()
        
        if len(player_balls) == 0:
            return 0.0, {}
        
        breakdown = {}
        total_points = 0.0
        
        # Calculate wickets
        wickets = len(player_balls[player_balls['is_wicket'] == True])
        breakdown['wickets'] = wickets
        total_points += wickets * self.POINTS['wicket']
        
        # Wicket haul bonuses
        if wickets >= 5:
            breakdown['5w_haul'] = True
            total_points += self.POINTS['bonus_5w']
        elif wickets >= 4:
            breakdown['4w_haul'] = True
            total_points += self.POINTS['bonus_4w']
        elif wickets >= 3:
            breakdown['3w_haul'] = True
            total_points += self.POINTS['bonus_3w']
        
        # Maiden overs (no runs in an over)
        # Group by over and check if total runs = 0
        player_balls['over'] = player_balls['ball'].apply(lambda x: int(x))
        maidens = 0
        for over_num in player_balls['over'].unique():
            over_balls = player_balls[player_balls['over'] == over_num]
            if over_balls['total_runs'].sum() == 0 and len(over_balls) >= 6:
                maidens += 1
        
        breakdown['maidens'] = maidens
        total_points += maidens * self.POINTS['maiden']
        
        # Economy rate bonus/penalty (min 2 overs = 12 balls)
        balls_bowled = len(player_balls)
        runs_conceded = player_balls['total_runs'].sum()
        breakdown['balls_bowled'] = balls_bowled
        breakdown['runs_conceded'] = runs_conceded
        
        if balls_bowled >= 12:
            overs = balls_bowled / 6
            economy = runs_conceded / overs
            breakdown['economy'] = economy
            
            if economy < 5:
                total_points += self.POINTS['economy_below_5']
                breakdown['economy_bonus'] = self.POINTS['economy_below_5']
            elif economy < 6:
                total_points += self.POINTS['economy_5_to_5.99']
                breakdown['economy_bonus'] = self.POINTS['economy_5_to_5.99']
            elif economy <= 7:
                total_points += self.POINTS['economy_6_to_7']
                breakdown['economy_bonus'] = self.POINTS['economy_6_to_7']
            elif economy >= 12:
                total_points += self.POINTS['economy_above_12']
                breakdown['economy_penalty'] = self.POINTS['economy_above_12']
            elif economy >= 11:
                total_points += self.POINTS['economy_11_to_12']
                breakdown['economy_penalty'] = self.POINTS['economy_11_to_12']
            elif economy >= 10:
                total_points += self.POINTS['economy_10_to_11']
                breakdown['economy_penalty'] = self.POINTS['economy_10_to_11']
        
        return total_points, breakdown
    
    def calculate_fielding_points(self, player: str, match_id: int) -> Tuple[float, Dict]:
        """
        Calculate fielding points for a player in a specific match.
        
        Args:
            player: Player name
            match_id: Match ID
            
        Returns:
            Tuple of (total_points, breakdown_dict)
        """
        match_balls = self.df[self.df['match_id'] == match_id].copy()
        
        breakdown = {}
        total_points = 0.0
        
        # Catches (fielders involved in catch dismissals)
        # Note: Cricsheet data may not always have fielder info, using approximation
        catches = len(match_balls[
            (match_balls['wicket_type'] == 'caught') & 
            (match_balls['player_dismissed'] != player)
        ])
        # This is approximate - ideally we'd track who took the catch
        # For now, we'll skip individual fielding attribution
        breakdown['catches'] = 0
        
        # Stumpings (for wicket keepers)
        stumpings = len(match_balls[
            (match_balls['wicket_type'] == 'stumped')
        ])
        breakdown['stumpings'] = 0
        
        # Run outs
        run_outs = len(match_balls[
            (match_balls['wicket_type'] == 'run out')
        ])
        breakdown['run_outs'] = 0
        
        # Note: Without detailed fielding data, we can't accurately assign fielding points
        # This would require additional data about who took catches, who effected run-outs etc.
        
        return total_points, breakdown
    
    def calculate_total_points(self, player: str, match_id: int) -> Tuple[float, Dict]:
        """
        Calculate total fantasy points for a player in a match.
        
        Args:
            player: Player name
            match_id: Match ID
            
        Returns:
            Tuple of (total_points, complete_breakdown)
        """
        batting_pts, batting_breakdown = self.calculate_batting_points(player, match_id)
        bowling_pts, bowling_breakdown = self.calculate_bowling_points(player, match_id)
        fielding_pts, fielding_breakdown = self.calculate_fielding_points(player, match_id)
        
        total = batting_pts + bowling_pts + fielding_pts
        
        breakdown = {
            'batting_points': batting_pts,
            'bowling_points': bowling_pts,
            'fielding_points': fielding_pts,
            'total_points': total,
            'batting_breakdown': batting_breakdown,
            'bowling_breakdown': bowling_breakdown,
            'fielding_breakdown': fielding_breakdown
        }
        
        return total, breakdown
    
    def create_training_dataset(self) -> pd.DataFrame:
        """
        Create training dataset with player-match level fantasy points.
        
        Returns:
            DataFrame with columns: match_id, player, team, venue, fantasy_points, etc.
        """
        training_data = []
        
        # Get all unique matches
        matches = self.df['match_id'].unique()
        
        print(f"Processing {len(matches)} matches for training data...")
        
        for i, match_id in enumerate(matches):
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(matches)} matches...")
            
            match_data = self.df[self.df['match_id'] == match_id]
            venue = match_data['venue'].iloc[0]
            date = match_data['start_date'].iloc[0]
            team1 = match_data['batting_team'].iloc[0]
            team2 = match_data['bowling_team'].iloc[0]
            
            # Get all players in this match
            players = set()
            players.update(match_data['striker'].unique())
            players.update(match_data['non_striker'].unique())
            players.update(match_data['bowler'].unique())
            players.discard('')
            
            for player in players:
                total_pts, breakdown = self.calculate_total_points(player, match_id)
                
                # Determine player's team
                player_team = None
                if player in match_data[match_data['batting_team'] == team1]['striker'].values:
                    player_team = team1
                elif player in match_data[match_data['batting_team'] == team2]['striker'].values:
                    player_team = team2
                elif player in match_data[match_data['bowling_team'] == team1]['bowler'].values:
                    player_team = team1
                else:
                    player_team = team2
                
                opposition = team2 if player_team == team1 else team1
                
                training_data.append({
                    'match_id': match_id,
                    'date': date,
                    'player': player,
                    'team': player_team,
                    'opposition': opposition,
                    'venue': venue,
                    'fantasy_points': total_pts,
                    'batting_points': breakdown['batting_points'],
                    'bowling_points': breakdown['bowling_points'],
                    'fielding_points': breakdown['fielding_points'],
                })
        
        df_training = pd.DataFrame(training_data)
        print(f"\nCreated training dataset with {len(df_training)} player-match records")
        
        return df_training


if __name__ == "__main__":
    # Test the calculator
    from data.data_loader import load_dataset
    
    df, loader = load_dataset()
    calculator = FantasyPointsCalculator(df)
    
    # Test on a random match
    match_id = df['match_id'].iloc[0]
    player = df['striker'].iloc[0]
    
    total, breakdown = calculator.calculate_total_points(player, match_id)
    print(f"\n{player} in match {match_id}:")
    print(f"Total Points: {total}")
    print(f"Breakdown: {breakdown}")
