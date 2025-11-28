"""
Team optimization module for selecting best fantasy team.
Implements Dream11 composition constraints.
"""

import pandas as pd
from typing import List, Dict, Tuple


class TeamSelector:
    """Select optimal fantasy team with Dream11 constraints."""
    
    # Dream11 constraints
    TEAM_SIZE = 11  # For playing XI
    FANTASY_TEAM_SIZE = 14  # For fantasy team recommendation
    MIN_BATSMEN = 1
    MAX_BATSMEN = 8
    MIN_BOWLERS = 1
    MAX_BOWLERS = 8
    MIN_ALL_ROUNDERS = 1
    MAX_ALL_ROUNDERS = 8
    MIN_WK = 1
    MAX_WK = 8
    MIN_PER_TEAM = 1
    MAX_PER_TEAM = 10
    
    def __init__(self):
        """Initialize team selector."""
        pass
    
    def determine_player_role(self, player_stats: Dict) -> str:
        """
        Determine player role based on statistics.
        
        Args:
            player_stats: Dictionary with player batting/bowling stats
            
        Returns:
            Role: 'Batsman', 'Bowler', 'All-rounder', 'WK'
        """
        bat_avg = player_stats.get('bat_avg_runs', 0)
        bowl_avg = player_stats.get('bowl_avg_wickets', 0)
        bat_matches = player_stats.get('bat_matches', 0)
        bowl_matches = player_stats.get('bowl_matches', 0)
        
        # Simple role determination logic
        # Improved role determination logic
        if bat_matches > 0 and bowl_matches > 0:
            # All-rounder if decent contribution in both
            if bat_avg > 10 and bowl_avg > 0.3:
                return 'All-rounder'
            elif bat_avg > 15:
                return 'Batsman'
            elif bowl_avg > 0.5:
                return 'Bowler'
            else:
                return 'All-rounder' # Utility player
        elif bat_matches > 0:
            return 'Batsman'
        elif bowl_matches > 0:
            return 'Bowler'
        else:
            # If no stats, check if they are likely a bowler (often bat at end)
            return 'Batsman'  # Default fallback
    
    def validate_team_composition(self, team: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate if team meets Dream11 composition rules.
        
        Args:
            team: DataFrame with selected players
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(team) > self.FANTASY_TEAM_SIZE:
            return False, f"Team has {len(team)} players, max is {self.FANTASY_TEAM_SIZE}"
        
        # Determine roles for each player
        roles = []
        for _, player in team.iterrows():
            role = self.determine_player_role(player)
            roles.append(role)
        
        team['role'] = roles
        
        # Count by role
        role_counts = team['role'].value_counts().to_dict()
        batsmen = role_counts.get('Batsman', 0)
        bowlers = role_counts.get('Bowler', 0)
        all_rounders = role_counts.get('All-rounder', 0)
        wk = role_counts.get('WK', 0)
        
        # Validate constraints
        if batsmen < self.MIN_BATSMEN:
            return False, f"Need at least {self.MIN_BATSMEN} batsman"
        if bowlers < self.MIN_BOWLERS:
            return False, f"Need at least {self.MIN_BOWLERS} bowler"
        
        # Count by team
        team_counts = team['team'].value_counts()
        for team_name, count in team_counts.items():
            if count < self.MIN_PER_TEAM or count > self.MAX_PER_TEAM:
                return False, f"Team {team_name} has {count} players (must be {self.MIN_PER_TEAM}-{self.MAX_PER_TEAM})"
        
        return True, "Valid composition"
    
    def select_fantasy_team(self, ranked_players: pd.DataFrame, 
                           team_tags: Dict[str, str]) -> pd.DataFrame:
        """
        Select top 14 players for fantasy team with balanced composition.
        
        Args:
            ranked_players: DataFrame with all players ranked by predicted points
            team_tags: Dictionary mapping player names to their teams
            
        Returns:
            DataFrame with selected 14 players
        """
        # Add team tags to players
        ranked_players['team'] = ranked_players['player'].map(team_tags)
        
        # Add roles
        roles = []
        for _, player in ranked_players.iterrows():
            role = self.determine_player_role(player)
            roles.append(role)
        ranked_players['role'] = roles
        
        # Greedy selection: pick top players while respecting constraints
        selected = []
        team_counts = {}
        role_counts = {'Batsman': 0, 'Bowler': 0, 'All-rounder': 0, 'WK': 0}
        
        for _, player in ranked_players.iterrows():
            if len(selected) >= self.FANTASY_TEAM_SIZE:
                break
            
            player_team = player['team']
            player_role = player['role']
            
            # Check team constraint
            current_team_count = team_counts.get(player_team, 0)
            if current_team_count >= self.MAX_PER_TEAM:
                continue
            
            # Check role constraints (soft constraints for flexibility)
            if player_role == 'Batsman' and role_counts['Batsman'] >= self.MAX_BATSMEN:
                continue
            if player_role == 'Bowler' and role_counts['Bowler'] >= self.MAX_BOWLERS:
                continue
            if player_role == 'All-rounder' and role_counts['All-rounder'] >= self.MAX_ALL_ROUNDERS:
                continue
            if player_role == 'WK' and role_counts['WK'] >= self.MAX_WK:
                continue
            
            # Add player
            selected.append(player)
            team_counts[player_team] = current_team_count + 1
            role_counts[player_role] += 1
        
        fantasy_team = pd.DataFrame(selected)
        
        # Ensure minimum requirements
        if len(fantasy_team) < 11:
            print("Warning: Could not select 11 players with constraints")
        
        return fantasy_team.reset_index(drop=True)
    
    def suggest_captain_vice_captain(self, fantasy_team: pd.DataFrame) -> Tuple[str, str]:
        """
        Suggest captain and vice-captain (top 2 predicted performers).
        
        Args:
            fantasy_team: Selected fantasy team DataFrame
            
        Returns:
            Tuple of (captain_name, vice_captain_name)
        """
        sorted_team = fantasy_team.sort_values('predicted_points', ascending=False)
        
        captain = sorted_team.iloc[0]['player']
        vice_captain = sorted_team.iloc[1]['player'] if len(sorted_team) > 1 else captain
        
        return captain, vice_captain
    
    def calculate_expected_team_points(self, fantasy_team: pd.DataFrame,
                                      captain: str, vice_captain: str) -> float:
        """
        Calculate expected total points with captain multipliers.
        
        Args:
            fantasy_team: Selected fantasy team
            captain: Captain name (2x points)
            vice_captain: Vice-captain name (1.5x points)
            
        Returns:
            Expected total fantasy points
        """
        total = 0
        
        for _, player in fantasy_team.iterrows():
            points = player['predicted_points']
            player_name = player['player']
            
            if player_name == captain:
                total += points * 2
            elif player_name == vice_captain:
                total += points * 1.5
            else:
                total += points
        
        return total


if __name__ == "__main__":
    # Test
    selector = TeamSelector()
    
    # Create mock data
    mock_players = pd.DataFrame({
        'player': [f'Player{i}' for i in range(22)],
        'predicted_points': [50 - i for i in range(22)],
        'bat_avg_runs': [20, 18, 15, 10, 8, 5, 3, 1, 0, 0, 0, 15, 12, 10, 8, 5, 3, 2, 1, 0, 0, 0],
        'bowl_avg_wickets': [0, 0, 0, 0, 0, 1, 2, 3, 4, 3, 2, 0, 0, 1, 2, 3, 4, 3, 2, 1, 1, 0],
    })
    
    team_tags = {f'Player{i}': 'Team1' if i < 11 else 'Team2' for i in range(22)}
    
    fantasy_team = selector.select_fantasy_team(mock_players, team_tags)
    print(f"Selected {len(fantasy_team)} players")
    print(fantasy_team[['player', 'team', 'role', 'predicted_points']].to_string())
    
    captain, vc = selector.suggest_captain_vice_captain(fantasy_team)
    print(f"\nCaptain: {captain}, Vice-Captain: {vc}")
