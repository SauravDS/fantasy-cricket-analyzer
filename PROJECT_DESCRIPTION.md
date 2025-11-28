# Fantasy Cricket Team Predictor
## CSA T20 Challenge Match Intelligence System

### 🎯 Project Overview
A comprehensive web-based fantasy cricket prediction platform that leverages historical ball-by-ball data from the CSA (Cricket South Africa) T20 Challenge to generate data-driven fantasy team recommendations. The system analyzes player performance, ground statistics, and head-to-head matchups to predict optimal Dream11 fantasy teams for upcoming matches.

---

## 📊 Dataset Information
- **Source**: Cricsheet ball-by-ball dataset
- **Competition**: CSA T20 Challenge (South Africa)
- **Data Points**: 65,051 ball-by-ball records
- **Teams Covered**: 11 teams (Boland, Cape Cobras, Dolphins, Impi, Knights, KwaZulu-Natal Inland, Lions, North West, Titans, Warriors, Western Province)
- **Venues**: 16 cricket grounds across South Africa
- **Data Fields**: 22 columns including match details, player names, runs, wickets, extras, and dismissal information

---

## 🎮 Application Features

### 1. **Match Setup Wizard**
**Step 1: Team Selection**
- Interactive UI to select two competing teams from the CSA Challenge roster
- Visual team cards with team colors and historical statistics
- Validation to ensure two different teams are selected

**Step 2: Ground Selection**
- Dropdown/card-based selection of venue from 16 available grounds
- Display ground statistics: average scores, wicket trends, pitch behavior
- Historical performance context for selected teams at the chosen venue

### 2. **Player Pool Management**
**Smart Player Discovery**
- Automatically aggregates all unique players who have represented either selected team
- Displays comprehensive player pool from both teams combined
- Shows player roles (Batsman, Bowler, All-rounder, Wicket-keeper)

**Player Tagging Interface**
- User selects exactly 22 players from the pool
- For each selected player, user tags them to their respective team
- Visual badges showing team association
- Real-time validation ensuring balanced selection

### 3. **Team Lineup Generator**
**Intelligent Lineup Creation**
- Processes the 22 tagged players to generate two playing XI lineups
- Ensures balanced team composition:
  - Minimum batsmen, bowlers, all-rounders
  - Exactly 1 wicket-keeper per team
  - 11 players per team
- Displays lineups side-by-side with role distribution

### 4. **Statistical Analysis Engine**

**Individual Player Statistics**
- **Batting Metrics**: Runs scored, strike rate, boundaries (4s/6s), average, consistency
- **Bowling Metrics**: Wickets taken, economy rate, bowling average, dot ball percentage
- **All-rounder Index**: Combined performance score
- **Recent Form**: Last 5 matches performance trend

**Ground-Specific Analysis**
- Player performance at the selected venue
- Ground-specific batting/bowling averages
- Venue suitability scores for each player

**Head-to-Head Matchup Analysis**
- Historical performance in matches between the two selected teams
- Player vs specific opposition statistics
- Wicket-taking potential against specific batting lineups
- Run-scoring potential against specific bowling attacks

### 5. **Fantasy Team Predictor**

**Dream11 Point System Implementation**
- **Batting Points**:
  - Run: +1 point per run
  - Boundary (4): +1 bonus point
  - Six (6): +2 bonus points
  - 30-run bonus: +4 points
  - Half-century (50): +8 points
  - Century (100): +16 points
  - Duck (0 runs by batsman): -2 points

- **Bowling Points**:
  - Wicket: +25 points
  - 3-wicket bonus: +4 points
  - 4-wicket bonus: +8 points
  - 5-wicket bonus: +16 points
  - Maiden over: +12 points

- **Fielding Points**:
  - Catch: +8 points
  - Stumping: +12 points
  - Run out (direct): +12 points
  - Run out (indirect): +6 points

- **Economy Rate (Bowling) - Minimum 2 overs**:
  - Below 5 runs/over: +6 points
  - Between 5-5.99: +4 points
  - Between 6-7: +2 points
  - Between 10-11: -2 points
  - Between 11-12: -4 points
  - Above 12: -6 points

- **Strike Rate (Batting) - Minimum 10 balls**:
  - Above 170: +6 points
  - 150-170: +4 points
  - 130-150: +2 points
  - 60-70: -2 points
  - 50-60: -4 points
  - Below 50: -6 points

**Prediction Algorithm**
- Analyzes historical performance using Dream11 scoring
- Calculates expected fantasy points for each of the 22 players
- Considers:
  - Individual form and statistics
  - Ground/venue suitability
  - Head-to-head performance
  - Role importance in match context
  - Consistency and reliability metrics

**Final Output**
- Ranked list of all 22 players by predicted fantasy points
- **Top 14 Players** highlighted as the recommended fantasy team
- Suggested team composition:
  - Captain (2x points) recommendation
  - Vice-Captain (1.5x points) recommendation
  - Balanced role distribution (batsmen, bowlers, all-rounders, wicket-keeper)
- Detailed reasoning for each player's ranking
- Expected points range for each player

### 6. **Interactive Dashboard**
- Visual charts showing predicted performance distribution
- Team comparison metrics
- Downloadable fantasy team sheet
- Match prediction summary

---

## 🛠️ Technical Architecture

### Frontend
- **Framework**: Vanilla HTML, CSS, JavaScript (or Vite/Next.js for enhanced performance)
- **Styling**: Modern CSS with responsive design, gradients, and smooth animations
- **Data Visualization**: Chart.js or D3.js for statistical visualizations
- **User Experience**: Step-by-step wizard interface with progress indicators

### Backend/Data Processing
- **Data Handling**: JavaScript-based CSV parsing (Papa Parse library)
- **Statistical Calculations**: Custom algorithms for metrics computation
- **Caching**: Local storage for processed statistics to improve performance
- **Fantasy Points Engine**: Modular calculator following Dream11 rules

### Data Flow
```
all_matches.csv → Parser → Statistical Analyzer → Matchup Engine → Fantasy Predictor → UI Display
```

### Key Modules
1. **DataLoader**: CSV parsing and data structuring
2. **StatsCalculator**: Player and ground statistics computation
3. **MatchupAnalyzer**: Head-to-head performance analysis
4. **FantasyPointsEngine**: Dream11 points calculation
5. **PredictionModel**: Player ranking and team recommendation
6. **UIController**: User interface management and interaction handling

---

## 🎨 Design Principles
- **Premium Aesthetic**: Modern, vibrant color schemes with cricket-themed gradients
- **Intuitive Navigation**: Clear step-by-step flow with contextual guidance
- **Data-Driven Insights**: Visual representations of statistics and predictions
- **Responsive Design**: Seamless experience across desktop, tablet, and mobile devices
- **Performance Optimized**: Fast data processing with smooth animations

---

## 🚀 User Journey

1. **Land on Home Page** → See app introduction and "Start New Prediction" button
2. **Select Teams** → Choose two competing teams from CSA Challenge
3. **Choose Ground** → Select venue with contextual statistics
4. **Build Player Pool** → View all available players from both teams
5. **Tag 22 Players** → Select and assign players to their respective teams
6. **View Lineups** → See generated playing XI for both teams
7. **Get Predictions** → Receive ranked fantasy team recommendations
8. **Review Insights** → Explore detailed statistics and reasoning
9. **Export Team** → Download or share the fantasy team

---

## 📈 Success Metrics
- Accurate player statistics computation from historical data
- Meaningful predictions based on multi-dimensional analysis
- Smooth, lag-free user experience
- Clear, actionable fantasy team recommendations
- Professional, engaging visual design

---

## 🎯 Target Users
- Fantasy cricket enthusiasts
- CSA T20 Challenge followers
- Data-driven sports bettors
- Cricket analysts and strategists
- Dream11 platform users

---

## 🔮 Future Enhancements (Out of Current Scope)
- Live match integration
- Player injury and availability tracking
- Weather condition analysis
- Machine learning-based predictions
- Multi-platform mobile apps
- User accounts and prediction history
- Community features and leaderboards

---

**Project Goal**: Deliver a fully functional, visually stunning, and statistically robust fantasy cricket prediction web app that helps users make informed Dream11 team selections for CSA T20 Challenge matches.
