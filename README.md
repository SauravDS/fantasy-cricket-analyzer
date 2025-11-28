# Fantasy Cricket Team Predictor 🏏

An ML-powered web application that predicts optimal Dream11 fantasy cricket teams using historical CSA T20 Challenge ball-by-ball data.

## Features

- **Machine Learning Predictions**: XGBoost/Random Forest models trained on historical fantasy points
- **Dream11 Point System**: Full implementation of official Dream11 scoring rules
- **Contextual Analysis**: Ground-specific and opposition-specific performance metrics
- **Smart Team Selection**: Automated selection of top 14 players with balanced composition
- **Captain/VC Recommendations**: Data-driven captain and vice-captain suggestions
- **Interactive Web UI**: Beautiful Streamlit interface with step-by-step wizard

## Project Structure

```
fantasy-cricket-analyzer/
├── all_matches.csv              # CSA T20 Challenge ball-by-ball dataset
├── app.py                       # Main Streamlit application
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── src/
│   ├── data/
│   │   ├── data_loader.py      # CSV loading and preprocessing
│   │   └── data_aggregator.py  # Player statistics aggregation
│   │
│   ├── features/
│   │   ├── player_features.py       # Batting/bowling feature extraction
│   │   └── contextual_features.py   # Ground/opposition features
│   │
│   ├── fantasy/
│   │   └── points_calculator.py     # Dream11 points calculation
│   │
│   ├── ml/
│   │   └── predictor.py             # ML prediction engine
│   │
│   └── optimization/
│       └── team_selector.py         # Fantasy team optimization
│
├── scripts/
│   └── train_model.py          # Model training script
│
└── models/
    ├── fantasy_predictor.pkl   # Trained ML model (generated)
    └── feature_names.pkl       # Feature names (generated)
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the ML Model

Before running the app, you need to train the machine learning model:

```bash
python scripts/train_model.py
```

This will:
- Load and process the dataset
- Calculate historical fantasy points
- Extract features for all player-match combinations
- Train multiple ML models (Random Forest, XGBoost, Gradient Boosting)
- Save the best performing model to `models/fantasy_predictor.pkl`

**Note**: Training may take 10-30 minutes depending on your hardware.

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

### Step 1: Select Teams
- Choose two teams that will compete in the match
- Example: Warriors vs Titans

### Step 2: Select Ground
- Pick the venue where the match will be played
- View historical ground statistics

### Step 3: Build Player Pool
- Select exactly 22 players from both teams
- Assign each player to their respective team (Team 1 or Team 2)

### Step 4: Get Predictions
- Click "Generate Predictions"
- ML model analyzes each player's:
  - Historical performance
  - Recent form
  - Ground suitability
  - Opposition matchup
  - Consistency metrics

### Step 5: Review Fantasy Team
- View top 14 recommended players
- See captain and vice-captain suggestions
- Review complete 22-player ranking
- Check expected fantasy points

## Dream11 Point System

### Batting
- Run: +1 point
- Boundary (4): +1 bonus
- Six (6): +2 bonus
- 30 runs: +4 bonus
- Half-century (50): +8 bonus
- Century (100): +16 bonus
- Duck (dismissed for 0): -2 points

### Bowling
- Wicket: +25 points
- 3-wicket haul: +4 bonus
- 4-wicket haul: +8 bonus
- 5-wicket haul: +16 bonus
- Maiden over: +12 points

### Fielding
- Catch: +8 points
- Stumping: +12 points
- Run out (direct): +12 points
- Run out (indirect): +6 points

### Economy Rate (min 2 overs)
- Below 5: +6 points
- 5-5.99: +4 points
- 6-7: +2 points
- 10-11: -2 points
- 11-12: -4 points
- Above 12: -6 points

### Strike Rate (min 10 balls)
- Above 170: +6 points
- 150-170: +4 points
- 130-150: +2 points
- 60-70: -2 points
- 50-60: -4 points
- Below 50: -6 points

## ML Model Details

### Features Used
- **Batting Features**: Average runs, strike rate, recent form, consistency
- **Bowling Features**: Average wickets, economy rate, recent form, consistency
- **Contextual Features**: Performance at specific grounds, vs specific opposition
- **Form Features**: Last 5 matches performance with recency weighting
- **Consistency Features**: Standard deviation and coefficient of variation

### Models Evaluated
1. Random Forest Regressor
2. XGBoost Regressor
3. Gradient Boosting Regressor

The best performing model (typically XGBoost) is automatically selected based on R² score.

### Training Methodology
- **Temporal Split**: Uses date-based train/test split to avoid data leakage
- **Feature Engineering**: Extracts features from data BEFORE each match
- **Target Variable**: Historical Dream11 fantasy points calculated from actual match data

## Dataset Information

- **Source**: Cricsheet (CSA T20 Challenge)
- **Records**: 65,051 ball-by-ball entries
- **Matches**: ~100+ matches
- **Teams**: 11 CSA teams
- **Venues**: 16 cricket grounds across South Africa
- **Columns**: 22 fields including match details, players, runs, wickets, dismissals

## Technical Stack

- **Python 3.8+**
- **Streamlit**: Web UI framework
- **pandas**: Data manipulation
- **scikit-learn**: ML models and preprocessing
- **XGBoost**: Gradient boosting
- **joblib**: Model persistence
- **matplotlib/seaborn/plotly**: Visualizations

## Limitations

- Predictions are based on historical data only
- Does not account for:
  - Player injuries
  - Recent team changes
  - Weather conditions
  - Match importance/pressure
  - Real-time match situation
- Fielding points are approximate (limited fielding data in Cricsheet)

## Future Enhancements

- Add player injury tracking
- Incorporate weather data
- Live match prediction updates
- More advanced ensemble models
- Player comparison visualizations
- Historical prediction accuracy tracking
- Export predictions to CSV/PDF

## License

MIT License

## Acknowledgments

- Data source: Cricsheet (https://cricsheet.org/)
- Dream11 scoring system reference
- CSA T20 Challenge

---

**Built with ❤️ for cricket fantasy enthusiasts**
