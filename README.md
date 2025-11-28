# Fantasy Cricket Analytics Platform

A machine learning-powered web application for predicting fantasy cricket player performance and generating optimal team selections. Upload any cricket dataset, train custom models, and generate data-driven predictions for Dream11 and other fantasy platforms.

## Features

- Universal dataset support for any T20 cricket league (IPL, BBL, CPL, etc.)
- On-demand ML model training with configurable scope
- Multiple algorithm evaluation (Random Forest, XGBoost, Gradient Boosting)
- Dream11-compliant fantasy points calculation
- Contextual analysis (venue, opposition, form, consistency)
- Interactive team builder with 22-player pool
- Model library for managing multiple league-specific models
- Material Design UI with professional analytics dashboard

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Start the Application

```bash
streamlit run app.py
```

The application will launch at `http://localhost:8501`

### 2. Upload Dataset

Navigate to "Data Ingestion" and upload a ball-by-ball cricket CSV file with the following required columns:

- `match_id`
- `batting_team`
- `bowling_team`
- `striker`
- `bowler`
- `runs_off_bat`
- `extras`
- `venue`

The platform automatically creates a `total_runs` column from `runs_off_bat` and `extras`.

### 3. Train Model

Go to "Model Training" to configure and execute the training protocol:

- Choose training scope (complete dataset or recent matches only)
- Monitor real-time progress and performance metrics
- Save trained models to the repository for future use

### 4. Generate Predictions

Complete the workflow:

1. Select competing teams (Squad Configuration)
2. Choose match venue (Venue Analysis)
3. Build 22-player pool (Roster Management)
4. Generate performance forecast

The system will output:
- Ranked list of all 22 players with predicted points
- Top 3 picks (Captain, Vice-Captain, Top Pick)
- Optimal 11-player lineup

## Project Structure

```
fantasy-cricket-analyzer/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── models/
│   └── library/                    # Saved model repository
├── src/
│   ├── data/
│   │   └── data_loader.py         # Dataset processing
│   ├── fantasy/
│   │   └── points_calculator.py   # Dream11 points engine
│   ├── ml/
│   │   ├── feature_engineering.py # Feature extraction
│   │   ├── trainer.py             # Model training pipeline
│   │   ├── predictor.py          # Prediction engine
│   │   └── model_library.py      # Model persistence
│   └── optimization/
│       └── team_selector.py       # Team optimization
└── scripts/
    └── train_model.py             # Offline training script
```

## Technical Details

### Machine Learning Pipeline

The training pipeline implements:

1. Data validation and preprocessing
2. Fantasy points calculation from historical data
3. Feature engineering (batting, bowling, form, consistency, venue, opposition)
4. Multi-model training with cross-validation
5. Automatic best model selection based on R² score
6. Model persistence with metadata

### Prediction Features

Per player, the model considers:

- Batting: average, strike rate, boundaries, high scores
- Bowling: wickets, economy, maidens, consistency
- Form: weighted recent performance (last 5 matches)
- Consistency: standard deviation, coefficient of variation
- Venue: historical performance at selected ground
- Opposition: matchup-specific statistics

### Dream11 Scoring System

Full implementation of official Dream11 point rules:

**Batting**: Runs (+1/run), boundaries (+1/4, +2/6), milestones (30/50/100 runs), duck penalty (-2)

**Bowling**: Wickets (+25), wicket hauls (+4/8/16), maidens (+12)

**Fielding**: Catches (+8), stumpings (+12), run-outs (+6/12)

**Bonuses/Penalties**: Economy rate bonuses/penalties, strike rate bonuses/penalties

## Technology Stack

- Python 3.8+
- Streamlit (web framework)
- pandas (data processing)
- scikit-learn (ML models, preprocessing)
- XGBoost (gradient boosting)
- joblib (model serialization)

## Limitations

The system does not account for:

- Player injuries or unavailability
- Real-time team changes or announcements
- Weather conditions or pitch reports
- Match context (knockout vs league stage)
- In-play match dynamics

Predictions are statistical estimates based on historical data only.

## License

MIT License

## Acknowledgments

Dataset compatibility: Cricsheet format (https://cricsheet.org/)

Created by Saurav
