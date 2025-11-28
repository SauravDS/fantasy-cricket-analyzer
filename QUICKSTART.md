# QUICKSTART GUIDE 🚀

Get started with the Fantasy Cricket Team Predictor in 3 easy steps!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Your `all_matches.csv` file in the project root

## Quick Setup

### 1. Install Dependencies (30 seconds)

```bash
pip install -r requirements.txt
```

This installs:
- streamlit (web UI)
- pandas, numpy (data processing)
- scikit-learn, xgboost (machine learning)
- matplotlib, seaborn, plotly (visualizations)

### 2. Train the ML Model (10-30 minutes)

```bash
python scripts/train_model.py
```

What happens:
1. ✅ Loads 65K+ ball-by-ball records
2. ✅ Calculates historical Dream11 fantasy points
3. ✅ Extracts player features (batting, bowling, form, ground stats)
4. ✅ Trains 3 ML models (Random Forest, XGBoost, Gradient Boosting)
5. ✅ Saves best model to `models/fantasy_predictor.pkl`

**Expected Output:**
```
FANTASY CRICKET ML MODEL TRAINING
============================================================
Loading dataset...
Loaded 65051 ball records from 288 matches
Calculating fantasy points...
Processing 288 matches for training data...
Created training dataset with 5000+ player-match records
Engineering features...

Training Random Forest...
Random Forest Performance:
  MAE:  12.34
  RMSE: 18.56
  R²:   0.678

Training XGBoost...
XGBoost Performance:
  MAE:  11.23
  RMSE: 16.89
  R²:   0.723

Training Gradient Boosting...
Gradient Boosting Performance:
  MAE:  11.56
  RMSE: 17.23
  R²:   0.705

==================================================
Best Model: XGBoost (R² = 0.723)
==================================================

Model saved to: models/fantasy_predictor.pkl

TRAINING COMPLETE!
```

### 3. Run the App (Instantly!)

```bash
streamlit run app.py
```

Your browser will open to `http://localhost:8501`

---

## Using the App

### Flow Diagram
```
🏠 Home → 🎯 Teams → 🏟️ Ground → 👥 Players → 📊 Predictions
```

### Detailed Steps

#### Step 1: Navigate to "Team Selection"
- Choose **Team 1** (e.g., Warriors)
- Choose **Team 2** (e.g., Titans)
- Click "Confirm Team Selection"

#### Step 2: Navigate to "Ground Selection"
- Select venue (e.g., Newlands, Cape Town)
- View ground statistics
- Click "Confirm Ground Selection"

#### Step 3: Navigate to "Player Pool"
- Select exactly **22 players** from the dropdown
- For each player, assign to Team 1 or Team 2
- Click "Confirm Player Selection & Tags"

#### Step 4: Navigate to "Predictions"
- Click "🚀 Generate Predictions"
- Wait 5-10 seconds for ML processing

#### Step 5: Review Results
- **Top 14 Fantasy Team** displayed
- **Captain** (2x points) highlighted in gold
- **Vice-Captain** (1.5x points) highlighted in silver
- **All 22 players ranked** by predicted fantasy points
- **Expected team total** calculated

---

## Example Walkthrough

### Sample Match Setup
- **Teams**: Western Province vs Boland
- **Venue**: Newlands, Cape Town
- **Sample Players** (22):
  - **Western Province**: BE Hendricks, GA Stuurman, K Verreynne, etc.
  - **Boland**: PJ Malan, JN Malan, C Jonker, etc.

### Expected Results
```
Top 14 Fantasy Team:
Rank Player             Team              Role         Points
1    BE Hendricks      Western Province  Bowler       38.5
2    PJ Malan          Boland            Batsman      36.2
3    K Verreynne       Western Province  Batsman      34.8
...

👑 Captain: BE Hendricks (77.0 points with 2x)
🥈 Vice-Captain: PJ Malan (54.3 points with 1.5x)

💯 Expected Team Total: 425.6 points
```

---

## Troubleshooting

### Issue: "ML Model Not Found"
**Solution**: Run `python scripts/train_model.py` first

### Issue: "ModuleNotFoundError: No module named 'xgboost'"
**Solution**: Run `pip install -r requirements.txt`

### Issue: "FileNotFoundError: all_matches.csv"
**Solution**: Ensure the CSV file is in the project root directory

### Issue: Training is slow
**Normal**: Training on 65K records with feature engineering takes time
- Expected: 10-30 minutes depending on your CPU
- Progress updates: Printed every 100 records

### Issue: Predictions seem off
**Check**:
1. Model R² score should be > 0.6 (displayed during training)
2. Players have sufficient historical data
3. Selected players actually played for the selected teams

---

## Tips for Best Results

1. **More Data = Better Predictions**: The model learns from historical matches. Players with more match history get more accurate predictions.

2. **Recent Form Matters**: The model weights recent matches more heavily.

3. **Ground Context**: Players who performed well at the selected venue in the past will score higher.

4. **Opposition Matchup**: Historical performance against the opposing team is factored in.

5. **Captain Choice**: The app suggests the player with highest predicted points as captain for maximum fantasy points.

---

## What's Next?

After getting comfortable with the app:

1. **Experiment**: Try different team combinations
2. **Compare**: Test predictions against actual Dream11 outcomes (if available)
3. **Customize**: Modify ML model parameters in `scripts/train_model.py`
4. **Enhance**: Add more features (weather, pitch type, etc.)

---

## Support

For issues or questions:
- Check the full [README.md](README.md) for detailed documentation
- Review the code comments in each module
- Ensure Python 3.8+ is installed

---

**Happy Predicting! 🏏✨**
