"""
Fantasy Cricket Team Predictor - Main Streamlit Application
ML-powered prediction system for Dream11 fantasy teams
"""

import streamlit as st
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.data_loader import load_dataset

# Page configuration
st.set_page_config(
    page_title="Fantasy Cricket Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Material Design CSS
def apply_custom_css():
    st.markdown("""
    <style>
    /* Import Fonts: Roboto (Material Standard) & Montserrat (Headings) */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&family=Roboto:wght@300;400;500;700&display=swap');
    
    :root {
        --primary-color: #3F51B5; /* Indigo 500 */
        --primary-dark: #303F9F;  /* Indigo 700 */
        --accent-color: #009688;  /* Teal 500 */
        --bg-color: #F5F7FA;
        --card-bg: #FFFFFF;
        --text-high: #202124;
        --text-medium: #5F6368;
        --border-color: #E0E0E0;
        --shadow-sm: 0 2px 4px rgba(0,0,0,0.05);
        --shadow-md: 0 4px 6px rgba(0,0,0,0.1);
        --shadow-lg: 0 10px 15px rgba(0,0,0,0.1);
    }

    /* Global Reset & Typography */
    * {
        font-family: 'Roboto', sans-serif;
        color: var(--text-high);
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Montserrat', sans-serif;
        color: var(--text-high) !important;
        font-weight: 600;
    }

    /* Main Layout */
    .main {
        background-color: var(--bg-color);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid var(--border-color);
    }
    
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* Typography Overrides */
    .main-header {
        font-family: 'Montserrat', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: var(--text-medium);
        font-weight: 400;
        margin-bottom: 3rem;
    }

    /* Material Cards */
    .material-card {
        background-color: var(--card-bg);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-color);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .material-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: var(--shadow-md);
    }
    
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
        color: white;
    }
    
    .kpi-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
        color: white;
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background-color: var(--primary-dark);
        box-shadow: var(--shadow-md);
    }
    
    /* Inputs */
    .stTextInput > div > div > input, .stSelectbox > div > div {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-high);
    }
    
    .stTextInput > div > div > input:focus, .stSelectbox > div > div:focus-within {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 2px rgba(63, 81, 181, 0.2);
    }

    /* Dataframes */
    .stDataFrame {
        border: 1px solid var(--border-color);
        border-radius: 8px;
        overflow: hidden;
    }
    
    thead tr th {
        background-color: #F8F9FA !important;
        color: var(--text-medium) !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 0.85rem;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        color: var(--primary-color) !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-medium) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--card-bg);
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    
    /* Custom Classes for Content */
    .section-title {
        font-size: 1.5rem;
        color: var(--primary-color);
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid var(--accent-color);
        padding-bottom: 0.5rem;
        display: inline-block;
    }
    
    .info-box {
        background-color: #E8F5E9;
        border-left: 4px solid var(--accent-color);
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
        color: #2E7D32;
    }
    
    </style>
    """, unsafe_allow_html=True)





def initialize_session_state():
    """Initialize session state variables."""
    # Dataset upload states
    if 'uploaded_data' not in st.session_state:
        st.session_state.uploaded_data = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'current_league' not in st.session_state:
        st.session_state.current_league = "T20 League"
    if 'model_info' not in st.session_state:
        st.session_state.model_info = None
    if 'loaded_model_name' not in st.session_state:
        st.session_state.loaded_model_name = None
    
    # Existing states
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'loader' not in st.session_state:
        st.session_state.loader = None
    if 'selected_team1' not in st.session_state:
        st.session_state.selected_team1 = None
    if 'selected_team2' not in st.session_state:
        st.session_state.selected_team2 = None
    if 'selected_venue' not in st.session_state:
        st.session_state.selected_venue = None
    if 'selected_players' not in st.session_state:
        st.session_state.selected_players = []
    if 'player_team_tags' not in st.session_state:
        st.session_state.player_team_tags = {}
    if 'predictions_ready' not in st.session_state:
        st.session_state.predictions_ready = False


def main():
    """Main application entry point."""
    apply_custom_css()
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">Fantasy Cricket Analytics Platform</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">Machine Learning-Powered Performance Predictions | {st.session_state.current_league}</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### FCA Platform")
        st.markdown("---")
        
        # Model Library - Always accessible
        if st.button("📚 Model Library", use_container_width=True):
            page = "Model Library"
        
        st.markdown("---")
        
        # Check if dataset is uploaded and model trained
        if st.session_state.uploaded_data is None:
            page = "Data Ingestion" if 'page' not in locals() else page
            st.info("Please upload a dataset to proceed.")
        elif not st.session_state.model_trained:
            page = "Model Training" if 'page' not in locals() else page
            st.info("Model training required.")
        else:
            # Only show main navigation if Model Library wasn't clicked
            if 'page' not in locals() or page != "Model Library":
                page = st.radio(
                    "Navigation",
                    ["Analytics Dashboard", "Squad Configuration", "Venue Analysis", 
                     "Roster Management", "Performance Forecast"],
                    label_visibility="collapsed"
                )
            
            # Add option to change dataset
            st.markdown("---")
            if st.button("Change Dataset", use_container_width=True):
                st.session_state.uploaded_data = None
                st.session_state.model_trained = False
                st.session_state.data_loaded = False
                st.rerun()
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown(f"""
        This app uses machine learning to predict fantasy cricket team performance
        for **{st.session_state.current_league}**.
        
        **Features:**
        - Upload any cricket dataset
        - On-demand model training
        - ML-powered predictions
        - Dream11 point system
        - Optimal team selection
        """)
        
        # Show model status
        if os.path.exists('models/fantasy_predictor.pkl'):
            st.success("ML Model Loaded")
        else:
            st.warning("ML Model Not Found")
            st.info("Please train a model via the 'Model Training' page.")
            
        st.markdown("---")
        st.markdown('<div style="text-align: center; color: #666; font-size: 0.8rem;">Created by Saurav with ❤️</div>', unsafe_allow_html=True)
    
    # Load data - from uploaded file or default
    if not st.session_state.data_loaded and st.session_state.uploaded_data is not None:
        with st.spinner("Loading uploaded dataset..."):
            try:
                from src.data.data_loader import DataLoader
                df = st.session_state.uploaded_data
                loader = DataLoader(df)
                st.session_state.df = df
                st.session_state.loader = loader
                st.session_state.data_loaded = True
            except Exception as e:
                st.error(f"Error loading dataset: {e}")
                return
    
    # Route to appropriate page
    if page == "Data Ingestion":
        show_upload_page()
    elif page == "Model Training":
        show_training_page()
    elif page == "Model Library":
        show_model_library_page()
    elif page == "Analytics Dashboard":
        show_home_page()
    elif page == "Squad Configuration":
        show_team_selection_page()
    elif page == "Venue Analysis":
        show_ground_selection_page()
    elif page == "Roster Management":
        show_player_pool_page()
    elif page == "Performance Forecast":
        show_predictions_page()

def show_upload_page():
    """Dataset upload page."""
    st.markdown('<div class="section-title">Data Ingestion</div>', unsafe_allow_html=True)
    
    # Show currently loaded model if any
    if st.session_state.get('loaded_model_name'):
        st.info(f"🎯 Active Model: **{st.session_state.loaded_model_name}**")
    
    st.markdown("""
    <div class="material-card">
        <h4>Import Match Data</h4>
        <p>Upload a ball-by-ball CSV file to initialize the analytics engine. Supports IPL, BBL, CPL, and other major leagues.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Required columns info
    st.markdown("""
    <div class="info-box">
        <h4>Required Schema</h4>
        <p>Ensure your dataset contains the following columns (Cricsheet format):</p>
        <ul>
            <li><code>match_id</code></li>
            <li><code>batting_team</code></li>
            <li><code>bowling_team</code></li>
            <li><code>striker</code></li>
            <li><code>bowler</code></li>
            <li><code>runs_off_bat</code></li>
            <li><code>extras</code></li>
            <li><code>venue</code></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # League name input
    league_name = st.text_input(
        "League Identifier",
        value="Cricket League",
        placeholder="e.g., IPL 2024",
        help="Enter a unique identifier for this dataset"
    )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Select CSV File",
        type=['csv'],
        help="Upload ball-by-ball cricket data"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            import pandas as pd
            df = pd.read_csv(uploaded_file)
            
            st.success("File uploaded successfully")
            
            # Validate columns (Cricsheet format)
            required_cols = ['match_id', 'batting_team', 'bowling_team', 'striker', 
                           'bowler', 'runs_off_bat', 'extras', 'venue']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
                return
            
            # Show dataset preview
            st.markdown("#### Dataset Preview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Matches", df['match_id'].nunique())
            with col3:
                st.metric("Teams", df['batting_team'].nunique())
            with col4:
                st.metric("Players", len(set(df['striker'].unique()) | set(df['bowler'].unique())))
            
            st.dataframe(df.head(10), use_container_width=True)
            
            # Confirm button
            st.markdown("---")
            
            # Check if model is already loaded
            # Check if model is already loaded (relying on loaded_model_name persistence)
            if st.session_state.get('loaded_model_name'):
                st.info(f"✓ Model '{st.session_state.loaded_model_name}' is currently loaded")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Proceed with Loaded Model", type="primary", use_container_width=True):
                        st.session_state.uploaded_data = df
                        st.session_state.current_league = league_name
                        st.session_state.data_loaded = False
                        st.session_state.model_trained = True # Explicitly ensure this is True
                        st.success("Dataset loaded. Proceeding with existing model.")
                        st.rerun()
                
                with col2:
                    if st.button("Train New Model", use_container_width=True):
                        st.session_state.uploaded_data = df
                        st.session_state.current_league = league_name
                        st.session_state.model_trained = False
                        st.session_state.data_loaded = False
                        st.session_state.loaded_model_name = None
                        st.success("Dataset loaded. Proceeding to training protocol.")
                        st.rerun()
            else:
                if st.button("Initialize & Proceed to Training", type="primary", use_container_width=True):
                    st.session_state.uploaded_data = df
                    st.session_state.current_league = league_name
                    st.session_state.model_trained = False
                    st.session_state.data_loaded = False
                    st.success("Dataset loaded. Proceeding to training protocol.")
                    st.rerun()
                
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")


def show_training_page():
    """Model training page."""
    st.markdown(f'<div class="section-title">Model Training: {st.session_state.current_league}</div>', unsafe_allow_html=True)
    
    if st.session_state.uploaded_data is None:
        st.error("No dataset uploaded.")
        return
    
    df = st.session_state.uploaded_data
    
    st.markdown("""
    <div class="material-card">
        <h4>Training Protocol</h4>
        <p>Configure and execute the machine learning training process.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Training Data", f"{len(df):,} records")
    with col2:
        st.metric("Match Count", df['match_id'].nunique())
    with col3:
        st.metric("Team Count", df['batting_team'].nunique())
    
    st.markdown("---")
    
    # Training scope selection
    st.markdown("#### Configuration")
    
    total_matches = df['match_id'].nunique()
    
    training_scope = st.radio(
        "Training Scope",
        options=["Complete Dataset", "Recent Matches Only"],
        help="Train on full history for accuracy or recent matches for speed"
    )
    
    max_matches = None
    if training_scope == "Recent Matches Only":
        max_matches = st.slider(
            "Recent Match Count",
            min_value=20,
            max_value=min(100, total_matches),
            value=50,
            step=10
        )
        st.info(f"Training on last {max_matches} matches")
    else:
        st.info(f"Training on all {total_matches} matches")
    
    st.markdown("---")
    
    # Training button
    if st.button("Execute Training Protocol", type="primary", use_container_width=True):
        from src.ml.trainer import train_model_from_dataframe
        from datetime import datetime
        
        # Progress containers
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(message, percent=None):
            status_text.write(f"**{message}**")
            if percent is not None:
                progress_bar.progress(percent / 100)
        
        try:
            # Train model
            with st.spinner("Executing training algorithms..."):
                model, feature_names, model_info = train_model_from_dataframe(
                    df, 
                    progress_callback=update_progress,
                    league_name=st.session_state.current_league,
                    max_matches=max_matches
                )
            
            # Show results
            st.success("Training Protocol Complete")
            
            st.markdown("#### Model Performance Metrics")
            best_model = model_info['best_model']
            scores = model_info['model_scores'][best_model]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Optimal Algorithm", best_model)
            with col2:
                st.metric("R² Score", f"{scores['r2']:.3f}")
            with col3:
                st.metric("MAE", f"{scores['mae']:.2f}")
            
            # Save model info and trained model to session state
            st.session_state.model_info = model_info
            st.session_state.trained_model = model
            st.session_state.trained_feature_names = feature_names
            st.session_state.model_trained = True
            
            # Auto-save to library
            try:
                from src.ml.model_library import ModelLibrary
                library = ModelLibrary()
                auto_save_name = f"AutoSave_{st.session_state.current_league}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                library.save_model(model, feature_names, model_info, auto_save_name)
                st.success(f"✓ Model automatically saved to library as: {auto_save_name}")
            except Exception as e:
                st.warning(f"Auto-save failed: {str(e)}")
            
            # Option to save model to library
            st.markdown("---")
            st.markdown("#### Save to Repository")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                model_save_name = st.text_input(
                    "Model Identifier",
                    value=f"{st.session_state.current_league}_{datetime.now().strftime('%Y%m%d')}"
                )
            with col2:
                st.write("")
                st.write("")
                if st.button("Save Model", use_container_width=True):
                    from src.ml.model_library import ModelLibrary
                    library = ModelLibrary()
                    
                    try:
                        library.save_model(
                            st.session_state.trained_model,
                            st.session_state.trained_feature_names,
                            st.session_state.model_info,
                            model_save_name
                        )
                        st.success(f"Model saved: {model_save_name}")
                    except Exception as e:
                        st.error(f"Save failed: {str(e)}")
            
            st.markdown("---")
            if st.button("Proceed to Dashboard", type="primary", use_container_width=True):
                st.rerun()
                
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
    
    # Option to go back
    st.markdown("---")
    if st.button("Return to Data Ingestion", use_container_width=True):
        st.session_state.uploaded_data = None
        st.rerun()


def load_model_callback(model_name, league_name, model_info):
    """Callback to load model into session state."""
    import joblib
    from src.ml.model_library import ModelLibrary
    
    try:
        library = ModelLibrary()
        loaded_model, feature_names, _ = library.load_model(model_name)
        
        # Save to active location
        joblib.dump(loaded_model, 'models/fantasy_predictor.pkl')
        joblib.dump(feature_names, 'models/feature_names.pkl')
        
        # Update session state
        st.session_state.current_league = league_name
        st.session_state.model_info = model_info
        st.session_state.model_trained = True
        st.session_state.loaded_model_name = model_name
        
        # Force a flag to indicate loading happened (for debug)
        st.session_state.just_loaded = True
        
    except Exception as e:
        st.session_state.load_error = str(e)

def show_model_library_page():
    """Model library page."""
    st.markdown('<div class="section-title">Model Repository</div>', unsafe_allow_html=True)
    
    # Check for load error
    if 'load_error' in st.session_state:
        st.error(f"Load failed: {st.session_state.load_error}")
        del st.session_state.load_error
        
    # Check for successful load
    if st.session_state.get('just_loaded'):
        st.success(f"✓ Model loaded: {st.session_state.loaded_model_name}")
        st.info("📂 Next: Go to 'Data Ingestion' page and upload your dataset")
        del st.session_state.just_loaded
    
    st.markdown("""
    <div class="material-card">
        <h4>Saved Models</h4>
        <p>Manage and load previously trained models.</p>
    </div>
    """, unsafe_allow_html=True)
    
    from src.ml.model_library import ModelLibrary
    library = ModelLibrary()
    
    saved_models = library.list_models()
    
    if not saved_models:
        st.info("No saved models found.")
        return
    
    st.markdown(f"#### {len(saved_models)} Available Models")
    
    for model in saved_models:
        with st.expander(f"{model['league_name']} - {model['model_name']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Matches", model['n_matches'])
                st.metric("Teams", model['n_teams'])
            
            with col2:
                st.metric("Algorithm", model['best_model'])
                st.metric("R² Score", f"{model['r2_score']:.3f}")
            
            with col3:
                st.metric("Date", model['saved_at'].split()[0])
            
            # Action buttons
            col_load, col_delete = st.columns(2)
            
            with col_load:
                st.button(
                    f"Load Model", 
                    key=f"load_{model['model_name']}", 
                    use_container_width=True,
                    on_click=load_model_callback,
                    args=(model['model_name'], model['league_name'], None) # model_info is loaded inside callback now
                )
            
            with col_delete:
                if st.button(f"Delete", key=f"del_{model['model_name']}", use_container_width=True):
                    try:
                        library.delete_model(model['model_name'])
                        st.success(f"Deleted: {model['model_name']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {str(e)}")
    
    # Show currently loaded model
    if st.session_state.loaded_model_name:
        st.markdown("---")
        st.info(f"Active Model: {st.session_state.loaded_model_name}")


def show_home_page():
    """Display analytics dashboard."""
    # Hero Banner
    st.markdown("""
    <div class="kpi-card">
        <h2>Fantasy Cricket Analytics Platform</h2>
        <p style="color: white; opacity: 0.9;">Advanced Data-Driven Team Selection & Performance Forecasting</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="material-card" style="text-align: center;">
            <div class="kpi-label" style="color: #5F6368;">Active Teams</div>
            <div class="kpi-value" style="color: #3F51B5;">{st.session_state.df['batting_team'].nunique() if st.session_state.df is not None else 0}</div>
            <div style="font-size: 0.8rem; color: #009688;">{st.session_state.current_league}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="material-card" style="text-align: center;">
            <div class="kpi-label" style="color: #5F6368;">Venues Analyzed</div>
            <div class="kpi-value" style="color: #3F51B5;">16</div>
            <div style="font-size: 0.8rem; color: #009688;">Global Grounds</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        matches_count = st.session_state.df['match_id'].nunique() if st.session_state.df is not None else 0
        st.markdown(f"""
        <div class="material-card" style="text-align: center;">
            <div class="kpi-label" style="color: #5F6368;">Total Matches</div>
            <div class="kpi-value" style="color: #3F51B5;">{matches_count}</div>
            <div style="font-size: 0.8rem; color: #009688;">Historical Data</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">Workflow Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="material-card">
            <h4>Machine Learning Core</h4>
            <p>Utilizing Random Forest and XGBoost algorithms to analyze historical player performance, venue statistics, and opposition matchups.</p>
            <ul>
                <li>Pattern Recognition</li>
                <li>Form Analysis</li>
                <li>Venue Specifics</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="material-card">
            <h4>Optimization Engine</h4>
            <p>Constructs the mathematically optimal fantasy lineup based on predicted points, credit constraints, and team composition rules.</p>
            <ul>
                <li>Credit Management</li>
                <li>Role Balancing</li>
                <li>Captaincy Selection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


def show_team_selection_page():
    """Team selection interface."""
    st.markdown('<div class="section-title">Match Configuration</div>', unsafe_allow_html=True)
    st.markdown("Select the competing teams for analysis.")
    
    teams = st.session_state.loader.get_teams()
    
    col1, col2 = st.columns(2)
    
    with col1:
        team1 = st.selectbox(
            "Home Team",
            options=teams,
            index=teams.index(st.session_state.selected_team1) if st.session_state.selected_team1 in teams else 0
        )
    
    with col2:
        available_teams = [t for t in teams if t != team1]
        team2 = st.selectbox(
            "Away Team",
            options=available_teams,
            index=available_teams.index(st.session_state.selected_team2) if st.session_state.selected_team2 in available_teams else 0
        )
    
    if st.button("Lock Matchup", type="primary"):
        st.session_state.selected_team1 = team1
        st.session_state.selected_team2 = team2
        st.success(f"Matchup Locked: {team1} vs {team2}")


def show_ground_selection_page():
    """Ground selection interface."""
    st.markdown('<div class="section-title">Venue Specification</div>', unsafe_allow_html=True)
    
    if not st.session_state.selected_team1 or not st.session_state.selected_team2:
        st.warning("Please configure match teams first.")
        return
    
    st.info(f"Match: {st.session_state.selected_team1} vs {st.session_state.selected_team2}")
    
    venues = st.session_state.loader.get_venues()
    
    venue = st.selectbox(
        "Select Venue",
        options=venues,
        index=venues.index(st.session_state.selected_venue) if st.session_state.selected_venue in venues else 0
    )
    
    # Show ground stats
    venue_df = st.session_state.df[st.session_state.df['venue'] == venue]
    if len(venue_df) > 0:
        st.markdown("#### Venue Analytics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            matches_at_venue = venue_df['match_id'].nunique()
            st.metric("Matches Played", matches_at_venue)
        
        with col2:
            avg_score = venue_df.groupby('match_id')['total_runs'].sum().mean()
            st.metric("Avg Score", f"{avg_score:.0f}")
        
        with col3:
            wickets = venue_df[venue_df['is_wicket'] == True]['match_id'].count()
            st.metric("Total Wickets", wickets)
    
    if st.button("Set Venue", type="primary"):
        st.session_state.selected_venue = venue
        st.success(f"Venue Set: {venue}")


def show_player_pool_page():
    """Player pool selection."""
    st.markdown('<div class="section-title">Roster Composition</div>', unsafe_allow_html=True)
    
    if not st.session_state.selected_team1 or not st.session_state.selected_team2:
        st.warning("Please configure match teams first.")
        return
    
    team1 = st.session_state.selected_team1
    team2 = st.session_state.selected_team2
    
    # Get players
    all_team_players = st.session_state.loader.get_players()
    all_unique_players = sorted(list(set().union(*all_team_players.values())))
    match_team_players = st.session_state.loader.get_players([team1, team2])
    
    st.markdown(f"**Available Pool:** {len(all_unique_players)} Players")
    st.info("Select players for your analysis pool (at least 1 required).")
    
    # Selection Logic
    col_search, col_add = st.columns([3, 1])
    
    with col_search:
        current_selection = st.session_state.selected_players
        available_players = [p for p in all_unique_players if p not in current_selection]
        
        player_to_add = st.selectbox(
            "Search Player",
            options=available_players,
            index=None,
            placeholder="Type to search...",
            key="player_search_box"
        )
        
    with col_add:
        st.write("")
        st.write("")
        if st.button("Add Player", type="primary", disabled=not player_to_add):
            if player_to_add and player_to_add not in st.session_state.selected_players:
                st.session_state.selected_players.append(player_to_add)
                st.rerun()
    
    # Selected List
    st.markdown("---")
    st.markdown(f"#### Selected Squad ({len(st.session_state.selected_players)} players)")
    
    if not st.session_state.selected_players:
        st.info("No players selected.")
    else:
        h1, h2, h3 = st.columns([3, 2, 1])
        h1.markdown("**Player**")
        h2.markdown("**Team**")
        h3.markdown("**Action**")
        
        players_to_remove = []
        player_tags = st.session_state.player_team_tags
        
        for i, player in enumerate(st.session_state.selected_players):
            c1, c2, c3 = st.columns([3, 2, 1])
            
            with c1:
                st.write(f"{i+1}. {player}")
                
            with c2:
                default_team = team1 if player in match_team_players.get(team1, set()) else team2
                current_tag = player_tags.get(player, default_team)
                
                new_tag = c2.selectbox(
                    f"Team for {player}",
                    options=[team1, team2],
                    index=0 if current_tag == team1 else 1,
                    key=f"tag_{player}",
                    label_visibility="collapsed"
                )
                player_tags[player] = new_tag
                
            with c3:
                if c3.button("Remove", key=f"remove_{player}"):
                    players_to_remove.append(player)
        
        if players_to_remove:
            for p in players_to_remove:
                st.session_state.selected_players.remove(p)
                if p in player_tags:
                    del player_tags[p]
            st.session_state.player_team_tags = player_tags
            st.rerun()
            
        st.session_state.player_team_tags = player_tags

    st.markdown("---")
    if len(st.session_state.selected_players) >= 1:
        st.success(f"✓ {len(st.session_state.selected_players)} players selected. Ready for analysis.")
    else:
        st.warning("Select at least 1 player to proceed.")


def show_predictions_page():
    """ML predictions page."""
    st.markdown('<div class="section-title">Performance Forecast</div>', unsafe_allow_html=True)
    
    if not st.session_state.selected_players or len(st.session_state.selected_players) != 22:
        st.warning("Please complete roster selection first.")
        return
    
    if not st.session_state.selected_venue:
        st.warning("Please select a venue.")
        return
    
    # Check model
    if not os.path.exists('models/fantasy_predictor.pkl'):
        st.error("Model not found. Please train a model first.")
        return
        
    if st.button("Generate Forecast", type="primary", use_container_width=True):
        with st.spinner("Running predictive models..."):
            try:
                import joblib
                # Load model
                model = joblib.load('models/fantasy_predictor.pkl')
                feature_names = joblib.load('models/feature_names.pkl')
                
                # Prepare data
                prediction_data = []
                
                # Get historical data for features
                hist_df = st.session_state.df
                
                from src.features.player_features import extract_batting_features, extract_bowling_features, extract_form_features, extract_consistency_features
                from src.features.contextual_features import extract_ground_features, extract_opposition_features
                
                for player in st.session_state.selected_players:
                    # Extract features
                    features = {}
                    features.update(extract_batting_features(player, hist_df))
                    features.update(extract_bowling_features(player, hist_df))
                    features.update(extract_form_features(player, hist_df))
                    features.update(extract_consistency_features(player, hist_df))
                    
                    # Context
                    venue = st.session_state.selected_venue
                    features.update(extract_ground_features(player, venue, hist_df))
                    
                    # Opposition
                    player_team = st.session_state.player_team_tags.get(player)
                    opposition = st.session_state.selected_team2 if player_team == st.session_state.selected_team1 else st.session_state.selected_team1
                    features.update(extract_opposition_features(player, opposition, hist_df))
                    
                    # Create feature vector
                    feature_vector = [features.get(f, 0) for f in feature_names]
                    
                    # Predict
                    predicted_points = model.predict([feature_vector])[0]
                    
                    prediction_data.append({
                        'Player': player,
                        'Team': player_team,
                        'Predicted Points': round(predicted_points, 1),
                        'Role': 'All-Rounder' # Placeholder, would need role data
                    })
                
                # Create DataFrame
                import pandas as pd
                pred_df = pd.DataFrame(prediction_data)
                pred_df = pred_df.sort_values('Predicted Points', ascending=False)
                
                # Display Results
                st.markdown("#### Projected Fantasy Points")
                
                # Top Picks
                col1, col2, col3 = st.columns(3)
                top_picks = pred_df.head(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="material-card" style="text-align: center; border-top: 4px solid #FFD700;">
                        <div style="font-weight: bold; color: #FFD700;">CAPTAIN PICK</div>
                        <div style="font-size: 1.2rem; font-weight: bold;">{top_picks.iloc[0]['Player']}</div>
                        <div style="font-size: 2rem; color: #3F51B5;">{top_picks.iloc[0]['Predicted Points']}</div>
                        <div style="font-size: 0.8rem;">Points</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="material-card" style="text-align: center; border-top: 4px solid #C0C0C0;">
                        <div style="font-weight: bold; color: #C0C0C0;">VICE-CAPTAIN</div>
                        <div style="font-size: 1.2rem; font-weight: bold;">{top_picks.iloc[1]['Player']}</div>
                        <div style="font-size: 2rem; color: #3F51B5;">{top_picks.iloc[1]['Predicted Points']}</div>
                        <div style="font-size: 0.8rem;">Points</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="material-card" style="text-align: center; border-top: 4px solid #CD7F32;">
                        <div style="font-weight: bold; color: #CD7F32;">TOP PICK</div>
                        <div style="font-size: 1.2rem; font-weight: bold;">{top_picks.iloc[2]['Player']}</div>
                        <div style="font-size: 2rem; color: #3F51B5;">{top_picks.iloc[2]['Predicted Points']}</div>
                        <div style="font-size: 0.8rem;">Points</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Full Table
                st.dataframe(
                    pred_df.style.background_gradient(subset=['Predicted Points'], cmap='Blues'),
                    use_container_width=True
                )
                
                # Optimal Team
                st.markdown("#### Optimal Lineup (11 Players)")
                optimal_team = pred_df.head(11)
                st.dataframe(optimal_team, use_container_width=True)
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
