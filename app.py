import os
import pickle
import signal
import sys
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)

# Global variables
models = {}
team_stats = {}
MODEL_READY = False

# ==================== DATA SOURCES ====================

URLS = {
    'premier': [
        'https://www.football-data.co.uk/mmz4281/2425/E0.csv',
        'https://www.football-data.co.uk/mmz4281/2324/E0.csv',
        'https://www.football-data.co.uk/mmz4281/2223/E0.csv',
        'https://www.football-data.co.uk/mmz4281/2122/E0.csv',
        'https://www.football-data.co.uk/mmz4281/2021/E0.csv'
    ],
    'championship': [
        'https://www.football-data.co.uk/mmz4281/2425/E1.csv',
        'https://www.football-data.co.uk/mmz4281/2324/E1.csv',
        'https://www.football-data.co.uk/mmz4281/2223/E1.csv',
        'https://www.football-data.co.uk/mmz4281/2122/E1.csv',
        'https://www.football-data.co.uk/mmz4281/2021/E1.csv'
    ]
}

# ==================== DATA PROCESSING ====================

def load_data(urls):
    """Load and combine data from multiple URLs"""
    all_data = []
    
    for league, league_urls in urls.items():
        for url in league_urls:
            try:
                df = pd.read_csv(url)
                df['League'] = league  # Add league identifier
                all_data.append(df)
                print(f"Loaded data from {url}")
            except Exception as e:
                print(f"Error loading {url}: {str(e)}")
                continue
    
    if not all_data:
        raise ValueError("No data could be loaded from any source")
    
    return pd.concat(all_data, ignore_index=True)

def preprocess_data(df):
    """Preprocess the raw data with proper pandas operations"""
    df = df.copy()
    
    # Select and validate essential columns
    essential_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
    available_cols = [col for col in essential_cols if col in df.columns]
    df = df.loc[:, available_cols + ['League']]
    
    # Fill numeric columns safely
    numeric_cols = df.select_dtypes(include=np.number).columns
    df.loc[:, numeric_cols] = df[numeric_cols].fillna(0)
    
    # Convert categorical columns safely
    categorical_cols = ['HomeTeam', 'AwayTeam', 'FTR', 'League']
    for col in categorical_cols:
        if col in df.columns:
            df.loc[:, col] = pd.Categorical(df[col])
    
    return df

def calculate_team_form(df, team_col, result_col, window=5):
    """Calculate team form over last N matches"""
    form = (df[result_col] == 'H').rolling(window).mean() if team_col == 'HomeTeam' else \
           (df[result_col] == 'A').rolling(window).mean()
    return form

def engineer_features(df):
    """Create enhanced features for modeling"""
    df = df.copy()
    
    # Add basic match features
    df['GoalDifference'] = df['FTHG'] - df['FTAG']
    df['TotalShots'] = df['HS'] + df['AS']
    df['ShotAccuracy'] = (df['HST'] + df['AST']) / (df['HS'] + df['AS'] + 1e-6)
    
    # Calculate team forms
    teams = pd.unique(pd.concat([df['HomeTeam'], df['AwayTeam']]))
    team_stats = {}
    
    for team in teams:
        # Home stats
        home_matches = df[df['HomeTeam'] == team].sort_values('Date')
        home_matches['HomeForm'] = calculate_team_form(home_matches, 'HomeTeam', 'FTR')
        
        # Away stats
        away_matches = df[df['AwayTeam'] == team].sort_values('Date')
        away_matches['AwayForm'] = calculate_team_form(away_matches, 'AwayTeam', 'FTR')
        
        # Store rolling averages
        team_stats[team] = {
            'home_goals_scored': home_matches['FTHG'].mean(),
            'home_goals_conceded': home_matches['FTAG'].mean(),
            'home_shots': home_matches['HS'].mean(),
            'home_form': home_matches['HomeForm'].iloc[-1] if len(home_matches) > 0 else 0.5,
            
            'away_goals_scored': away_matches['FTAG'].mean(),
            'away_goals_conceded': away_matches['FTHG'].mean(),
            'away_shots': away_matches['AS'].mean(),
            'away_form': away_matches['AwayForm'].iloc[-1] if len(away_matches) > 0 else 0.5,
        }
    
    # Create match features
    features = []
    labels = []
    
    for idx, match in df.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        if home_team not in team_stats or away_team not in team_stats:
            continue
            
        home_stats = team_stats[home_team]
        away_stats = team_stats[away_team]
        
        # Enhanced feature vector
        feature = [
            home_stats['home_goals_scored'],
            home_stats['home_goals_conceded'],
            home_stats['home_shots'],
            home_stats['home_form'],
            
            away_stats['away_goals_scored'],
            away_stats['away_goals_conceded'],
            away_stats['away_shots'],
            away_stats['away_form'],
            
            # Difference features
            home_stats['home_goals_scored'] - away_stats['away_goals_conceded'],
            away_stats['away_goals_scored'] - home_stats['home_goals_conceded'],
            
            # League indicator
            1 if match['League'] == 'premier' else 0
        ]
        
        features.append(feature)
        labels.append(match['FTR'])
    
    return np.array(features), np.array(labels), team_stats

# ==================== MODEL TRAINING ====================

def train_random_forest(X, y):
    """Train optimized Random Forest model"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature processing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
    selector.fit(X_train_scaled, y_train)
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Train model with balanced class weights
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_selected, y_train)
    
    accuracy = model.score(X_test_selected, y_test)
    print(f"Random Forest Accuracy: {accuracy:.4f}")
    
    # Calculate calibration data
    calibration_data = calculate_calibration(model, X_test_selected, y_test)
    
    return {
        'model': model,
        'scaler': scaler,
        'selector': selector,
        'accuracy': accuracy,
        'calibration': calibration_data
    }

def train_neural_network(X, y):
    """Train optimized Neural Network"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Feature processing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
    selector.fit(X_train_scaled, y_train_encoded)
    X_train_selected = selector.transform(X_train_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Build model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_selected.shape[1],)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train with early stopping
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    history = model.fit(
        X_train_selected, y_train_encoded,
        validation_data=(X_test_selected, y_test_encoded),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate
    _, accuracy = model.evaluate(X_test_selected, y_test_encoded, verbose=0)
    print(f"Neural Network Accuracy: {accuracy:.4f}")
    
    # Calculate calibration data
    calibration_data = calculate_calibration(model, X_test_selected, y_test, is_nn=True, label_encoder=label_encoder)
    
    return {
        'model': model,
        'scaler': scaler,
        'selector': selector,
        'label_encoder': label_encoder,
        'accuracy': accuracy,
        'calibration': calibration_data
    }

def calculate_calibration(model, X, y_true, is_nn=False, label_encoder=None):
    """Calculate model calibration by binning predictions and comparing to actual outcomes"""
    # Make predictions
    if is_nn:
        y_pred_probs = model.predict(X, verbose=0)
        if label_encoder:
            y_pred = label_encoder.inverse_transform(np.argmax(y_pred_probs, axis=1))
    else:
        y_pred_probs = model.predict_proba(X)
        y_pred = model.classes_[np.argmax(y_pred_probs, axis=1)]
    
    # For each class, create calibration data
    calibration_data = {}
    
    if is_nn and label_encoder:
        classes = label_encoder.classes_
    else:
        classes = model.classes_
    
    for i, cls in enumerate(classes):
        # Get probabilities for this class
        prob_true = y_pred_probs[:, i]
        
        # Bin probabilities
        bins = np.linspace(0, 1, 11)  # 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
        bin_indices = np.digitize(prob_true, bins) - 1
        
        # Calculate actual fraction of positives in each bin
        actual_probs = []
        bin_counts = []
        y_true_cls = (y_true == cls)
        
        for bin_idx in range(len(bins) - 1):
            mask = (bin_indices == bin_idx)
            if np.sum(mask) > 0:
                actual_prob = np.mean(y_true_cls[mask])
                actual_probs.append(actual_prob)
                bin_counts.append(np.sum(mask))
            else:
                actual_probs.append(np.nan)
                bin_counts.append(0)
        
        # Store calibration data for this class
        calibration_data[cls] = {
            'bins': bins[:-1] + 0.05,  # Use bin centers
            'predicted': bins[:-1] + 0.05,
            'actual': actual_probs,
            'counts': bin_counts
        }
    
    return calibration_data

# ==================== PREDICTION HANDLING ====================

def predict_match(home_team, away_team, model_type='rf'):
    """Make prediction with proper feature alignment"""
    global models, team_stats
    
    if home_team not in team_stats or away_team not in team_stats:
        return None
    
    home_stats = team_stats[home_team]
    away_stats = team_stats[away_team]
    
    # Feature vector must match training exactly
    features = [
        home_stats['home_goals_scored'],
        home_stats['home_goals_conceded'],
        home_stats['home_shots'],
        home_stats['home_form'],
        
        away_stats['away_goals_scored'],
        away_stats['away_goals_conceded'],
        away_stats['away_shots'],
        away_stats['away_form'],
        
        home_stats['home_goals_scored'] - away_stats['away_goals_conceded'],
        away_stats['away_goals_scored'] - home_stats['home_goals_conceded'],
        
        1  # Default to Premier League for prediction
    ]
    
    features = np.array(features).reshape(1, -1)
    model_info = models[model_type]
    
    try:
        # Apply preprocessing
        features_scaled = model_info['scaler'].transform(features)
        features_selected = model_info['selector'].transform(features_scaled)
        
        # Make prediction
        if model_type == 'rf':
            prediction = model_info['model'].predict(features_selected)[0]
            probabilities = model_info['model'].predict_proba(features_selected)[0]
            classes = model_info['model'].classes_
        else:  # nn
            probabilities = model_info['model'].predict(features_selected, verbose=0)[0]
            prediction_idx = np.argmax(probabilities)
            prediction = model_info['label_encoder'].inverse_transform([prediction_idx])[0]
            classes = model_info['label_encoder'].classes_
        
        # Format results
        outcome_mapping = {'H': f"{home_team} win", 'D': "Draw", 'A': f"{away_team} win"}
        
        # Get reliability information for each outcome
        reliability = {}
        calibration = model_info['calibration']
        
        for cls, prob in zip(classes, probabilities):
            if cls in calibration:
                # Find the closest bin
                bin_idx = np.digitize(prob, calibration[cls]['bins']) - 1
                if 0 <= bin_idx < len(calibration[cls]['actual']):
                    actual_prob = calibration[cls]['actual'][bin_idx]
                    if not np.isnan(actual_prob):
                        reliability[outcome_mapping[cls]] = {
                            'predicted_prob': prob,
                            'actual_prob': actual_prob,
                            'reliability': f"{actual_prob:.1%}",
                            'count': calibration[cls]['counts'][bin_idx]
                        }
        
        return {
            'prediction': outcome_mapping[prediction],
            'probabilities': {outcome_mapping[cls]: f"{prob:.1%}" 
                             for cls, prob in zip(classes, probabilities)},
            'model_type': 'Random Forest' if model_type == 'rf' else 'Neural Network',
            'reliability': reliability
        }
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None

# ==================== FLASK ROUTES ====================

@app.route('/', methods=['GET', 'POST'])
def index():
    global MODEL_READY
    
    if not MODEL_READY:
        initialize_application()
    
    if request.method == 'POST':
        home_team = request.form.get('home_team')
        away_team = request.form.get('away_team')
        model_type = request.form.get('model_type', 'rf')
        
        result = predict_match(home_team, away_team, model_type)
        
        if result:
            return render_template('index.html',
                                teams=sorted(list(team_stats.keys())),
                                result=result,
                                home_team=home_team,
                                away_team=away_team,
                                model_type=model_type)
    
    return render_template('index.html',
                         teams=sorted(list(team_stats.keys())))

@app.route('/predict', methods=['POST'])
def predict():
    global MODEL_READY
    
    if not MODEL_READY:
        initialize_application()
    
    data = request.get_json()
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    model_type = data.get('model_type', 'rf')
    
    if not home_team or not away_team:
        return jsonify({'error': 'Missing team names'}), 400
    
    result = predict_match(home_team, away_team, model_type)
    
    if not result:
        return jsonify({'error': 'Prediction failed - check team names'}), 400
    
    return jsonify(result)

# ==================== APPLICATION SETUP ====================

def initialize_application():
    global models, team_stats, MODEL_READY
    
    print("Initializing application...")
    
    try:
        # Try to load pre-trained models first
        with open('model_data.pkl', 'rb') as f:
            data = pickle.load(f)
            models = data['models']
            team_stats = data['team_stats']
        print("Loaded pre-trained models")
        MODEL_READY = True
        return
    except Exception as e:
        print(f"Could not load pre-trained models: {str(e)}")
    
    # Fall back to training new models
    print("Loading data...")
    df = load_data(URLS)
    
    print("Preprocessing data...")
    df = preprocess_data(df)
    
    print("Engineering features...")
    X, y, team_stats = engineer_features(df)
    
    print("Training models...")
    models = {
        'rf': train_random_forest(X, y),
        'nn': train_neural_network(X, y)
    }
    
    # Save models
    with open('model_data.pkl', 'wb') as f:
        pickle.dump({'models': models, 'team_stats': team_stats}, f)
    print("Models trained and saved")
    
    MODEL_READY = True

# Graceful shutdown handler
def handle_shutdown(signum, frame):
    print("\nServer shutting down gracefully...")
    sys.exit(0)

# ==================== MAIN EXECUTION ====================

if __name__ == '__main__':
    # Setup graceful shutdown
    signal.signal(signal.SIGINT, handle_shutdown)
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create basic HTML template if it doesn't exist
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Football Predictor Pro</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .form-container { background: #f5f5f5; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        select, button { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #4CAF50; color: white; cursor: pointer; }
        .result { margin-top: 20px; padding: 15px; background: #e8f5e9; border-radius: 4px; }
        .prob-bar { height: 20px; background: #ddd; margin-top: 5px; border-radius: 3px; overflow: hidden; }
        .prob-fill { height: 100%; background: #4CAF50; width: 0%; transition: width 0.5s; }
        .reliability { margin-top: 15px; padding: 10px; background: #e3f2fd; border-radius: 4px; }
        .reliability-item { margin-bottom: 8px; }
        .reliability-label { font-weight: bold; }
        .reliability-value { color: #1976d2; }
        .reliability-note { font-size: 0.9em; color: #666; margin-top: 5px; }
    </style>
</head>
<body>
    <h1>Football Match Predictor</h1>
    <div class="form-container">
        <form method="POST">
            <div class="form-group">
                <label for="home_team">Home Team:</label>
                <select id="home_team" name="home_team" required>
                    <option value="">Select Home Team</option>
                    {% for team in teams %}
                        <option value="{{ team }}" {% if home_team == team %}selected{% endif %}>{{ team }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="form-group">
                <label for="away_team">Away Team:</label>
                <select id="away_team" name="away_team" required>
                    <option value="">Select Away Team</option>
                    {% for team in teams %}
                        <option value="{{ team }}" {% if away_team == team %}selected{% endif %}>{{ team }}</option>
                    {% endfor %}
                </select>
            </div>
            <div class="form-group">
                <label for="model_type">Model Type:</label>
                <select id="model_type" name="model_type">
                    <option value="rf" {% if model_type == 'rf' %}selected{% endif %}>Random Forest</option>
                    <option value="nn" {% if model_type == 'nn' %}selected{% endif %}>Neural Network</option>
                </select>
            </div>
            <button type="submit">Predict Outcome</button>
        </form>
        
        {% if result %}
        <div class="result">
            <h2>{{ home_team }} vs {{ away_team }}</h2>
            <p><strong>Prediction:</strong> {{ result.prediction }}</p>
            <p><strong>Model:</strong> {{ result.model_type }}</p>
            <h3>Probabilities:</h3>
            {% for outcome, prob in result.probabilities.items() %}
                <div>
                    <span>{{ outcome }}: {{ prob }}</span>
                    <div class="prob-bar">
                        <div class="prob-fill" style="width: {{ prob }}"></div>
                    </div>
                </div>
            {% endfor %}
            
            <div class="reliability">
                <h3>Model Reliability Analysis</h3>
                <p class="reliability-note">Based on historical predictions with similar confidence levels:</p>
                {% for outcome, data in result.reliability.items() %}
                    <div class="reliability-item">
                        <span class="reliability-label">{{ outcome }}:</span>
                        <span class="reliability-value">
                            Model predicted {{ "%.1f"|format(data.predicted_prob * 100) }}%, 
                            actual outcome was {{ data.reliability }} (based on {{ data.count }} similar predictions)
                        </span>
                    </div>
                {% endfor %}
                <p class="reliability-note">
                    A well-calibrated model would have predicted probabilities matching actual outcomes.
                    Differences indicate areas where the model may be overconfident or underconfident.
                </p>
            </div>
        </div>
        {% endif %}
    </div>
</body>
</html>
''')
    
    # Initialize application
    initialize_application()
    
    # Run app
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
