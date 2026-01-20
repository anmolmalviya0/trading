"""
V8 FINAL - AI META-MODEL
==========================
Machine Learning layer on top of signals.
Uses LightGBM to filter signals based on:
- Historical win/loss patterns
- Feature combinations that predict success

This is the AI brain that learns from past trades.
"""
import pandas as pd
import numpy as np
import yaml
import os
from typing import Dict, Optional

# Check if LightGBM is available
try:
    from lightgbm import LGBMClassifier
    import joblib
    HAS_ML = True
except ImportError:
    HAS_ML = False


def load_config():
    try:
        with open('config.yaml') as f:
            return yaml.safe_load(f)
    except:
        return {}


class AIMetaModel:
    """
    AI-powered signal filter.
    
    Uses machine learning to predict which signals are more likely to win.
    The model is trained on historical signal outcomes and learns patterns
    that simple rules cannot capture.
    """
    
    def __init__(self, model_path: str = 'ai_model.joblib'):
        self.model_path = model_path
        self.model = None
        self.config = load_config()
        self.threshold = self.config.get('ai', {}).get('confidence_threshold', 0.55)
        
        if HAS_ML and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"🧠 AI Model loaded: {model_path}")
    
    def extract_features(self, signal: Dict) -> np.ndarray:
        """Extract features from signal for AI prediction"""
        features = [
            signal.get('score', 0) / 100,
            signal.get('rsi', 50) / 100,
            signal.get('adx', 20) / 100,
            signal.get('num_signals', 0) / 5,
            1.0 if signal.get('direction') == 'BUY' else 0.0,
        ]
        return np.array(features).reshape(1, -1)
    
    def predict(self, signal: Dict) -> Dict:
        """
        Use AI to predict win probability.
        
        Returns enhanced signal with:
        - ai_confidence: probability of winning (0-1)
        - ai_approved: whether AI recommends the trade
        """
        if not HAS_ML or self.model is None:
            # No AI model - pass through
            signal['ai_enabled'] = False
            signal['ai_confidence'] = None
            signal['ai_approved'] = True  # Default approve if no AI
            return signal
        
        features = self.extract_features(signal)
        
        try:
            # Get probability of winning (class 1)
            proba = self.model.predict_proba(features)[0]
            confidence = proba[1] if len(proba) > 1 else proba[0]
            
            signal['ai_enabled'] = True
            signal['ai_confidence'] = float(confidence)
            signal['ai_approved'] = confidence >= self.threshold
            
            return signal
        except Exception as e:
            signal['ai_enabled'] = False
            signal['ai_confidence'] = None
            signal['ai_approved'] = True
            return signal
    
    @staticmethod
    def train(candidates_path: str, output_path: str = 'ai_model.joblib'):
        """
        Train AI model on historical signals.
        
        candidates_path: CSV with columns [score, rsi, adx, num_signals, direction, outcome]
        """
        if not HAS_ML:
            print("❌ LightGBM not installed. Run: pip install lightgbm")
            return
        
        df = pd.read_csv(candidates_path)
        
        # Features
        X = df[['score', 'rsi', 'adx', 'num_signals']].values
        X = np.column_stack([
            X[:, 0] / 100,  # score
            X[:, 1] / 100,  # rsi
            X[:, 2] / 100,  # adx
            X[:, 3] / 5,    # num_signals
            (df['direction'] == 'BUY').astype(float).values
        ])
        
        # Target
        y = df['outcome'].values  # 1 = win, 0 = loss
        
        # Train
        model = LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X, y)
        
        # Save
        joblib.dump(model, output_path)
        print(f"✅ AI Model trained and saved: {output_path}")
        
        # Show feature importance
        importance = model.feature_importances_
        features = ['score', 'rsi', 'adx', 'num_signals', 'direction']
        print("\n📊 Feature Importance:")
        for f, i in sorted(zip(features, importance), key=lambda x: -x[1]):
            print(f"  {f}: {i:.3f}")
