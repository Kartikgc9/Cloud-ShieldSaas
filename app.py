from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
try:
    from tensorflow.keras.models import load_model
except ImportError:
    # For environments where TensorFlow isn't available
    def load_model(path):
        print(f"TensorFlow not available, mocking model load: {path}")
        return None
import json
import os
import requests
import sqlite3
from functools import wraps
import uuid
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature

# Initialize Flask app
app = Flask(__name__)
# app.config['SECRET_KEY'] = 'your-secret-key-change-this' # Removed as per edit hint
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///cloudburst_saas.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config.from_object('config.Config') # Load SECRET_KEY from config.py
mail = Mail(app)

# Password reset serializer
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    name = db.Column(db.String(100), nullable=False)
    api_key = db.Column(db.String(100), unique=True)
    subscription_type = db.Column(db.String(50), default='free')
    api_calls_count = db.Column(db.Integer, default=0)
    api_calls_limit = db.Column(db.Integer, default=100)  # Free tier limit
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    email_verified = db.Column(db.Boolean, default=False)
    role = db.Column(db.String(20), default='user')  # New field for RBAC

class PredictionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    location = db.Column(db.String(100))
    prediction_6h = db.Column(db.Float)
    prediction_12h = db.Column(db.Float)
    risk_level = db.Column(db.String(20))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SystemStats(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    total_predictions = db.Column(db.Integer, default=0)
    total_users = db.Column(db.Integer, default=0)
    accuracy_6h = db.Column(db.Float, default=97.35)
    accuracy_12h = db.Column(db.Float, default=98.23)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Cloudburst Prediction Service
class CloudburstPredictionService:
    def __init__(self):
        self.model_6h = None
        self.model_12h = None
        self.load_models()

    def load_models(self):
        """Load pre-trained models"""
        try:
            if os.path.exists('Model6.h5'):
                self.model_6h = load_model('Model6.h5')
                print("✓ 6-hour model loaded")
            if os.path.exists('Model12.h5'):
                self.model_12h = load_model('Model12.h5')
                print("✓ 12-hour model loaded")
        except Exception as e:
            print(f"Error loading models: {e}")

    def get_weather_data(self, location):
        """Fetch current weather data for location"""
        # This is a simplified version - you'd integrate with real weather APIs
        sample_data = {
            'location': location,
            'temperature': np.random.normal(25, 5),
            'humidity': np.random.normal(70, 15),
            'rainfall': np.random.exponential(2) if np.random.random() < 0.3 else 0,
            'wind_speed': np.random.normal(8, 3),
            'pressure': np.random.normal(1013, 5),
            'timestamp': datetime.now().isoformat()
        }
        return sample_data

    def prepare_prediction_data(self, weather_data):
        """Convert weather data to model input format"""
        # Simplified - in production you'd generate proper GAF images
        features = np.array([
            weather_data['rainfall'],
            weather_data['temperature'],
            weather_data['humidity'],
            weather_data['wind_speed'],
            weather_data['pressure'],
            weather_data['pressure']  # Using pressure for both pressure fields
        ])

        # Create mock GAF-like data for demonstration
        mock_gaf = np.random.random((1, 6, 256, 256)).astype(np.float32)
        return mock_gaf

    def predict_cloudburst(self, location):
        """Make cloudburst prediction for location"""
        try:
            # Get weather data
            weather_data = self.get_weather_data(location)

            # Prepare input data
            input_data = self.prepare_prediction_data(weather_data)

            # Make predictions
            pred_6h = 0.5  # Default values if models not loaded
            pred_12h = 0.5

            if self.model_6h is not None:
                pred_6h = float(self.model_6h.predict(input_data, verbose=0)[0][0])

            if self.model_12h is not None:
                pred_12h = float(self.model_12h.predict(input_data, verbose=0)[0][0])

            # Determine risk level (updated thresholds)
            max_risk = max(pred_6h, pred_12h)
            if max_risk >= 0.9:
                risk_level = "HIGH"
            elif max_risk >= 0.5:
                risk_level = "MODERATE"
            else:
                risk_level = "LOW"

            return {
                'location': location,
                'predictions': {
                    '6_hour': {
                        'probability': pred_6h,
                        'percentage': f"{pred_6h * 100:.1f}%"
                    },
                    '12_hour': {
                        'probability': pred_12h,
                        'percentage': f"{pred_12h * 100:.1f}%"
                    }
                },
                'risk_level': risk_level,
                'weather_data': weather_data,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {'error': str(e)}

# Initialize prediction service
prediction_service = CloudburstPredictionService()

# API Rate Limiting
def check_api_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get API key from request
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')

        if not api_key:
            return jsonify({'error': 'API key required'}), 401

        # Find user by API key
        user = User.query.filter_by(api_key=api_key).first()
        if not user:
            return jsonify({'error': 'Invalid API key'}), 401

        # Check rate limit
        if user.api_calls_count >= user.api_calls_limit:
            return jsonify({
                'error': 'API limit exceeded',
                'limit': user.api_calls_limit,
                'calls_made': user.api_calls_count
            }), 429

        # Increment API call count
        user.api_calls_count += 1
        db.session.commit()

        return f(user, *args, **kwargs)
    return decorated_function

# Routes
@app.route('/')
def index():
    """Homepage"""
    stats = SystemStats.query.first()
    if not stats:
        stats = SystemStats(
            total_predictions=0,
            total_users=0,
            accuracy_6h=97.35,
            accuracy_12h=98.23
        )
        db.session.add(stats)
        db.session.commit()

    return render_template('index.html', stats=stats)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        email = request.form['email']
        name = request.form['name']
        password = request.form['password']

        # Check if user exists
        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))

        # Create new user
        user = User(
            email=email,
            name=name,
            password_hash=generate_password_hash(password),
            api_key=str(uuid.uuid4())
        )

        db.session.add(user)
        db.session.commit()

        # Send verification email
        token = serializer.dumps(user.email, salt='email-verify-salt')
        verify_url = url_for('verify_email', token=token, _external=True)
        msg = Message('Verify Your Email', recipients=[user.email])
        msg.body = f'Welcome to CloudBurst Predict! Please verify your email by clicking the link: {verify_url}\nIf you did not register, please ignore this email.'
        mail.send(msg)
        flash('Registration successful! Please check your email to verify your account.')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password_hash, password):
            if not user.email_verified:
                flash('Please verify your email before logging in. Check your inbox.')
                return redirect(url_for('login'))
            # Send welcome email on first login
            if user.last_login is None:
                msg = Message('Welcome to CloudBurst Predict!', recipients=[user.email])
                msg.body = f'Hi {user.name},\n\nWelcome to CloudBurst Predict! We are excited to have you on board.\n\nYou can now access your dashboard and start using our cloudburst prediction services.\n\nIf you have any questions, feel free to reply to this email.\n\nBest regards,\nThe CloudBurst Predict Team'
                mail.send(msg)
            login_user(user)
            user.last_login = datetime.utcnow()
            db.session.commit()
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password')

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    # Get user's recent predictions
    recent_predictions = PredictionLog.query.filter_by(user_id=current_user.id)\
                                          .order_by(PredictionLog.created_at.desc())\
                                          .limit(10).all()

    return render_template('dashboard.html', 
                          user=current_user, 
                          predictions=recent_predictions)

@app.route('/api/predict', methods=['POST'])
@check_api_limit
def api_predict(user):
    """API endpoint for predictions"""
    data = request.get_json()

    if not data or 'location' not in data:
        return jsonify({'error': 'Location required'}), 400

    location = data['location']

    # Helper: check if user is premium or admin
    def is_premium(user):
        return user.subscription_type in ['premium', 'admin']

    if is_premium(user):
        # Real prediction for premium/admin users
        result = prediction_service.predict_cloudburst(location)
        result['real_prediction'] = True
    else:
        # Demo/simulated data for free users
        # Generate deterministic demo data for consistency
        import hashlib, random
        seed = int(hashlib.sha256(location.encode()).hexdigest(), 16) % (10 ** 8)
        random.seed(seed)
        pred6h = random.uniform(0.2, 0.8)
        pred12h = random.uniform(0.2, 0.8)
        max_risk = max(pred6h, pred12h)
        if max_risk >= 0.7:
            risk_level = "HIGH"
        elif max_risk >= 0.5:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        result = {
            'location': location,
            'predictions': {
                '6_hour': {
                    'probability': pred6h,
                    'percentage': f"{pred6h * 100:.1f}%"
                },
                '12_hour': {
                    'probability': pred12h,
                    'percentage': f"{pred12h * 100:.1f}%"
                }
            },
            'risk_level': risk_level,
            'timestamp': datetime.now().isoformat(),
            'real_prediction': False
        }

    if 'error' not in result and result.get('real_prediction'):
        # Log prediction only for real predictions
        log_entry = PredictionLog(
            user_id=user.id,
            location=location,
            prediction_6h=result['predictions']['6_hour']['probability'],
            prediction_12h=result['predictions']['12_hour']['probability'],
            risk_level=result['risk_level']
        )
        db.session.add(log_entry)
        # Update system stats
        stats = SystemStats.query.first()
        if stats:
            stats.total_predictions += 1
        db.session.commit()

    return jsonify(result)

@app.route('/api/locations')
@check_api_limit
def api_locations(user):
    """API endpoint to get supported locations"""
    locations = [
        'Dehradun, Uttarakhand',
        'Nainital, Uttarakhand', 
        'Haridwar, Uttarakhand',
        'Rishikesh, Uttarakhand',
        'Tehri, Uttarakhand',
        'Chamoli, Uttarakhand',
        'Pithoragarh, Uttarakhand'
    ]

    return jsonify({
        'locations': locations,
        'total': len(locations)
    })

@app.route('/api/stats')
@check_api_limit
def api_stats(user):
    """API endpoint for system statistics"""
    stats = SystemStats.query.first()

    return jsonify({
        'total_predictions': stats.total_predictions if stats else 0,
        'total_users': User.query.count(),
        'accuracy_6h': stats.accuracy_6h if stats else 97.35,
        'accuracy_12h': stats.accuracy_12h if stats else 98.23,
        'your_api_calls': user.api_calls_count,
        'your_api_limit': user.api_calls_limit
    })

@app.route('/pricing')
def pricing():
    """Pricing page"""
    return render_template('pricing.html')

@app.route('/admin')
@login_required
def admin():
    """Admin dashboard"""
    if current_user.role != 'admin':
        return redirect(url_for('dashboard'))

    users = User.query.all()
    predictions = PredictionLog.query.order_by(PredictionLog.created_at.desc()).limit(50).all()

    return render_template('admin.html', users=users, predictions=predictions)

@app.route('/reset_password_request', methods=['GET', 'POST'])
def reset_password_request():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        if user:
            token = serializer.dumps(user.email, salt='password-reset-salt')
            reset_url = url_for('reset_password', token=token, _external=True)
            msg = Message('Password Reset Request', recipients=[user.email])
            msg.body = f'Click the link to reset your password: {reset_url}\nIf you did not request this, please ignore this email.'
            mail.send(msg)
            flash('A password reset link has been sent to your email.')
            return redirect(url_for('login'))
        else:
            flash('No account found with that email.')
    return render_template('reset_password_request.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = serializer.loads(token, salt='password-reset-salt', max_age=3600)
    except (SignatureExpired, BadSignature):
        flash('The password reset link is invalid or has expired.')
        return redirect(url_for('reset_password_request'))
    user = User.query.filter_by(email=email).first()
    if not user:
        flash('Invalid user.')
        return redirect(url_for('reset_password_request'))
    if request.method == 'POST':
        password = request.form['password']
        user.password_hash = generate_password_hash(password)
        db.session.commit()
        flash('Your password has been updated. Please log in.')
        return redirect(url_for('login'))
    return render_template('reset_password.html')

# Add verify_email route
@app.route('/verify_email/<token>')
def verify_email(token):
    try:
        email = serializer.loads(token, salt='email-verify-salt', max_age=86400)
    except (SignatureExpired, BadSignature):
        flash('The verification link is invalid or has expired.')
        return redirect(url_for('login'))
    user = User.query.filter_by(email=email).first()
    if not user:
        flash('Invalid user.')
        return redirect(url_for('login'))
    if user.email_verified:
        flash('Email already verified. Please log in.')
        return redirect(url_for('login'))
    user.email_verified = True
    db.session.commit()
    flash('Your email has been verified! You can now log in.')
    return redirect(url_for('login'))

# Initialize database when app starts
def init_db():
    """Initialize database and create admin user if needed"""
    with app.app_context():
        db.create_all()

        # Create admin user if not exists
        admin = User.query.filter_by(email='admin@cloudburstpredict.com').first()
        if not admin:
            admin = User(
                email='admin@cloudburstpredict.com',
                name='Admin',
                password_hash=generate_password_hash('admin123'),
                api_key=str(uuid.uuid4()),
                subscription_type='admin',
                api_calls_limit=10000,
                role='admin' # Ensure role is set
            )
            db.session.add(admin)
            db.session.commit()
            print(f"Admin user created with API key: {admin.api_key}")

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)