from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import torch
from datetime import datetime, timedelta, timezone
import re
import torchvision.transforms as transforms
from PIL import Image
import io
import torch.nn as nn
import numpy as np
import cv2
import base64
from scipy.spatial.distance import cdist
import pyrebase
import os
from functools import wraps
import json
from firebase_admin import credentials, db as admin_db, auth as admin_auth
import firebase_admin
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', os.urandom(24))
app.permanent_session_lifetime = timedelta(days=5)

# Firebase Configuration
firebase_config = {
    "apiKey": os.getenv("FIREBASE_API_KEY"),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
    "databaseURL": os.getenv("FIREBASE_DATABASE_URL"),
    "projectId": os.getenv("FIREBASE_PROJECT_ID"),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
    "appId": os.getenv("FIREBASE_APP_ID"),
    "measurementId": os.getenv("FIREBASE_MEASUREMENT_ID")
}

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': os.getenv('FIREBASE_DATABASE_URL')
    })
    logger.info("Firebase Admin SDK initialized successfully")

# Initialize Pyrebase
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()
db = firebase.database()

def save_user_data(user_id, email, username):
    """Save user data with improved error handling"""
    try:
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Saving user data for {user_id}")

        # Basic user data
        user_data = {
            "email": email,
            "username": username,
            "created_at": current_time,
            "last_login": "",
            "login_attempts": 0,
            "account_locked": False,
            "profile_created": current_time
        }

        # Save to Firebase
        ref = admin_db.reference(f'/users/{user_id}')
        ref.set(user_data)

        # Verify save
        saved_data = ref.get()
        if saved_data:
            logger.info(f"User data saved successfully for {user_id}")
            return True

        logger.error(f"Failed to verify saved data for {user_id}")
        return False

    except Exception as e:
        logger.error(f"Error saving user data: {str(e)}")
        return False

def get_user_data(user_id):
    """Get user data with proper error handling"""
    try:
        ref = admin_db.reference(f'/users/{user_id}')
        user_data = ref.get()
        if user_data:
            logger.info(f"Retrieved user data for {user_id}")
            return user_data
        logger.warning(f"No data found for user {user_id}")
        return None
    except Exception as e:
        logger.error(f"Error getting user data: {str(e)}")
        return None

def update_user_login(user_id, success=True):
    """Update user login status"""
    try:
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        ref = admin_db.reference(f'/users/{user_id}')
        
        if success:
            ref.update({
                "last_login": current_time,
                "login_attempts": 0,
                "account_locked": False
            })
        else:
            user_data = get_user_data(user_id)
            if user_data:
                attempts = user_data.get('login_attempts', 0) + 1
                ref.update({
                    "login_attempts": attempts,
                    "account_locked": attempts >= 5
                })
                return attempts >= 5
        return False
    except Exception as e:
        logger.error(f"Error updating user login: {str(e)}")
        return False

# Login decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

@app.route("/signup", methods=["GET", "POST"])
def signup():
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    if request.method == "POST":
        try:
            username = request.form.get("username")
            email = request.form.get("email")
            password = request.form.get("password")
            
            # Create user in Firebase Authentication
            user = auth.create_user_with_email_and_password(email, password)
            user_id = user['localId']
            
            # Save user data
            if save_user_data(user_id, email, username):
                # Send verification email
                auth.send_email_verification(user['idToken'])
                
                return render_template("login.html",
                                    message="Account created successfully! Please verify your email before logging in.",
                                    current_time=current_time)
            else:
                # Clean up if data save fails
                try:
                    admin_auth.delete_user(user_id)
                except Exception as e:
                    logger.error(f"Error cleaning up user: {str(e)}")
                
                return render_template("signup.html",
                                    error="Error saving user data. Please try again.",
                                    current_time=current_time)
                
        except Exception as e:
            error_message = str(e)
            logger.error(f"Signup error: {error_message}")
            
            if "EMAIL_EXISTS" in error_message:
                error = "Email already exists."
            else:
                error = f"Error creating account: {error_message}"
            
            return render_template("signup.html",
                                error=error,
                                current_time=current_time)
    
    return render_template("signup.html", current_time=current_time)

@app.route("/login", methods=["GET", "POST"])
def login():
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    if request.method == "POST":
        try:
            email = request.form.get("email")
            password = request.form.get("password")
            
            # Input validation
            if not email or not password:
                return render_template("login.html",
                                    error="Please enter both email and password.",
                                    email=email,
                                    current_time=current_time)
            
            # Try to authenticate with Firebase
            try:
                user = auth.sign_in_with_email_and_password(email, password)
            except Exception as auth_error:
                error_message = str(auth_error)
                logger.error(f"Authentication error: {error_message}")
                
                if "INVALID_LOGIN_CREDENTIALS" in error_message:
                    return render_template("login.html",
                                        error="Invalid email or password. Please try again.",
                                        email=email,
                                        current_time=current_time)
                elif "INVALID_EMAIL" in error_message:
                    return render_template("login.html",
                                        error="Please enter a valid email address.",
                                        email=email,
                                        current_time=current_time)
                else:
                    return render_template("login.html",
                                        error="Login failed. Please try again.",
                                        email=email,
                                        current_time=current_time)
            
            # Check email verification
            account_info = auth.get_account_info(user['idToken'])
            email_verified = account_info['users'][0]['emailVerified']
            uid = account_info['users'][0]['localId']
            
            if not email_verified:
                # Send verification email
                auth.send_email_verification(user['idToken'])
                return render_template("login.html",
                                    error="Please verify your email before logging in. A new verification email has been sent.",
                                    show_resend=True,
                                    email=email,
                                    current_time=current_time)
            
            # Get user data
            user_data = get_user_data(uid)
            if not user_data:
                logger.error(f"No user data found for UID: {uid}")
                return render_template("login.html",
                                    error="Error retrieving user data. Please try again.",
                                    email=email,
                                    current_time=current_time)
            
            # Update login status
            update_user_login(uid, True)
            
            # Set session data
            session.permanent = True
            session['user_id'] = uid
            session['email'] = email
            session['username'] = user_data.get('username', '')
            session['token'] = user['idToken']
            
            logger.info(f"Successful login for user: {email}")
            # Redirect to home page instead of dashboard
            return redirect(url_for('home'))
            
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return render_template("login.html",
                                error="An error occurred during login. Please try again.",
                                email=email,
                                current_time=current_time)
    
    return render_template("login.html", current_time=current_time)


@app.route("/resend-verification", methods=["POST"])
def resend_verification():
    try:
        data = request.get_json()
        email = data.get('email')
        
        if not email:
            return jsonify({"success": False, "error": "Email is required"}), 400
        
        # Get user by email and send verification
        user = auth.sign_in_with_email_and_password(email, request.form.get('password', ''))
        auth.send_email_verification(user['idToken'])
        
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error resending verification: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500



@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route("/")
def home():
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return render_template("index.html",
                         logged_in='user_id' in session,
                         username=session.get('username'),
                         current_time=current_time)

@app.route("/about-us")
def aboutUs():
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return render_template("about.html", 
                         logged_in='user_id' in session,
                         username=session.get('username'),
                         current_time=current_time)

@app.route("/forgery-detection")
@login_required
def forgery_detection_page():
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return render_template("forgery_detection.html", 
                         logged_in=True,
                         username=session.get('username'),
                         current_time=current_time)

@app.route("/fake-detection")
def fake_detection_page():
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return render_template("fake_detection.html", 
                         logged_in='user_id' in session,
                         username=session.get('username'),
                         current_time=current_time)

@app.route("/analysis-history")
@login_required
def analysis_history():
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    results = get_user_analysis_history(session['user_id'])
    return render_template("analysis_history.html", 
                         results=results,
                         logged_in=True,
                         username=session.get('username'),
                         current_time=current_time)

@app.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    if request.method == "POST":
        try:
            email = request.form.get("email")
            if not email:
                return render_template("reset_password.html",
                                    error="Please enter your email address.",
                                    current_time=current_time)
            
            # Send password reset email through Firebase
            auth.send_password_reset_email(email)
            
            return render_template("login.html",
                                message="Password reset email sent! Please check your inbox.",
                                current_time=current_time)
                                
        except Exception as e:
            logger.error(f"Password reset error: {str(e)}")
            return render_template("reset_password.html",
                                error="Error sending reset email. Please try again.",
                                current_time=current_time)
    
    return render_template("reset_password.html",
                         current_time=current_time)

@app.route("/detect_forge", methods=["POST"])
@login_required
def detect_forge():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file = request.files["file"]
        image_bytes = file.read()
        
        input_tensor = preprocess_image(image_bytes)
        forgery_map = detector.generate_forgery_map(input_tensor)
        visualization = apply_forgery_heatmap(image_bytes, forgery_map)
        
        buffered = io.BytesIO()
        visualization.save(buffered, format="PNG")
        visualization_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        with torch.no_grad():
            output, _ = model(input_tensor)
            confidence = output.mean().item()
        
        affected_area = float(np.mean(forgery_map > 0.5) * 100)
        num_regions = len(np.unique(forgery_map > 0.5)) - 1
        
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare analysis data
        analysis_data = {
            "timestamp": current_time,
            "prediction": "Fake" if confidence > 0.5 else "Real",
            "confidence": float(confidence),
            "affected_area_percentage": affected_area,
            "num_copied_regions": int(num_regions),
            "analyzed_by": session.get('username')
        }
        
        # Save analysis results to database
        if not save_analysis_result(session['user_id'], analysis_data):
            logger.warning("Failed to save analysis result to database")
        
        return jsonify({
            **analysis_data,
            "heatmap": visualization_base64
        })
    except Exception as e:
        logger.error(f"Error in detect_forge: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        file = request.files["file"]
        image_bytes = file.read()
        input_tensor = preprocess_image(image_bytes)

        with torch.no_grad():
            output_tensor, _ = model(input_tensor)
            confidence = output_tensor.mean().item()

        prediction = "Fake" if confidence > 0.5 else "Real"
        
        return jsonify({
            "prediction": prediction, 
            "confidence": float(confidence)
        })
    except Exception as e:
        logger.error(f"Error in predict: {str(e)}")
        return jsonify({"error": str(e)}), 500

def get_user_analysis_history(user_id):
    """Get user's analysis history"""
    try:
        ref = admin_db.reference(f'/analysis_results/{user_id}')
        results = ref.get()
        if results:
            # Convert to list and sort by timestamp
            history = [
                {**value, 'id': key} 
                for key, value in results.items()
            ]
            history.sort(key=lambda x: x['timestamp'], reverse=True)
            return history
        return []
    except Exception as e:
        logger.error(f"Error getting analysis history: {str(e)}")
        return []

def save_analysis_result(user_id, analysis_data):
    """Save analysis results to Firebase"""
    try:
        ref = admin_db.reference(f'/analysis_results/{user_id}')
        ref.push(analysis_data)
        return True
    except Exception as e:
        logger.error(f"Error saving analysis result: {str(e)}")
        return False

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html',
                         logged_in='user_id' in session,
                         username=session.get('username')), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return render_template('500.html',
                         logged_in='user_id' in session,
                         username=session.get('username')), 500

# if __name__ == "__main__":
#     # Verify database URL
#     db_url = os.getenv('FIREBASE_DATABASE_URL')
#     if not db_url:
#         raise ValueError("Database URL is not set in environment variables")
    
#     logger.info(f"Starting application with database URL: {db_url}")
#     app.run(debug=True)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)