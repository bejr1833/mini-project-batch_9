import os
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import numpy as np
import pandas as pd
from PIL import Image
import base64
from io import BytesIO
import uuid
from datetime import datetime

# Import face recognition utilities
from utils.face_recognitionss import extract_embedding, find_similar_person_from_file, add_new_image, SAVE_FOLDER 
from utils.auth import login_required, get_db, init_app, log_search, get_stats

# Configuration
from config import Config

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = SAVE_FOLDER

app.config.from_object(Config)  # Loading configuration from Config class

app.secret_key = os.urandom(24)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload


# Paths
CSV_PATH = app.config['CSV_PATH']
FACES_FOLDER = app.config['FACES_FOLDER']
EMBEDDINGS_PATH = app.config['EMBEDDINGS_PATH']

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_db_connection():
    conn = sqlite3.connect('face_recognition.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    conn.execute('''
    CREATE TABLE IF NOT EXISTS search_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        image_path TEXT,
        result_found BOOLEAN,
        matched_person_id TEXT,
        confidence REAL,
        search_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    conn.commit()
    conn.close()

# Initialize app with database functions
init_db()
init_app(app)

# Routes
@app.route('/')
def home():
    return render_template('home.html', logged_in='user_id' in session)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            
            # Redirect to the next page if provided, otherwise to home
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html', logged_in=False)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        conn = get_db_connection()
        existing_user = conn.execute('SELECT * FROM users WHERE username = ? OR email = ?', 
                                     (username, email)).fetchone()
        
        if existing_user:
            flash('Username or email already exists', 'danger')
            conn.close()
        else:
            hashed_password = generate_password_hash(password)
            conn.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                         (username, hashed_password, email))
            conn.commit()
            conn.close()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    
    return render_template('register.html', logged_in=False)

@app.route('/dashboard')
@login_required
def dashboard():
    # Get statistics
    stats = get_stats()
    
    # Get recent missing persons
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        missing_persons = []
        for _, row in df.iterrows():
            missing_persons.append({
                'id': row['id'],
                'name': row['label'],
                'missing_since': row['Missing Since'],
                'last_seen': row['Last Seen Location'],
                'contact': row['Mobile Number']
            })
    else:
        missing_persons = []
    
    return render_template('dashboard.html', stats=stats, missing_persons=missing_persons, logged_in=True)
@app.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('search.html', result={"error": "No file uploaded."})

        file = request.files['file']
        if file.filename == '':
            return render_template('search.html', result={"error": "No file selected."})

        result = find_similar_person_from_file(file)

        return render_template('search.html', result=result, uploaded_image=result.get("uploaded_image"))

    return render_template('search.html')

@app.route('/add_new', methods=['GET', 'POST'])
@login_required
def add_new():
    if request.method == 'POST':
        person_id = request.form['person_id']
        name = request.form['name']
        last_seen = request.form['last_seen']
        missing_since = request.form['missing_since']
        phone = request.form['phone']

        # Handling the file upload
        file = request.files['file']
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Call the function to add the new person to the database
            result = add_new_image(file_path, person_id, name, last_seen, missing_since, phone)
            return result  # Or redirect to another page with success message
        else:
            return "❌ Invalid image format. Please upload a valid image."

    return render_template('add_new.html')  # Render your 'add_new.html' template

@app.route('/api/person/<person_id>')
@login_required
def get_person(person_id):
    """API endpoint to get person details"""
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        person = df[df['id'] == person_id]
        
        if not person.empty:
            person_data = person.iloc[0]
            image_path = f"{FACES_FOLDER}/{person_data['id']}"
            
            # Convert image to base64
            if os.path.exists(image_path):
                img = Image.open(image_path)
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                return jsonify({
                    'id': person_data['id'],
                    'name': person_data['label'],
                    'missing_since': person_data['Missing Since'],
                    'last_seen': person_data['Last Seen Location'],
                    'contact': person_data['Mobile Number'],
                    'image': img_base64
                })
    
    return jsonify({'error': 'Person not found'}), 404

@app.errorhandler(404)
def page_not_found(e):
    return render_template('errors/404.html', logged_in='user_id' in session), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('errors/500.html', logged_in='user_id' in session), 500

@app.context_processor
def utility_processor():
    def now(format='%Y'):
        return datetime.now().strftime(format)
    return dict(now=now)

if __name__ == '__main__':
    # Make sure required directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    os.makedirs(FACES_FOLDER, exist_ok=True)
    
    app.run(debug=app.config['DEBUG'])