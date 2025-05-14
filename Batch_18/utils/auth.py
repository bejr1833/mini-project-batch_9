import sqlite3
from functools import wraps
from flask import session, redirect, url_for, flash, g, current_app, request

def login_required(f):
    """
    Decorator to require login for specific routes
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def get_db():
    """
    Get database connection for the current context
    """
    if 'db' not in g:
        g.db = sqlite3.connect('face_recognition.db')
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    """
    Close database connection at the end of request
    """
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_app(app):
    """
    Initialize app with database functions
    """
    app.teardown_appcontext(close_db)
    
    # Create database tables if they don't exist
    with app.app_context():
        db = get_db()
        db.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        db.execute('''
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
        
        db.commit()

def get_user_by_id(user_id):
    """
    Get user details by ID
    """
    db = get_db()
    return db.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()

def get_user_by_username(username):
    """
    Get user details by username
    """
    db = get_db()
    return db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()

def log_search(user_id, image_path, result_found, matched_person_id=None, confidence=None):
    """
    Log search activity for analytics
    """
    db = get_db()
    db.execute(
        'INSERT INTO search_logs (user_id, image_path, result_found, matched_person_id, confidence) VALUES (?, ?, ?, ?, ?)',
        (user_id, image_path, result_found, matched_person_id, confidence)
    )
    db.commit()

def get_stats():
    """
    Get system statistics for dashboard
    """
    db = get_db()
    
    # Count total persons in dataset
    try:
        with open(current_app.config['CSV_PATH'], 'r') as f:
            total_persons = sum(1 for line in f) - 1  # Subtract 1 for header
    except Exception:
        total_persons = 0
    
    # Count searches
    searches = db.execute('SELECT COUNT(*) as count FROM search_logs').fetchone()['count']
    
    # Count successful matches
    matches = db.execute('SELECT COUNT(*) as count FROM search_logs WHERE result_found = 1').fetchone()['count']
    
    return {
        'total_persons': total_persons,
        'searches': searches,
        'matches': matches
    }