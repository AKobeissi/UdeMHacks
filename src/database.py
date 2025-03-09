import sqlite3
import streamlit as st
import pandas as pd
from datetime import datetime
import hashlib
import uuid
import os
from config import DB_NAME, DEFAULT_ADMIN, DEFAULT_DOCTOR, DEFAULT_TECHNICIAN

def get_connection():
    """Get a connection to the database"""
    try:
        conn = sqlite3.connect(DB_NAME)
        return conn
    except sqlite3.Error as e:
        st.error(f"Database connection error: {e}")
        return None
def setup_database():
    """Initialize the database and create tables if they don't exist"""
    conn = get_connection()
    if conn is None:
        return None
    
    c = conn.cursor()
    
    # Create tables
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE,
        password_hash TEXT,
        salt TEXT,
        role TEXT,
        email TEXT,
        full_name TEXT,
        date_registered DATETIME
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY,
        name TEXT,
        location TEXT,
        latitude REAL,
        longitude REAL,
        date_collected DATETIME,
        status TEXT,
        user_id INTEGER,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS samples (
        id INTEGER PRIMARY KEY,
        patient_id INTEGER,
        image_path TEXT,
        ai_diagnosis TEXT,
        ai_confidence REAL,
        doctor_diagnosis TEXT,
        doctor_verified INTEGER DEFAULT 0,
        doctor_id INTEGER,
        date_diagnosed DATETIME,
        FOREIGN KEY (patient_id) REFERENCES patients (id),
        FOREIGN KEY (doctor_id) REFERENCES users (id)
    )
    ''')
    
    # Create default admin user if not exists
    c.execute("SELECT * FROM users WHERE username = ?", (DEFAULT_ADMIN["username"],))
    if not c.fetchone():
        salt = uuid.uuid4().hex
        password_hash = hashlib.sha256((DEFAULT_ADMIN["password"] + salt).encode()).hexdigest()
        c.execute(
            "INSERT INTO users (username, password_hash, salt, role, email, full_name, date_registered) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (DEFAULT_ADMIN["username"], password_hash, salt, "admin", DEFAULT_ADMIN["email"], 
             DEFAULT_ADMIN["full_name"], datetime.now())
        )
    
    # Create default doctor user if not exists
    c.execute("SELECT * FROM users WHERE username = ?", (DEFAULT_DOCTOR["username"],))
    if not c.fetchone():
        salt = uuid.uuid4().hex
        password_hash = hashlib.sha256((DEFAULT_DOCTOR["password"] + salt).encode()).hexdigest()
        c.execute(
            "INSERT INTO users (username, password_hash, salt, role, email, full_name, date_registered) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (DEFAULT_DOCTOR["username"], password_hash, salt, "doctor", DEFAULT_DOCTOR["email"], 
             DEFAULT_DOCTOR["full_name"], datetime.now())
        )
    
    # Create default technician user if not exists
    c.execute("SELECT * FROM users WHERE username = ?", (DEFAULT_TECHNICIAN["username"],))
    if not c.fetchone():
        salt = uuid.uuid4().hex
        password_hash = hashlib.sha256((DEFAULT_TECHNICIAN["password"] + salt).encode()).hexdigest()
        c.execute(
            "INSERT INTO users (username, password_hash, salt, role, email, full_name, date_registered) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (DEFAULT_TECHNICIAN["username"], password_hash, salt, "technician", DEFAULT_TECHNICIAN["email"], 
             DEFAULT_TECHNICIAN["full_name"], datetime.now())
        )
    
    conn.commit()
    conn.close()
    
    # Check and fix any missing columns
    check_and_fix_samples_table()
    fix_patients_schema()


def check_and_fix_patients_table():
    """Check if the patients table has the user_id column and add it if missing"""
    conn = get_connection()
    if conn is None:
        return False
    
    try:
        c = conn.cursor()
        
        # Check if the column exists
        c.execute("PRAGMA table_info(patients)")
        columns = [info[1] for info in c.fetchall()]
        
        # If user_id column doesn't exist, add it
        if "user_id" not in columns:
            c.execute("ALTER TABLE patients ADD COLUMN user_id INTEGER")
            
            # Update existing records to set a default user_id (admin user)
            c.execute("UPDATE patients SET user_id = 1 WHERE user_id IS NULL")
            
            conn.commit()
            st.success("Added missing user_id column to patients table and updated existing records")
        
        conn.close()
        return True
    except sqlite3.Error as e:
        st.error(f"Database schema check failed: {e}")
        conn.close()
        return False


def fix_patients_schema():
    """Fix the patients table schema by adding missing columns"""
    conn = get_connection()
    if conn is None:
        return False
    
    try:
        c = conn.cursor()
        
        # Check if the columns exist
        c.execute("PRAGMA table_info(patients)")
        columns = [info[1] for info in c.fetchall()]
        
        # Add missing columns if needed
        if "user_id" not in columns:
            try:
                c.execute("ALTER TABLE patients ADD COLUMN user_id INTEGER")
                st.success("Added missing user_id column to patients table")
            except sqlite3.Error as e:
                st.error(f"Error adding user_id column: {e}")
        
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        st.error(f"Database schema check failed: {e}")
        if conn:
            conn.close()
        return False
def reset_database():
    """Reset the database by dropping and recreating all tables - USE WITH CAUTION"""
    import os
    from config import DB_NAME
    
    # Close any connections
    try:
        conn = get_connection()
        if conn:
            conn.close()
    except:
        pass
    
    # Delete the database file
    try:
        if os.path.exists(DB_NAME):
            os.remove(DB_NAME)
            st.success(f"Database file {DB_NAME} deleted")
        else:
            st.warning(f"Database file {DB_NAME} not found")
    except Exception as e:
        st.error(f"Error deleting database file: {e}")
        return False
    
    # Re-create everything
    setup_database()
    return True
# def reset_database():
#     """Reset the database by dropping and recreating all tables - USE WITH CAUTION"""
#     import os
#     from config import DB_NAME
    
#     # Close any connections
#     try:
#         conn = get_connection()
#         if conn:
#             conn.close()
#     except:
#         pass
    
#     # Delete the database file
#     try:
#         if os.path.exists(DB_NAME):
#             os.remove(DB_NAME)
#             st.success(f"Database file {DB_NAME} deleted")
#         else:
#             st.warning(f"Database file {DB_NAME} not found")
#     except Exception as e:
#         st.error(f"Error deleting database file: {e}")
#         return False
    
#     # Re-create everything
#     setup_database()
#     return True

def execute_query(query, params=(), fetchone=False):
    """Execute a SQL query and return results"""
    conn = get_connection()
    if conn is None:
        return None
    
    try:
        c = conn.cursor()
        c.execute(query, params)
        
        if query.strip().upper().startswith(("SELECT", "PRAGMA")):
            if fetchone:
                result = c.fetchone()
            else:
                result = c.fetchall()
        else:
            conn.commit()
            result = c.lastrowid
        
        conn.close()
        return result
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        conn.close()
        return None

def execute_read_query(query, params=()):
    """Execute a SQL query and return results as a DataFrame"""
    conn = get_connection()
    if conn is None:
        return pd.DataFrame()
    
    try:
        result = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return result
    except (sqlite3.Error, pd.io.sql.DatabaseError) as e:
        st.error(f"Database error: {e}")
        conn.close()
        return pd.DataFrame()

def create_user(username, password, role, email, full_name):
    """Create a new user"""
    salt = uuid.uuid4().hex
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    
    # Insert the new user
    user_id = execute_query(
        "INSERT INTO users (username, password_hash, salt, role, email, full_name, date_registered) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (username, password_hash, salt, role, email, full_name, datetime.now())
    )
    
    # If creating a patient, also create a patient record
    if role == "patient" and user_id:
        execute_query(
            "INSERT INTO patients (name, location, date_collected, status, user_id) VALUES (?, ?, ?, ?, ?)",
            (full_name, "Not specified", datetime.now(), "Registered", user_id)
        )
    
    return user_id

def verify_user(username, password, role=None):
    """Verify user credentials"""
    if role:
        user = execute_query(
            "SELECT id, password_hash, salt, role, full_name FROM users WHERE username = ? AND role = ?",
            (username, role), fetchone=True
        )
    else:
        user = execute_query(
            "SELECT id, password_hash, salt, role, full_name FROM users WHERE username = ?",
            (username,), fetchone=True
        )
    
    if user and verify_password(user[1], user[2], password):
        return {
            "id": user[0],
            "role": user[3],
            "full_name": user[4]
        }
    return None

def verify_password(stored_hash, stored_salt, provided_password):
    """Verify a password against its stored hash"""
    computed_hash = hashlib.sha256((provided_password + stored_salt).encode()).hexdigest()
    return computed_hash == stored_hash

def username_exists(username):
    """Check if a username already exists"""
    result = execute_query(
        "SELECT COUNT(*) FROM users WHERE username = ?",
        (username,), fetchone=True
    )
    return result[0] > 0 if result else False

def save_sample(patient_id, image_path, ai_diagnosis, ai_confidence):
    """Save a new sample"""
    return execute_query(
        "INSERT INTO samples (patient_id, image_path, ai_diagnosis, ai_confidence, date_diagnosed) VALUES (?, ?, ?, ?, ?)",
        (patient_id, image_path, ai_diagnosis, ai_confidence, datetime.now())
    )

def update_sample_verification(sample_id, doctor_diagnosis, verified, doctor_id):
    """Update a sample with doctor verification"""
    try:
        # Try the original query with doctor_id
        result = execute_query(
            "UPDATE samples SET doctor_diagnosis = ?, doctor_verified = ?, doctor_id = ? WHERE id = ?",
            (doctor_diagnosis, 1 if verified else 0, doctor_id, sample_id)
        )
        return result
    except sqlite3.Error:
        # Fallback if doctor_id column doesn't exist
        result = execute_query(
            "UPDATE samples SET doctor_diagnosis = ?, doctor_verified = ? WHERE id = ?",
            (doctor_diagnosis, 1 if verified else 0, sample_id)
        )
        return result

def check_and_fix_samples_table():
    """Check if the samples table has the doctor_id column and add it if missing"""
    conn = get_connection()
    if conn is None:
        return False
    
    try:
        c = conn.cursor()
        
        # Check if the column exists
        c.execute("PRAGMA table_info(samples)")
        columns = [info[1] for info in c.fetchall()]
        
        # If doctor_id column doesn't exist, add it
        if "doctor_id" not in columns:
            c.execute("ALTER TABLE samples ADD COLUMN doctor_id INTEGER")
            conn.commit()
            st.success("Added missing doctor_id column to samples table")
        
        conn.close()
        return True
    except sqlite3.Error as e:
        st.error(f"Database schema check failed: {e}")
        conn.close()
        return False
    
def create_patient(name, location, latitude, longitude, user_id):
    """Create a new patient record"""
    return execute_query(
        "INSERT INTO patients (name, location, latitude, longitude, date_collected, status, user_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (name, location, latitude, longitude, datetime.now(), "Registered", user_id)
    )

def update_patient_status(patient_id, status):
    """Update a patient's status"""
    return execute_query(
        "UPDATE patients SET status = ? WHERE id = ?",
        (status, patient_id)
    )

def get_patient_by_user_id(user_id):
    """Get patient record by user ID"""
    return execute_query(
        "SELECT id FROM patients WHERE user_id = ?",
        (user_id,), fetchone=True
    )

def get_patient_samples(patient_id):
    """Get all samples for a patient"""
    return execute_read_query(
        """
        SELECT p.id, p.name, p.location, s.ai_diagnosis, s.doctor_diagnosis, 
               s.doctor_verified, s.ai_confidence, s.date_diagnosed, s.image_path
        FROM patients p
        JOIN samples s ON p.id = s.patient_id
        WHERE p.id = ?
        ORDER BY s.date_diagnosed DESC
        """,
        params=[patient_id]
    )

def get_unverified_samples(confidence_threshold=None, order_by="confidence"):
    """Get unverified samples for doctor review"""
    query = """
    SELECT s.id, p.id as patient_id, p.name, p.location, s.image_path, s.ai_diagnosis, s.ai_confidence, 
           s.doctor_diagnosis, s.doctor_verified, s.date_diagnosed,
           u.full_name as technician_name
    FROM samples s
    JOIN patients p ON s.patient_id = p.id
    JOIN users u ON p.user_id = u.id
    WHERE s.doctor_verified = 0
    """
    
    params = []
    
    if confidence_threshold is not None:
        query += " AND s.ai_confidence < ?"
        params.append(confidence_threshold)
    
    if order_by == "confidence":
        query += " ORDER BY s.ai_confidence ASC, s.date_diagnosed DESC"
    else:
        query += " ORDER BY s.date_diagnosed DESC, s.ai_confidence ASC"
    
    return execute_read_query(query, params)

def get_technician_patients(user_id):
    """Get patients for a technician"""
    if user_id:
        return execute_read_query(
            "SELECT id, name FROM patients WHERE user_id = ? ORDER BY name",
            params=[user_id]
        )
    else:
        return execute_read_query(
            "SELECT id, name FROM patients ORDER BY name"
        )

def get_technician_samples(user_id):
    """Get samples uploaded by a technician"""
    if user_id:
        return execute_read_query(
            """
            SELECT s.id, p.name, s.image_path, s.ai_diagnosis, s.ai_confidence, 
                   s.doctor_diagnosis, s.doctor_verified, s.date_diagnosed
            FROM samples s
            JOIN patients p ON s.patient_id = p.id
            WHERE p.user_id = ?
            ORDER BY s.date_diagnosed DESC
            LIMIT 20
            """,
            params=[user_id]
        )
    else:
        return execute_read_query(
            """
            SELECT s.id, p.name, s.image_path, s.ai_diagnosis, s.ai_confidence, 
                   s.doctor_diagnosis, s.doctor_verified, s.date_diagnosed
            FROM samples s
            JOIN patients p ON s.patient_id = p.id
            ORDER BY s.date_diagnosed DESC
            LIMIT 20
            """
        )

def get_doctor_verifications(doctor_id):
    """Get samples verified by a doctor"""
    return execute_read_query(
        """
        SELECT s.id, p.name, s.date_diagnosed, s.ai_diagnosis, s.doctor_diagnosis
        FROM samples s
        JOIN patients p ON s.patient_id = p.id
        WHERE s.doctor_verified = 1 AND s.doctor_id = ?
        ORDER BY s.date_diagnosed DESC
        """,
        params=[doctor_id]
    )

def get_outbreak_data():
    """Get data for the outbreak map"""
    return execute_read_query(
        """
        SELECT p.latitude, p.longitude, s.ai_diagnosis, s.doctor_diagnosis, s.doctor_verified
        FROM patients p
        JOIN samples s ON p.id = s.patient_id
        WHERE p.latitude IS NOT NULL AND p.longitude IS NOT NULL
        """
    )

def get_outbreak_statistics():
    """Get statistics for outbreak visualization"""
    return execute_read_query(
        """
        SELECT 
            CASE 
                WHEN s.doctor_verified = 1 THEN s.doctor_diagnosis 
                ELSE s.ai_diagnosis 
            END as diagnosis,
            COUNT(*) as count
        FROM samples s
        GROUP BY diagnosis
        ORDER BY count DESC
        """
    )

def get_all_users():
    """Get all users"""
    return execute_read_query(
        "SELECT id, username, role, email, full_name, date_registered FROM users"
    )

def delete_user(user_id):
    """Delete a user"""
    return execute_query("DELETE FROM users WHERE id = ?", (user_id,))