import hashlib
import uuid
import streamlit as st
from functools import wraps
import database

def hash_password(password, salt=None):
    """Create a password hash with salt"""
    if salt is None:
        salt = uuid.uuid4().hex
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return password_hash, salt

def login_user(username, password, role=None):
    """Authenticate a user and set session state"""
    user = database.verify_user(username, password, role)
    
    if user:
        st.session_state.logged_in = True
        st.session_state.user_id = user["id"]
        st.session_state.user_role = user["role"]
        st.session_state.username = username
        st.session_state.full_name = user["full_name"]
        return True
    
    return False

def logout_user():
    """Clear the session state to log out the user"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]

def login_required(func):
    """Decorator to require login for a function"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not st.session_state.get('logged_in', False):
            st.error("Please log in to access this page")
            return None
        return func(*args, **kwargs)
    return wrapper

def role_required(allowed_roles):
    """Decorator to require specific role(s) for a function"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not st.session_state.get('logged_in', False):
                st.error("Please log in to access this page")
                return None
            
            if st.session_state.get('user_role') not in allowed_roles:
                st.error(f"You need to be a {' or '.join(allowed_roles)} to access this page")
                return None
                
            return func(*args, **kwargs)
        return wrapper
    return decorator

def register_user(username, password, full_name, email, role="patient"):
    """Register a new user"""
    # Check if username already exists
    if database.username_exists(username):
        return False, "Username already exists"
    
    # Create the user
    user_id = database.create_user(username, password, role, email, full_name)
    
    if user_id:
        return True, f"User {username} created successfully"
    else:
        return False, "Error creating user"