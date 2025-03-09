import streamlit as st

def setup_styles():
    """Setup custom styling for the app"""
    # Set custom CSS styles
    st.markdown("""
    <style>
    /* Main container */
    .main {
        background-color: #f8f9fa;
        padding: 20px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #e9ecef;
    }
    
    /* Headers */
    h1 {
        color: #2c3e50;
        margin-bottom: 20px;
    }
    
    h2 {
        color: #34495e;
        margin-bottom: 15px;
    }
    
    h3 {
        color: #3498db;
        margin-bottom: 10px;
    }
    
    /* Cards/boxes for content */
    .stCard {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Success messages */
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        border-color: #c3e6cb;
    }
    
    /* Error messages */
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        border-color: #f5c6cb;
    }
    
    /* Warning messages */
    .stWarning {
        background-color: #fff3cd;
        color: #856404;
        border-color: #ffeeba;
    }
    
    /* Info messages */
    .stInfo {
        background-color: #d1ecf1;
        color: #0c5460;
        border-color: #bee5eb;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 8px 16px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #2980b9;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    /* Tables */
    .dataframe {
        width: 100%;
        border-collapse: collapse;
    }
    
    .dataframe th {
        background-color: #3498db;
        color: white;
        text-align: left;
        padding: 8px;
    }
    
    .dataframe td {
        padding: 8px;
        border-bottom: 1px solid #ddd;
    }
    
    .dataframe tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    
    .dataframe tr:hover {
        background-color: #e1e1e1;
    }
    
    /* Custom info box */
    .info-box {
        background-color: #d1ecf1;
        border-left: 5px solid #0c5460;
        padding: 10px 15px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    
    /* Custom header formatting */
    .custom-header {
        background: linear-gradient(90deg, #3498db, #2c3e50);
        color: white;
        padding: 10px 15px;
        border-radius: 5px;
        margin-bottom: 25px;
    }
    
    /* Role badges */
    .badge {
        display: inline-block;
        padding: 3px 7px;
        font-size: 12px;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 10px;
    }
    .badge-patient {
        background-color: #28a745;
        color: white;
    }
    .badge-doctor {
        background-color: #007bff;
        color: white;
    }
    .badge-technician {
        background-color: #6f42c1;
        color: white;
    }
    .badge-admin {
        background-color: #dc3545;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

def formatted_header(title):
    """Display a formatted header with the given title"""
    st.markdown(f'<div class="custom-header"><h1>{title}</h1></div>', unsafe_allow_html=True)

def info_box(text):
    """Display an info box with the given text"""
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)

def role_badge(role):
    """Return HTML for a role badge"""
    colors = {
        "patient": "badge-patient",
        "doctor": "badge-doctor",
        "technician": "badge-technician",
        "admin": "badge-admin"
    }
    badge_class = colors.get(role.lower(), "badge-secondary")
    return f'<span class="badge {badge_class}">{role}</span>'