import streamlit as st

def setup_styles():
    """Add custom CSS for better UI"""
    st.markdown("""
    <style>
        .main-header {text-align: center; color: #0c326f; margin-bottom: 20px;}
        .sub-header {color: #2c5282; margin-top: 20px;}
        .success-text {color: #2e7d32;}
        .warning-text {color: #ff9800;}
        .error-text {color: #d32f2f;}
        .info-box {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #0c326f;
            margin-bottom: 15px;
        }
        .role-badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            margin-left: 10px;
        }
        .role-admin {background-color: #d81b60; color: white;}
        .role-doctor {background-color: #1e88e5; color: white;}
        .role-patient {background-color: #43a047; color: white;}
        .role-technician {background-color: #fb8c00; color: white;}
    </style>
    """, unsafe_allow_html=True)

def formatted_header(text, level="h1"):
    """Create a formatted header"""
    if level == "h1":
        st.markdown(f"<h1 class='main-header'>{text}</h1>", unsafe_allow_html=True)
    elif level == "h2":
        st.markdown(f"<h2 class='sub-header'>{text}</h2>", unsafe_allow_html=True)
    elif level == "h3":
        st.markdown(f"<h3 class='sub-header'>{text}</h3>", unsafe_allow_html=True)

def info_box(text):
    """Create an info box with styled content"""
    st.markdown(f"<div class='info-box'>{text}</div>", unsafe_allow_html=True)

def role_badge(role):
    """Create a colored badge for user roles"""
    return f"<span class='role-badge role-{role}'>{role.capitalize()}</span>"