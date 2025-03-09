import streamlit as st
import time
import auth
from styles import formatted_header, info_box

def display_login_page():
    """Display the login page with tabs for patient login, doctor login, and registration"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        formatted_header("Parasite Diagnosis System")
        info_box("Please sign in to access the system.")
        
        # Create tabs for Sign In and Register
        tab1, tab2, tab3, tab4 = st.tabs(["Patient Sign In", "Doctor Sign In", "Technician Sign In", "Register New Account"])
        
        with tab1:
            with st.form("patient_login_form"):
                st.subheader("Patient Sign In")
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit_button = st.form_submit_button("Login", use_container_width=True)
                
                # Demo account info
                st.markdown("---")
                st.markdown("<small>For testing, register a new patient account</small>", unsafe_allow_html=True)
                
                if submit_button:
                    if username and password:
                        if auth.login_user(username, password, "patient"):
                            st.success(f"Welcome back, {st.session_state.full_name}!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
                    else:
                        st.warning("Please enter both username and password")

        with tab2:
            with st.form("doctor_login_form"):
                st.subheader("Doctor Sign In")
                doctor_username = st.text_input("Username")
                doctor_password = st.text_input("Password", type="password")
                doctor_submit_button = st.form_submit_button("Login", use_container_width=True)
                
                # Demo doctor account info
                st.markdown("---")
                st.markdown("<small>For testing: Username: doctor1, Password: doctor123</small>", unsafe_allow_html=True)
                
                if doctor_submit_button:
                    if doctor_username and doctor_password:
                        if auth.login_user(doctor_username, doctor_password, "doctor"):
                            st.success(f"Welcome back, Dr. {st.session_state.full_name}!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
                    else:
                        st.warning("Please enter both username and password")
                        
        with tab3:
            with st.form("technician_login_form"):
                st.subheader("Technician Sign In")
                tech_username = st.text_input("Username")
                tech_password = st.text_input("Password", type="password")
                tech_submit_button = st.form_submit_button("Login", use_container_width=True)
                
                # Demo technician account info
                st.markdown("---")
                st.markdown("<small>For testing, register a new technician account</small>", unsafe_allow_html=True)
                
                if tech_submit_button:
                    if tech_username and tech_password:
                        if auth.login_user(tech_username, tech_password, "technician"):
                            st.success(f"Welcome back, {st.session_state.full_name}!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
                    else:
                        st.warning("Please enter both username and password")
        
        with tab4:
            with st.form("register_form"):
                st.subheader("Create Your Account")
                
                new_username = st.text_input("Choose Username")
                new_password = st.text_input("Choose Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                full_name = st.text_input("Full Name")
                email = st.text_input("Email")
                role = st.selectbox("Account Type", ["patient", "technician"])  # Add technician option
                
                register_button = st.form_submit_button("Create Account", use_container_width=True)
                
                if register_button:
                    if not (new_username and new_password and confirm_password and full_name and email):
                        st.warning("Please fill in all fields")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        success, message = auth.register_user(new_username, new_password, full_name, email, role)
                        
                        if success:
                            st.success(f"Account created successfully! You can now sign in.")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(message)