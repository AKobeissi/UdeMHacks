import streamlit as st
import pandas as pd
import time
import database
import auth
from styles import formatted_header, role_badge

@auth.role_required(["admin"])
def display_user_management():
    """Display the user management page for admins"""
    formatted_header("User Management")
    
    # Get all users
    users = database.get_all_users()
    
    # Create tabs for different management functions
    tab1, tab2, tab3 = st.tabs(["Existing Users", "Create New User", "Delete User"])
    
    with tab1:
        st.subheader("Existing Users")
        
        # Add role filter
        role_filter = st.selectbox("Filter by role", ["All"] + list(users['role'].unique()))
        if role_filter != "All":
            filtered_users = users[users['role'] == role_filter]
        else:
            filtered_users = users
            
        # Format the roles with colored badges
        def format_role(role):
            return role_badge(role)
        
        # Display the table with formatted roles
        st.dataframe(
            filtered_users,
            column_config={
                "id": "ID",
                "username": "Username",
                "role": "Role",
                "email": "Email",
                "full_name": "Full Name",
                "date_registered": st.column_config.DatetimeColumn("Registration Date", format="MMM DD, YYYY")
            },
            hide_index=True,
            use_container_width=True
        )
    
    with tab2:
        st.subheader("Create New User")
        with st.form("create_user_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            role = st.selectbox("Role", ["technician", "doctor", "patient", "admin"])
            email = st.text_input("Email")
            full_name = st.text_input("Full Name")
            
            col1, col2 = st.columns(2)
            with col1:
                submit_button = st.form_submit_button("Create User", use_container_width=True)
            with col2:
                clear_button = st.form_submit_button("Clear Form", use_container_width=True)
            
            if submit_button:
                if not (username and password and confirm_password and role and email and full_name):
                    st.warning("Please fill in all fields")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    # Register the user
                    success, message = auth.register_user(username, password, full_name, email, role)
                    
                    if success:
                        st.success(message)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(message)
    
    with tab3:
        st.subheader("Delete User")
        with st.form("delete_user_form"):
            user_to_delete = st.selectbox("Select User", users['username'])
            confirm_delete = st.checkbox("I understand this action cannot be undone")
            
            delete_button = st.form_submit_button("Delete User", use_container_width=True)
            
            if delete_button:
                if confirm_delete:
                    # Don't allow deleting your own account
                    if user_to_delete == st.session_state.username:
                        st.error("You cannot delete your own account")
                    else:
                        user_id = users[users['username'] == user_to_delete]['id'].values[0]
                        
                        # Check if there are associated patient records
                        patient_count = database.execute_query(
                            "SELECT COUNT(*) FROM patients WHERE user_id = ?", 
                            (user_id,), 
                            fetchone=True
                        )
                        has_patients = patient_count[0] > 0 if patient_count else False
                        
                        # Check if there are associated samples
                        sample_count = database.execute_query(
                            "SELECT COUNT(*) FROM samples WHERE doctor_id = ?", 
                            (user_id,), 
                            fetchone=True
                        )
                        has_samples = sample_count[0] > 0 if sample_count else False
                        
                        if has_patients or has_samples:
                            st.warning("This user has associated records. Delete these first or reassign them.")
                            if st.button("Delete anyway (will orphan records)"):
                                if database.delete_user(user_id):
                                    st.success(f"User '{user_to_delete}' deleted successfully")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("Error deleting user")
                        else:
                            if database.delete_user(user_id):
                                st.success(f"User '{user_to_delete}' deleted successfully")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Error deleting user")
                else:
                    st.warning("Please confirm deletion")