import streamlit as st
from pages import login, patient, doctor, technician, admin, map
import database
from styles import setup_styles

def main():
    # Set page config
    st.set_page_config(page_title="Parasite Diagnosis System", layout="wide")
    
    # Setup custom styling
    setup_styles()
    
    # Initialize session state if needed
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'full_name' not in st.session_state:
        st.session_state.full_name = None
        
    # Setup database
    database.setup_database()
    
    # Sidebar for navigation
    st.sidebar.title("Parasite Diagnosis System")
    
    # Show different content based on login status
    if st.session_state.logged_in:
        # Display user info
        st.sidebar.write(f"Logged in as: {st.session_state.full_name}")
        st.sidebar.write(f"Role: {st.session_state.user_role}")
        
        # Logout button
        if st.sidebar.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Navigation options based on user role
        if st.session_state.user_role == "admin":
            page = st.sidebar.radio("Navigation", 
                ["Technician Portal", "Doctor Portal", "Patient Portal", "Outbreak Map", "User Management"])
        elif st.session_state.user_role == "technician":
            page = st.sidebar.radio("Navigation", 
                ["Technician Portal", "Outbreak Map"])
        elif st.session_state.user_role == "doctor":
            page = st.sidebar.radio("Navigation", 
                ["Doctor Portal", "Outbreak Map"])
        elif st.session_state.user_role == "patient":
            page = st.sidebar.radio("Navigation", 
                ["Patient Portal", "Outbreak Map"])
        
        # Display the selected page
        if page == "Patient Portal":
            patient.display_patient_portal()
        elif page == "Doctor Portal":
            doctor.display_doctor_portal()
        elif page == "Technician Portal":
            technician.display_technician_portal()
        elif page == "Outbreak Map":
            map.display_outbreak_map()
        elif page == "User Management":
            admin.display_user_management()
    else:
        # Show login page if not logged in
        login.display_login_page()

if __name__ == "__main__":
    main()