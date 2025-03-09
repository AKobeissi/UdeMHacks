import streamlit as st
import os
import time
import folium
from streamlit_folium import folium_static
import database
import auth
import ml_model
from styles import formatted_header, info_box
from utils import save_uploaded_image
from datetime import datetime

@auth.role_required(["technician", "admin"])
def display_technician_portal():
    """Display the technician portal for uploading samples"""
    formatted_header("Technician Portal")
    info_box("Upload and manage patient samples for parasite diagnosis")
        
    # Create tabs for different technician functions
    tab1, tab2 = st.tabs(["Upload New Sample", "View Previous Samples"])
    
    with tab1:
        # Get patients that this technician has registered (or all for admin)
        patients = database.get_technician_patients(
            None if st.session_state.user_role == "admin" else st.session_state.user_id
        )
        
        # Check if there are any patients
        if patients.empty:
            st.warning("No patients found. Please register a patient first.")
            
            # New patient form
            with st.form("new_patient_form"):
                st.subheader("Register New Patient")
                patient_name = st.text_input("Patient Name")
                location = st.text_input("Location")
                
                col1, col2 = st.columns(2)
                with col1:
                    latitude = st.number_input("Latitude", -90.0, 90.0, format="%.6f")
                with col2:
                    longitude = st.number_input("Longitude", -180.0, 180.0, format="%.6f")
                
                # Add a map for selecting coordinates
                m = folium.Map(location=[0, 0], zoom_start=2)
                folium_static(m)
                st.caption("Click on the map to select approximate coordinates (for demonstration)")
                
                submit_button = st.form_submit_button("Register Patient", use_container_width=True)
                
                if submit_button and patient_name and location:
                    # Create the patient
                    patient_id = database.create_patient(
                        patient_name, location, latitude, longitude, st.session_state.user_id
                    )
                    
                    if patient_id:
                        st.success(f"Patient {patient_name} registered successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Error registering patient")
        else:
            # Two options: new patient or existing patient
            option = st.radio("Choose option:", ["Select Existing Patient", "Register New Patient"])
            
            if option == "Register New Patient":
                # New patient form
                with st.form("new_patient_form_2"):
                    st.subheader("Register New Patient")
                    patient_name = st.text_input("Patient Name")
                    location = st.text_input("Location")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        latitude = st.number_input("Latitude", -90.0, 90.0, format="%.6f")
                    with col2:
                        longitude = st.number_input("Longitude", -180.0, 180.0, format="%.6f")
                    
                    # Add a map for selecting coordinates
                    m = folium.Map(location=[0, 0], zoom_start=2)
                    folium_static(m)
                    st.caption("Click on the map to select approximate coordinates (for demonstration)")
                    
                    submit_button = st.form_submit_button("Register Patient", use_container_width=True)
                    
                    if submit_button and patient_name and location:
                        # Create the patient
                        patient_id = database.create_patient(
                            patient_name, location, latitude, longitude, st.session_state.user_id
                        )
                        
                        if patient_id:
                            st.success(f"Patient {patient_name} registered successfully!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Error registering patient")
            else:
                # Select existing patient
                selected_patient = st.selectbox(
                    "Select Patient", 
                    options=patients['id'].tolist(),
                    format_func=lambda x: patients.loc[patients['id'] == x, 'name'].iloc[0]
                )
                
                # Get selected patient details outside of the form
                patient_info = database.execute_query(
                    "SELECT name, location, latitude, longitude FROM patients WHERE id = ?",
                    (selected_patient,), fetchone=True
                )
                
                if patient_info:
                    name, location, latitude, longitude = patient_info
                    
                    st.success(f"Selected patient: {name}")
                    st.write(f"Location: {location}")
                    
                    # Display patient location on mini map if coordinates available
                    if latitude is not None and longitude is not None:
                        m = folium.Map(location=[latitude, longitude], zoom_start=9)
                        folium.Marker([latitude, longitude], popup=location).add_to(m)
                        folium_static(m, width=400, height=300)
                    
                    # Separate the file upload from the form
                    st.subheader(f"Upload Sample for {name}")
                    uploaded_file = st.file_uploader("Upload microscope image", type=["jpg", "jpeg", "png"])
                    
                    if uploaded_file is not None:
                        # Display the image
                        st.image(uploaded_file, caption="Uploaded image", use_container_width =True)
                        
                        # Analyze button (outside form)
                        if st.button("Analyze Sample"):
                            with st.spinner("Analyzing sample..."):
                                # Save the uploaded image
                                image_path, image = save_uploaded_image(uploaded_file, selected_patient)
                                
                                if image_path and image:
                                    # Update patient status
                                    database.update_patient_status(selected_patient, "Sample Collected")
                                    
                                    # Process image and get prediction
                                    diagnosis, confidence = ml_model.analyze_sample(image)
                                    
                                    # Save sample info
                                    sample_id = database.save_sample(
                                        selected_patient, image_path, diagnosis, confidence
                                    )
                                    
                                    if sample_id:
                                        # Display the result
                                        st.success("Sample analysis complete!")
                                        
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.metric("AI Diagnosis", diagnosis)
                                        with col2:
                                            st.metric("Confidence", f"{confidence:.2f}")
                                        
                                        if confidence < 0.7:
                                            st.warning("⚠️ Low confidence prediction. Flagged for doctor review.")
                                        else:
                                            st.info("Sample has been sent for doctor verification.")
                                    else:
                                        st.error("Failed to save the sample to the database.")
                                else:
                                    st.error("Failed to process the uploaded image.")
                else:
                    st.error("Patient information not found. Please try again.")
    
    with tab2:
        # Show previous samples
        st.subheader("Previous Samples")
        
        # Add filters
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            days = st.slider("Days back", 1, 90, 30)
        with filter_col2:
            verified_only = st.checkbox("Show verified only")
        
        verified_condition = "s.doctor_verified = 1" if verified_only else "1=1"
        date_condition = f"s.date_diagnosed > datetime('now', '-{days} days')"
        
        # Get user ID for query (None for admin to see all samples)
        query_user_id = None if st.session_state.user_role == "admin" else st.session_state.user_id
        
        # # Get samples for this technician
        # if query_user_id:
        #     samples = database.execute_read_query(f"""
        #         SELECT s.id, p.name, s.image_path, s.ai_diagnosis, s.ai_confidence, 
        #                s.doctor_diagnosis, s.doctor_verified, s.date_diagnosed
        #         FROM samples s
        #         JOIN patients p ON s.patient_id = p.id
        #         WHERE 
        #             {verified_condition} AND 
        #             {date_condition} AND
        #             p.user_id = ?
        #         ORDER BY s.date_diagnosed DESC
        #         LIMIT 20
        #     """, params=[query_user_id])
        # else:
        #     # For admin, show all samples
        #     samples = database.execute_read_query(f"""
        #         SELECT s.id, p.name, s.image_path, s.ai_diagnosis, s.ai_confidence, 
        #                s.doctor_diagnosis, s.doctor_verified, s.date_diagnosed
        #         FROM samples s
        #         JOIN patients p ON s.patient_id = p.id
        #         WHERE 
        #             {verified_condition} AND 
        #             {date_condition}
        #         ORDER BY s.date_diagnosed DESC
        #         LIMIT 20
        #     """)
        # Modified query that doesn't rely on p.user_id
        try:
            samples = database.execute_read_query(f"""
                SELECT s.id, p.name, s.image_path, s.ai_diagnosis, s.ai_confidence, 
                    s.doctor_diagnosis, s.doctor_verified, s.date_diagnosed
                FROM samples s
                JOIN patients p ON s.patient_id = p.id
                WHERE 
                    {verified_condition} AND 
                    {date_condition}
                ORDER BY s.date_diagnosed DESC
                LIMIT 20
            """)
            
            st.info("Note: Showing all samples due to database schema issue")
            
        except Exception as e:
            st.error(f"Error fetching samples: {str(e)}")
            samples = pd.DataFrame()  # Create an empty DataFrame on error

        # Replace with this temporary approach:
        # try:
        #     if query_user_id:
        #         samples = database.execute_read_query(f"""
        #             SELECT s.id, p.name, s.image_path, s.ai_diagnosis, s.ai_confidence, 
        #                 s.doctor_diagnosis, s.doctor_verified, s.date_diagnosed
        #             FROM samples s
        #             JOIN patients p ON s.patient_id = p.id
        #             WHERE 
        #                 {verified_condition} AND 
        #                 {date_condition} AND
        #                 p.user_id = ?
        #             ORDER BY s.date_diagnosed DESC
        #             LIMIT 20
        #         """, params=[query_user_id])
        #     else:
        #         # For admin, show all samples
        #         samples = database.execute_read_query(f"""
        #             SELECT s.id, p.name, s.image_path, s.ai_diagnosis, s.ai_confidence, 
        #                 s.doctor_diagnosis, s.doctor_verified, s.date_diagnosed
        #             FROM samples s
        #             JOIN patients p ON s.patient_id = p.id
        #             WHERE 
        #                 {verified_condition} AND 
        #                 {date_condition}
        #             ORDER BY s.date_diagnosed DESC
        #             LIMIT 20
        #         """)
        # except:
        #     # Fallback to a simpler query without the user_id filter
        #     st.warning("Using fallback query mode due to database schema issue")
        #     samples = database.execute_read_query(f"""
        #         SELECT s.id, p.name, s.image_path, s.ai_diagnosis, s.ai_confidence, 
        #             s.doctor_diagnosis, s.doctor_verified, s.date_diagnosed
        #         FROM samples s
        #         JOIN patients p ON s.patient_id = p.id
        #         WHERE 
        #             {verified_condition} AND 
        #             {date_condition}
        #         ORDER BY s.date_diagnosed DESC
        #         LIMIT 20
        #     """)
        
        if not samples.empty:
            # Display as a data table first
            st.dataframe(
                samples[['name', 'ai_diagnosis', 'ai_confidence', 'doctor_verified', 'date_diagnosed']],
                column_config={
                    "name": "Patient",
                    "ai_diagnosis": "AI Diagnosis",
                    "ai_confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
                    "doctor_verified": st.column_config.CheckboxColumn("Verified"),
                    "date_diagnosed": "Date"
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Then show detailed expandable cards
            st.markdown("### Sample Details")
            for _, row in samples.iterrows():
                with st.expander(f"Sample: {row['name']} - {row['date_diagnosed']}"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if os.path.exists(row['image_path']):
                            st.image(row['image_path'], caption="Sample Image", width=300)
                        else:
                            st.error("Image file not found")
                    
                    with col2:
                        st.write(f"AI Diagnosis: {row['ai_diagnosis']} (Confidence: {row['ai_confidence']:.2f})")
                        if row['doctor_verified']:
                            st.write(f"Doctor's Diagnosis: {row['doctor_diagnosis']}")
                            st.success("✅ Verified by doctor")
                        else:
                            st.warning("⏳ Awaiting doctor verification")
        else:
            st.info("No samples found for the selected criteria")


            