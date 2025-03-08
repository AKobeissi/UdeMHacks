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
        
        patient_names = ["New Patient"] + patients['name'].tolist()
        selected_patient = st.selectbox("Select Patient", patient_names)
        
        if selected_patient == "New Patient":
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
            # Get selected patient details
            patient_id = patients[patients['name'] == selected_patient]['id'].values[0]
            
            with st.form("sample_upload_form"):
                st.subheader(f"Upload Sample for {selected_patient}")
                
                # Get patient location
                patient_info = database.execute_query(
                    "SELECT location, latitude, longitude FROM patients WHERE id = ?",
                    (patient_id,), fetchone=True
                )
                
                if patient_info is not None:
                    location, latitude, longitude = patient_info
                    
                    st.write(f"Location: {location}")
                    
                    if latitude is not None and longitude is not None:
                        st.write(f"Coordinates: ({latitude}, {longitude})")
                        
                        # Display patient location on mini map
                        m = folium.Map(location=[latitude, longitude], zoom_start=9)
                        folium.Marker([latitude, longitude], popup=location).add_to(m)
                        folium_static(m, width=400, height=300)
                    
                    uploaded_file = st.file_uploader("Upload microscope image", type=["jpg", "jpeg", "png"])
                    
                    submit_button = st.form_submit_button("Submit Sample for Analysis", use_container_width=True)
                    
                    if submit_button and uploaded_file is not None:
                        # Save the uploaded image
                        image_path, image = save_uploaded_image(uploaded_file, patient_id)
                        
                        if image_path and image:
                            # Update patient status
                            database.update_patient_status(patient_id, "Sample Collected")
                            
                            # Process image and get prediction
                            with st.spinner("Analyzing sample..."):
                                diagnosis, confidence = ml_model.analyze_sample(image)
                            
                            # Save sample info
                            sample_id = database.save_sample(patient_id, image_path, diagnosis, confidence)
                            
                            if sample_id:
                                st.success(f"Sample uploaded and analyzed successfully!")
                                
                                # Display result
                                result_col1, result_col2 = st.columns(2)
                                with result_col1:
                                    st.metric("AI Diagnosis", diagnosis)
                                with result_col2:
                                    st.metric("Confidence", f"{confidence:.2%}")
                                
                                # Flag low confidence predictions
                                if confidence < 0.7:
                                    st.warning("⚠️ Low confidence prediction. Flagged for doctor review.")
                            else:
                                st.error("Error saving sample information")
                else:
                    st.error(f"Patient information not found. Please try again.")
    
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
        
        # Get samples for this technician
        samples = database.execute_read_query(f"""
            SELECT s.id, p.name, s.image_path, s.ai_diagnosis, s.ai_confidence, 
                   s.doctor_diagnosis, s.doctor_verified, s.date_diagnosed
            FROM samples s
            JOIN patients p ON s.patient_id = p.id
            WHERE 
                {verified_condition} AND 
                {date_condition} AND
                p.user_id = ?
            ORDER BY s.date_diagnosed DESC
            LIMIT 20
        """, params=[st.session_state.user_id if st.session_state.user_role != "admin" else None])
        
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