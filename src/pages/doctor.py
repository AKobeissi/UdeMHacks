import streamlit as st
import os
import pandas as pd
import time
import database
import auth
from styles import formatted_header, info_box
from utils import generate_doctor_insights

@auth.role_required(["doctor", "admin"])
def display_doctor_portal():
    """Display the doctor portal for reviewing diagnoses"""
    formatted_header("Doctor Portal")
    info_box("Review AI diagnoses and provide expert verification")
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Pending Reviews", "Search Patients", "My Verifications"])
    
    with tab1:
        # Filter options
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            filter_verified = st.radio("Show", ["Unverified only", "All samples"])
        
        with filter_col2:
            filter_confidence = st.slider("Max confidence threshold", 0.0, 1.0, 0.7, 
                                          help="Show samples with AI confidence below this value")
        
        with filter_col3:
            sort_order = st.selectbox("Sort by", ["Lowest confidence first", "Most recent first"])
        
        # Get the samples based on filters
        confidence_threshold = filter_confidence if filter_verified == "Unverified only" else None
        order_by = "confidence" if sort_order == "Lowest confidence first" else "date"
        
        samples = database.get_unverified_samples(confidence_threshold, order_by)
        
        if samples.empty:
            st.info("No samples to review matching your criteria.")
        else:
            st.write(f"Found {len(samples)} samples to review")
            
            # Show a summary table first
            st.dataframe(
                samples[['name', 'ai_diagnosis', 'ai_confidence', 'date_diagnosed']],
                column_config={
                    "name": "Patient",
                    "ai_diagnosis": "AI Diagnosis",
                    "ai_confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
                    "date_diagnosed": "Date"
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Then show detailed review cards
            for _, row in samples.iterrows():
                with st.expander(f"Patient: {row['name']} - AI Diagnosis: {row['ai_diagnosis']} (Conf: {row['ai_confidence']:.2f})"):
                    # Display sample info
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if os.path.exists(row['image_path']):
                            st.image(row['image_path'], caption="Sample Image")
                        else:
                            st.error("Image file not found")
                    
                    with col2:
                        st.write(f"Location: {row['location']}")
                        st.write(f"Submitted by: {row['technician_name']}")
                        st.write(f"Date: {row['date_diagnosed']}")
                        
                        # Add a toggle for showing clinical assessment
                        show_assessment = st.checkbox("Show Clinical Assessment", value=True, key=f"show_assess_{row['id']}")
                        
                        if show_assessment:
                            # Clinical assessment
                            with st.spinner("Generating clinical assessment..."):
                                doctor_insights = generate_doctor_insights(
                                    row['ai_diagnosis'], 
                                    row['ai_confidence'], 
                                    row['location']
                                )
                            
                            st.markdown("### Clinical Assessment")
                            st.markdown(doctor_insights)
                        
                        st.markdown("### Your Diagnosis")
                        # Verification form
                        with st.form(f"verify_form_{row['id']}"):
                            doctor_diagnosis = st.text_input("Enter diagnosis", value=row['ai_diagnosis'], key=f"diag_{row['id']}")
                            
                            # Add common diagnoses as quick select
                            common_diagnoses = ["Plasmodium falciparum", "Plasmodium vivax", "No parasite detected"]
                            selected_diagnosis = st.selectbox(
                                "Or select common diagnosis", 
                                ["Select..."] + common_diagnoses,
                                key=f"common_diag_{row['id']}"
                            )
                            
                            if selected_diagnosis != "Select...":
                                doctor_diagnosis = selected_diagnosis
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                verified = st.checkbox("Verify diagnosis", value=bool(row['doctor_verified']), key=f"verify_{row['id']}")
                            with col2:
                                add_to_notes = st.checkbox("Add clinical notes", key=f"notes_check_{row['id']}")
                            
                            if add_to_notes:
                                clinical_notes = st.text_area("Clinical notes", key=f"notes_{row['id']}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                submit = st.form_submit_button("Submit Diagnosis", use_container_width=True)
                            with col2:
                                flag = st.form_submit_button("Flag for Second Opinion", use_container_width=True)
                            
                            if submit:
                                # Update the sample with doctor verification
                                success = database.update_sample_verification(
                                    row['id'], 
                                    doctor_diagnosis, 
                                    verified, 
                                    st.session_state.user_id
                                )
                                
                                if success:
                                    st.success("Diagnosis updated successfully!")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("Error updating diagnosis")
                            
                            if flag:
                                # This would typically flag for a second opinion in a real system
                                st.info("Sample flagged for second opinion review.")
    
    with tab2:
        st.subheader("Search Patients")
        
        search_term = st.text_input("Enter patient name")
        
        if search_term:
            # Search for patients
            patients = database.execute_read_query("""
                SELECT id, name, location, status
                FROM patients
                WHERE name LIKE ?
                ORDER BY name
            """, params=[f"%{search_term}%"])
            
            if not patients.empty:
                st.write(f"Found {len(patients)} patients")
                
                for _, patient in patients.iterrows():
                    with st.expander(f"Patient: {patient['name']} ({patient['location']})"):
                        # Get patient samples
                        samples = database.execute_read_query("""
                            SELECT s.id, s.image_path, s.ai_diagnosis, s.ai_confidence, 
                                   s.doctor_diagnosis, s.doctor_verified, s.date_diagnosed
                            FROM samples s
                            WHERE s.patient_id = ?
                            ORDER BY s.date_diagnosed DESC
                        """, params=[patient['id']])
                        
                        if not samples.empty:
                            st.write(f"Found {len(samples)} samples")
                            
                            for _, sample in samples.iterrows():
                                sample_col1, sample_col2 = st.columns([1, 2])
                                
                                with sample_col1:
                                    if os.path.exists(sample['image_path']):
                                        st.image(sample['image_path'], caption=f"Sample from {sample['date_diagnosed']}", width=200)
                                
                                with sample_col2:
                                    st.write(f"Date: {sample['date_diagnosed']}")
                                    st.write(f"AI Diagnosis: {sample['ai_diagnosis']} (Conf: {sample['ai_confidence']:.2f})")
                                    
                                    if sample['doctor_verified']:
                                        st.write(f"Doctor Diagnosis: {sample['doctor_diagnosis']}")
                                        st.success("✅ Verified")
                                    else:
                                        st.warning("⏳ Awaiting verification")
                        else:
                            st.info("No samples found for this patient")
            else:
                st.info(f"No patients found matching '{search_term}'")
    
    with tab3:
        st.subheader("My Verifications")
        
        # Get samples verified by this doctor
        my_verifications = database.get_doctor_verifications(st.session_state.user_id)
        
        if not my_verifications.empty:
            st.write(f"You have verified {len(my_verifications)} samples")
            
            # Display verifications as a table
            st.dataframe(
                my_verifications,
                column_config={
                    "name": "Patient",
                    "date_diagnosed": "Date",
                    "ai_diagnosis": "AI Diagnosis",
                    "doctor_diagnosis": "Your Diagnosis"
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Show statistics
            st.subheader("Your Verification Statistics")
            
            # Calculate agreement rate with AI
            agreement = (my_verifications['ai_diagnosis'] == my_verifications['doctor_diagnosis']).mean() * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Verifications", len(my_verifications))
            with col2:
                st.metric("Agreement with AI", f"{agreement:.1f}%")
            
            # Show most common diagnoses
            diagnoses_count = my_verifications['doctor_diagnosis'].value_counts()
            
            st.subheader("Your Most Common Diagnoses")
            if not diagnoses_count.empty:
                st.bar_chart(diagnoses_count)
        else:
            st.info("You haven't verified any samples yet")