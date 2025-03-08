import streamlit as st
import os
import database
import auth
from styles import formatted_header, info_box
from utils import generate_patient_insights

@auth.role_required(["patient", "admin"])
def display_patient_portal():
    """Display the patient portal with test results and health information"""
    formatted_header("Patient Portal")
    info_box("View your test results and health information")
    
    # For admin, show patient selection
    if st.session_state.user_role == "admin":
        patients = database.execute_read_query("SELECT id, name FROM patients ORDER BY name")
        
        if patients.empty:
            st.info("No patients registered yet")
            return
        
        selected_patient = st.selectbox("Select Patient", patients['name'])
        patient_id = patients[patients['name'] == selected_patient]['id'].values[0]
    else:
        # For patient, get their own results
        # First, get the patient record linked to this user
        patient_record = database.get_patient_by_user_id(st.session_state.user_id)
        
        if not patient_record:
            st.info("No patient record associated with your account. Please contact your healthcare provider.")
            return
        
        patient_id = patient_record[0]
    
    # Get patient's results
    results = database.get_patient_samples(patient_id)
    
    if results.empty:
        st.info("No test results available yet")
        
        # Add some helpful information
        st.markdown("""
        ### What to Expect
        
        Once your healthcare provider uploads and analyzes your sample:
        
        1. You'll see your test results here
        2. A doctor will review the AI diagnosis
        3. You'll receive information about your diagnosis and next steps
        
        If you have questions, please contact your healthcare provider.
        """)
    else:
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Test Results", "Health Information"])
        
        with tab1:
            # Profile summary
            st.subheader(f"Patient: {results.iloc[0]['name']}")
            st.write(f"Location: {results.iloc[0]['location']}")
            
            # Display test results in a more user-friendly format
            for i, row in results.iterrows():
                with st.expander(f"Test from {row['date_diagnosed'].split()[0]}", expanded=i==0):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if os.path.exists(row['image_path']):
                            st.image(row['image_path'], caption="Sample Image", width=300)
                        else:
                            st.error("Image file not found")
                    
                    with col2:
                        # Use doctor diagnosis if verified, otherwise use AI diagnosis
                        final_diagnosis = row['doctor_diagnosis'] if row['doctor_verified'] else row['ai_diagnosis']
                        verification_status = "Verified by doctor" if row['doctor_verified'] else "Awaiting doctor verification"
                        
                        st.write(f"Diagnosis: **{final_diagnosis}**")
                        
                        if row['doctor_verified']:
                            st.success(f"✅ {verification_status}")
                        else:
                            st.warning(f"⏳ {verification_status}")
                        
                        if row['doctor_verified']:
                            # Generate patient-friendly insights
                            insights = generate_patient_insights(final_diagnosis, row['ai_confidence'])
                            
                            st.markdown("### What this means for you:")
                            st.markdown(insights)
                        else:
                            st.info("Your results are still being reviewed by a doctor. Please check back later.")
        
        with tab2:
            st.subheader("Health Resources")
            
            # Display general health information based on location
            st.markdown(f"""
            ### Health Information for {results.iloc[0]['location']}
            
            #### Common Parasitic Diseases in Your Region
            
            Based on your location, here are some common parasitic diseases and prevention measures:
            
            1. **Malaria**
               - Use bed nets treated with insecticide
               - Apply insect repellent
               - Take antimalarial medication if recommended
            
            2. **Intestinal Parasites**
               - Wash hands thoroughly before eating
               - Drink clean, treated water
               - Properly wash fruits and vegetables
            
            3. **Schistosomiasis**
               - Avoid swimming in freshwater where the disease is common
               - Boil bathing water if from an unsafe source
            
            #### When to Seek Medical Help
            
            Contact your healthcare provider if you experience:
            - Persistent fever
            - Unexplained weight loss
            - Severe diarrhea or abdominal pain
            - Unusual skin changes or rashes
            """)