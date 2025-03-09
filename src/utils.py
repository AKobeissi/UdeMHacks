import os
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
from datetime import datetime
from PIL import Image
import io
from config import UPLOAD_DIR
from folium.plugins import HeatMap

def save_uploaded_image(uploaded_file, patient_id):
    """Save an uploaded image to the filesystem"""
    try:
        # Create upload directory if it doesn't exist
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Create a unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_path = f"{UPLOAD_DIR}/sample_{patient_id}_{timestamp}.jpg"
        
        # Save the image
        image = Image.open(io.BytesIO(uploaded_file.getvalue()))
        image.save(image_path)
        
        return image_path, image
    except Exception as e:
        st.error(f"Error saving image: {str(e)}")
        return None, None

def create_heatmap(df):
    """Create a folium heatmap from dataframe with lat/long coordinates and case density"""
    if df.empty or 'latitude' not in df.columns or 'longitude' not in df.columns:
        return None
    
    # Use doctor_diagnosis if verified, otherwise use ai_diagnosis
    df['final_diagnosis'] = df.apply(
        lambda x: x['doctor_diagnosis'] if x['doctor_verified'] == 1 else x['ai_diagnosis'], 
        axis=1
    )
    
    # Count diagnoses by location
    location_counts = df.groupby(['latitude', 'longitude']).size().reset_index(name='count')

    # Check if we have valid data
    if location_counts.empty:
        return None
        
    # Create map centered on average coordinates
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

    # Convert data to heatmap-compatible format
    heat_data = location_counts[['latitude', 'longitude', 'count']].values.tolist()

    # Add heatmap layer
    HeatMap(heat_data, min_opacity=0.2, radius=15, blur=10, max_zoom=12).add_to(m)
    
    return m

def generate_insights_with_gemini(prompt):
    """Generate insights using Google Gemini API"""
    try:
        from google import generativeai as genai
        import os
        
        # Try different ways to get the API key
        # api_key = os.getenv("GEMINI_API_KEY")
        api_key = "AIzaSyA1ctEo4qW3FCTXGbBRK57-0imsVRoqiBA"
        
        # Check if file-based API key exists as fallback
        if not api_key:
            try:
                # Try to load from a config file
                api_key_path = os.path.join(os.path.dirname(__file__), 'gemini_api_key.txt')
                if os.path.exists(api_key_path):
                    with open(api_key_path, 'r') as f:
                        api_key = f.read().strip()
            except Exception as e:
                st.warning(f"Couldn't load API key from file: {str(e)}")
        
        # If still no API key, show a more explicit message
        if not api_key:
            st.warning("Gemini API key not found. Please set GEMINI_API_KEY environment variable or create a gemini_api_key.txt file.")
            return None
        
        # Configure the API
        genai.configure(api_key=api_key)
        
        try:
            # Use the latest model
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt)
            return response.text
        except Exception as model_error:
            st.warning(f"Error with Gemini model: {str(model_error)}")
            
            # Try alternate model as fallback
            try:
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                return response.text
            except Exception as fallback_error:
                st.warning(f"Error with fallback model: {str(fallback_error)}")
                return None
            
    except ImportError:
        st.warning("Google Generative AI package not installed. Run: pip install google-generativeai")
        return None
    except Exception as e:
        st.warning(f"Gemini API error: {str(e)}")
        return None

def generate_patient_insights(diagnosis, confidence):
    """Generate patient-friendly insights about a diagnosis"""
    prompt = f"""
    Generate a brief summary for a patient diagnosed with {diagnosis} (confidence: {confidence:.2f}).
    Include:
    1. What this parasite is
    2. Common symptoms
    3. Treatment options
    4. Preventive measures
    Keep it simple, factual, and reassuring.
    Format the output with markdown.
    """
    
    # Try to get insights from Gemini
    gemini_response = generate_insights_with_gemini(prompt)
    if gemini_response:
        print("gemini response")
        return gemini_response
    
    # Fallback to predefined insights if Gemini is not available
    basic_insights = {
        "Plasmodium": """
        **What is it?** Plasmodium is a parasite that causes malaria, a serious blood disease.
        
        **Common symptoms:** Fever, chills, headache, muscle aches, and fatigue. Symptoms often come in cycles.
        
        **Treatment:** Anti-malarial medications prescribed by your doctor. Complete the full course of treatment.
        
        **Prevention:** Use bed nets, apply insect repellent, wear protective clothing, and take preventive medication if traveling to malaria-endemic areas.
        """,
        
        "Babesia": """
        **What is it?** Babesia is a parasite that infects red blood cells, causing babesiosis, an infection similar to malaria.
        
        **Common symptoms:** Fever, fatigue, headache, chills, and muscle aches. Some people may have no symptoms.
        
        **Treatment:** Combination antibiotic therapy prescribed by your doctor. Most people recover completely.
        
        **Prevention:** Avoid tick-infested areas, use insect repellent, wear protective clothing, and check for ticks after outdoor activities.
        """,
        
        "Leishmania": """
        **What is it?** Leishmania is a parasite that causes leishmaniasis, affecting the skin, mucous membranes, or internal organs.
        
        **Common symptoms:** Skin sores, ulcers, fever, weight loss, and enlarged spleen/liver in more severe cases.
        
        **Treatment:** Medications prescribed by your doctor. Treatment may be long but is usually effective.
        
        **Prevention:** Use bed nets and insect repellent, wear protective clothing, and avoid sandfly habitats.
        """,
        
        "Leukocyte": """
        **What is it?** Leukocytes are white blood cells that help fight infection. This is not a parasite but important cells in your immune system.
        
        **Common symptoms:** Not applicable - these are normal cells in your body.
        
        **Treatment:** Not required unless there's an abnormality in your white blood cell count, which would be addressed by your doctor.
        
        **Prevention:** Maintain general good health to support your immune system.
        """,
        
        "RBC": """
        **What is it?** RBCs (Red Blood Cells) are normal cells that carry oxygen in your blood. This is not a parasite.
        
        **Common symptoms:** Not applicable - these are normal cells in your body.
        
        **Treatment:** Not required unless there's an abnormality in your red blood cell count, which would be addressed by your doctor.
        
        **Prevention:** Maintain a healthy diet with adequate iron to support red blood cell production.
        """,
        
        "Toxoplasma": """
        **What is it?** Toxoplasma is a parasite that causes toxoplasmosis, usually mild but can be serious in certain individuals.
        
        **Common symptoms:** Often no symptoms, but may include flu-like symptoms. Can be serious in pregnant women and immunocompromised individuals.
        
        **Treatment:** Usually not needed for healthy people. Medications prescribed for at-risk groups or severe cases.
        
        **Prevention:** Cook meat thoroughly, wash fruits and vegetables, wear gloves when gardening, and avoid contact with cat feces.
        """,
        
        "Trichomonad": """
        **What is it?** Trichomonads are parasites that can cause infections in different parts of the body, including trichomoniasis, a common sexually transmitted infection.
        
        **Common symptoms:** In trichomoniasis: abnormal discharge, itching, and discomfort during urination. Some people have no symptoms.
        
        **Treatment:** Prescription antibiotic medication. Both partners need treatment to prevent reinfection.
        
        **Prevention:** Practice safe sex, limit sexual partners, and complete all medication as prescribed if infected.
        """,
        
        "Trypanosome": """
        **What is it?** Trypanosomes are parasites that cause diseases such as Chagas disease and sleeping sickness.
        
        **Common symptoms:** Fever, fatigue, body aches, and in Chagas disease, possible heart and digestive problems in chronic cases.
        
        **Treatment:** Antiparasitic drugs prescribed by your doctor, especially effective in early stages.
        
        **Prevention:** Avoid insect vectors (kissing bugs, tsetse flies), use bed nets and insect repellent, and improve housing conditions to prevent insect infestation.
        """,
        
        "No parasite detected": """
        **Good news!** No parasites were detected in your sample.
        
        **Next steps:** 
        - If you're still experiencing symptoms, consult with your doctor.
        - Consider other potential causes for your symptoms.
        - Continue practicing good hygiene and preventive measures.
        """
    }
    
    # Default message for parasites not in the dictionary
    default_insight = f"""
    **What is it?** {diagnosis} is a type of microorganism that can potentially cause infection.
    
    **Common symptoms:** May include fever, fatigue, and other symptoms specific to this organism.
    
    **Treatment:** Medications as prescribed by your doctor.
    
    **Prevention:** Practice good hygiene, avoid exposure to potential sources of infection, and follow your doctor's advice.
    """
    
    return basic_insights.get(diagnosis, default_insight)

def generate_doctor_insights(diagnosis, confidence, location):
    """Generate doctor-specific insights about a diagnosis"""
    prompt = f"""
    Generate a technical summary for a doctor reviewing a case of {diagnosis} (AI confidence: {confidence:.2f}).
    Patient location: {location}
    Include:
    1. Clinical significance of this finding
    2. Recommended confirmatory tests
    3. Treatment protocol recommendations
    4. Regional epidemiological considerations
    5. Follow-up recommendations
    Be concise and evidence-based.
    Format the output with markdown.
    """
    
    # Try to get insights from Gemini
    gemini_response = generate_insights_with_gemini(prompt)
    if gemini_response:
        return gemini_response
    
    # Fallback to predefined insights if Gemini is not available
    doctor_insights = {
        "Plasmodium": f"""
        **Clinical Significance:** Potential malaria infection requiring prompt evaluation; species identification critical for treatment decisions.
        
        **Confirmatory Tests:** Thick and thin blood smears, rapid diagnostic tests (RDTs), PCR for species identification.
        
        **Treatment Protocol:** 
        - P. falciparum: Artemisinin-based combination therapy (ACT)
        - P. vivax/ovale: Chloroquine plus primaquine after G6PD testing
        - P. malariae/knowlesi: Chloroquine if no resistance suspected
        
        **Epidemiological Considerations:** Check local resistance patterns in {location}; evaluate potential exposure history.
        
        **Follow-up:** Monitor parasite clearance, clinical symptoms, and treatment response; consider screening household members in endemic areas.
        """,
        
        "Babesia": f"""
        **Clinical Significance:** Babesiosis, potentially serious in asplenic, elderly, or immunocompromised patients.
        
        **Confirmatory Tests:** Peripheral blood smear with Giemsa stain, FISH, PCR for species identification (B. microti most common).
        
        **Treatment Protocol:** 
        - Mild to moderate: Atovaquone plus azithromycin for 7-10 days
        - Severe: Clindamycin plus quinine; consider exchange transfusion for high parasitemia (>10%)
        
        **Epidemiological Considerations:** Most common in northeastern U.S. but check for cases in {location}; assess tick exposure history.
        
        **Follow-up:** Monitor hemoglobin, platelets, liver function; patients may remain PCR-positive despite clinical improvement.
        """,
        
        "Leishmania": f"""
        **Clinical Significance:** Leishmaniasis with clinical presentation dependent on species (cutaneous, mucocutaneous, or visceral).
        
        **Confirmatory Tests:** Tissue biopsy with histopathology, culture, PCR for species identification.
        
        **Treatment Protocol:** 
        - Cutaneous: Local therapy for limited disease, systemic for extensive/complex cases
        - Visceral: Liposomal amphotericin B, miltefosine, or pentavalent antimonials
        
        **Epidemiological Considerations:** Determine endemic species in {location}; check travel history if not endemic.
        
        **Follow-up:** Monitor treatment response clinically, consider repeat testing to confirm cure; assess for relapse in visceral disease.
        """,
        
        "Leukocyte": f"""
        **Clinical Significance:** Presence of leukocytes is normal; assess for abnormal morphology or quantity that may indicate inflammation or infection.
        
        **Confirmatory Tests:** Complete blood count with differential, peripheral blood smear review.
        
        **Management Recommendations:** 
        - Determine if leukocytosis/leukopenia is present
        - Assess for reactive changes suggestive of viral or bacterial infection
        - Rule out hematologic malignancy if atypical cells present
        
        **Additional Considerations:** Correlation with clinical presentation and other laboratory findings necessary.
        
        **Follow-up:** Repeat CBC if clinically indicated; consider bone marrow evaluation if persistent abnormalities.
        """,
        
        "RBC": f"""
        **Clinical Significance:** Red blood cells identified; assess for parasites, morphologic abnormalities, or inclusions.
        
        **Confirmatory Tests:** Complete blood count, peripheral smear review, reticulocyte count, hemoglobin electrophoresis if indicated.
        
        **Management Recommendations:** 
        - Verify no intracellular parasites present
        - Assess for anemia, morphologic abnormalities
        - Consider underlying conditions affecting RBC morphology
        
        **Additional Considerations:** RBC parameters may indicate nutritional deficiencies or hemoglobinopathies endemic to {location}.
        
        **Follow-up:** Guided by clinical presentation and initial test results.
        """,
        
        "Toxoplasma": f"""
        **Clinical Significance:** Toxoplasma gondii infection; clinical significance varies from asymptomatic to severe based on immune status and pregnancy.
        
        **Confirmatory Tests:** Serologic testing (IgG/IgM), PCR for immunocompromised patients, amniocentesis PCR for suspected congenital infection.
        
        **Treatment Protocol:** 
        - Immunocompetent: Usually no treatment needed unless severe or persistent
        - Pregnant: Spiramycin or pyrimethamine-sulfadiazine with leucovorin based on gestational age and confirmed fetal infection
        - Immunocompromised: Pyrimethamine-sulfadiazine with leucovorin
        
        **Epidemiological Considerations:** Assess exposure to cats, consumption of undercooked meat, gardening activities in {location}.
        
        **Follow-up:** Serology monitoring for pregnancy; neuroimaging for CNS involvement; ophthalmologic evaluation.
        """,
        
        "Trichomonad": f"""
        **Clinical Significance:** Typically Trichomonas vaginalis if genital, or other species based on specimen source.
        
        **Confirmatory Tests:** Wet mount microscopy, culture, NAAT, or point-of-care antigen testing.
        
        **Treatment Protocol:** 
        - T. vaginalis: Metronidazole 2g single dose or 500mg BID for 7 days
        - Treat sexual partners concurrently
        - Consider alternative regimens for metronidazole resistance
        
        **Epidemiological Considerations:** Screen for other STIs; assess prevalence in {location}.
        
        **Follow-up:** Only if symptoms persist; test-of-cure not routinely recommended except in pregnancy.
        """,
        
        "Trypanosome": f"""
        **Clinical Significance:** Trypanosomiasis; determine if T. cruzi (Chagas) or T. brucei (sleeping sickness) based on morphology and geography.
        
        **Confirmatory Tests:** Giemsa-stained blood smears, PCR, serology (especially for chronic Chagas).
        
        **Treatment Protocol:** 
        - T. cruzi (acute): Benznidazole or nifurtimox
        - T. cruzi (chronic): Evaluate cardiac/GI involvement, consider treatment
        - T. brucei: Pentamidine or suramin (early), melarsoprol or eflornithine (late)
        
        **Epidemiological Considerations:** Assess travel/residence in endemic areas; determine vector exposure in {location}.
        
        **Follow-up:** Serial monitoring for parasitemia clearance; cardiac evaluation in Chagas; CSF examination in African trypanosomiasis.
        """,
        
        "No parasite detected": f"""
        **Clinical Significance:** Negative finding; rule out sampling error or low parasitemia.
        
        **Additional Testing:** 
        - Consider serial blood smears if clinical suspicion remains high
        - Molecular testing (PCR) for increased sensitivity
        - Serological testing for chronic infections
        - Alternate specimen types based on suspected pathogen
        
        **Management Considerations:** 
        - Evaluate for non-parasitic causes of symptoms
        - Consider empiric treatment if high clinical suspicion despite negative results
        
        **Epidemiological Context:** Review endemic parasitic diseases in {location} and recent outbreaks.
        
        **Follow-up:** Based on clinical presentation; repeat testing if symptoms persist.
        """
    }
    
    # Default message for parasites not in the dictionary
    default_insight = f"""
    **Clinical Significance:** {diagnosis} identified; requires clinical correlation.
    
    **Confirmatory Tests:** Specialized microscopy, PCR, or serology depending on the specific organism.
    
    **Treatment Protocol:** Consult current guidelines specific to {diagnosis}.
    
    **Epidemiological Considerations:** Determine if endemic or emerging in {location}; evaluate exposure history.
    
    **Follow-up:** Monitor clinical response to treatment; consider repeat testing to confirm clearance if appropriate.
    """
    
    return doctor_insights.get(diagnosis, default_insight)

def add_sample_data():
    """Add sample data for demonstration purposes"""
    import database
    
    try:
        # Sample data with the updated parasite types
        sample_data = [
            ("John Doe", "Nairobi", -1.286389, 36.817223, "Plasmodium"),
            ("Jane Smith", "Mombasa", -4.043740, 39.668208, "Babesia"),
            ("Alice Johnson", "Kisumu", -0.102400, 34.761700, "Leishmania"),
            ("Bob Brown", "Nakuru", -0.303099, 36.080025, "No parasite detected"),
            ("Charlie Davis", "Eldoret", 0.520271, 35.269680, "Trypanosome")
        ]
        
        for name, location, lat, lon, diagnosis in sample_data:
            # Create patient
            patient_id = database.create_patient(name, location, lat, lon, 1)  # user_id=1 (admin)
            
            # Create sample with AI diagnosis
            database.save_sample(patient_id, "uploads/sample_placeholder.jpg", diagnosis, 0.85)
        
        return True
    except Exception as e:
        st.error(f"Error adding sample data: {str(e)}")
        return False