import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import os
import sqlite3
from datetime import datetime
from google import genai
import io

# Set up database
def setup_database():
    conn = sqlite3.connect('parasite_diagnosis.db')
    c = conn.cursor()
    
    # Create tables if they don't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY,
        name TEXT,
        location TEXT,
        latitude REAL,
        longitude REAL,
        date_collected DATETIME,
        status TEXT
    )
    ''')
    
    c.execute('''
    CREATE TABLE IF NOT EXISTS samples (
        id INTEGER PRIMARY KEY,
        patient_id INTEGER,
        image_path TEXT,
        ai_diagnosis TEXT,
        ai_confidence REAL,
        doctor_diagnosis TEXT,
        doctor_verified INTEGER DEFAULT 0,
        date_diagnosed DATETIME,
        FOREIGN KEY (patient_id) REFERENCES patients (id)
    )
    ''')
    
    conn.commit()
    return conn, c

# Load the model
@st.cache_resource
def load_model():
    model = resnet50(pretrained=True)
    num_classes = 10  # Adjust based on number of parasite classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Load trained weights if available
    if os.path.exists('parasite_model.pth'):
        model.load_state_dict(torch.load('parasite_model.pth', map_location=torch.device('cpu')))
    
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# Make predictions
def predict(model, image):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probabilities[0][predicted[0]].item()
    
    # Placeholder for actual class names
    class_names = [
        "Plasmodium falciparum", 
        "Plasmodium vivax", 
        "Plasmodium malariae", 
        "Plasmodium ovale", 
        "Trypanosoma", 
        "Leishmania", 
        "Schistosoma", 
        "Filariasis", 
        "Entamoeba histolytica", 
        "No parasite detected"
    ]
    
    return class_names[predicted[0]], confidence

# Generate insights using Google Gemini API
def generate_insights(diagnosis, confidence):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "API key not found. Please set GEMINI_API_KEY environment variable."
        
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        Generate a brief summary for a patient diagnosed with {diagnosis} (confidence: {confidence:.2f}).
        Include:
        1. What this parasite is
        2. Common symptoms
        3. Treatment options
        4. Preventive measures
        Keep it simple, factual, and reassuring.
        """
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        return response.text
    except Exception as e:
        return f"Error generating insights: {str(e)}"

# Generate doctor insights using Google Gemini API
def generate_doctor_insights(diagnosis, confidence, patient_data):
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "API key not found. Please set GEMINI_API_KEY environment variable."
        
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        Generate a technical summary for a doctor reviewing a case of {diagnosis} (AI confidence: {confidence:.2f}).
        Patient location: {patient_data['location']}
        Include:
        1. Clinical significance of this finding
        2. Recommended confirmatory tests
        3. Treatment protocol recommendations
        4. Regional epidemiological considerations
        5. Follow-up recommendations
        Be concise and evidence-based.
        """
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        
        return response.text
    except Exception as e:
        return f"Error generating doctor insights: {str(e)}"

# Create heatmap
def create_heatmap(conn):
    df = pd.read_sql_query("""
    SELECT p.latitude, p.longitude, s.ai_diagnosis, s.doctor_diagnosis, s.doctor_verified
    FROM patients p
    JOIN samples s ON p.id = s.patient_id
    WHERE p.latitude IS NOT NULL AND p.longitude IS NOT NULL
    """, conn)
    
    # Use doctor_diagnosis if verified, otherwise use ai_diagnosis
    df['final_diagnosis'] = df.apply(
        lambda x: x['doctor_diagnosis'] if x['doctor_verified'] == 1 else x['ai_diagnosis'], 
        axis=1
    )
    
    # Count diagnoses by location
    location_counts = df.groupby(['latitude', 'longitude', 'final_diagnosis']).size().reset_index(name='count')
    
    # Create map centered on average coordinates
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)
    
    # Add markers for each location with diagnosis info
    for _, row in location_counts.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=int(5 + row['count'] * 2),  # Size based on count
            popup=f"{row['final_diagnosis']}: {row['count']} cases",
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.6
        ).add_to(m)
    
    return m

# Main Streamlit app
def main():
    st.set_page_config(page_title="Parasite Diagnosis System", layout="wide")
    
    conn, c = setup_database()
    model = load_model()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Technician Portal", "Doctor Portal", "Patient Portal", "Outbreak Map"])
    
    # Technician Portal
    if page == "Technician Portal":
        st.title("Technician Portal - Upload Samples")
        
        with st.form("sample_upload_form"):
            patient_name = st.text_input("Patient Name")
            location = st.text_input("Location")
            latitude = st.number_input("Latitude", -90.0, 90.0, format="%.6f")
            longitude = st.number_input("Longitude", -180.0, 180.0, format="%.6f")
            uploaded_file = st.file_uploader("Upload microscope image", type=["jpg", "jpeg", "png"])
            
            submit_button = st.form_submit_button("Submit Sample")
            
            if submit_button and uploaded_file is not None and patient_name and location:
                # Save patient info
                c.execute("""
                INSERT INTO patients (name, location, latitude, longitude, date_collected, status)
                VALUES (?, ?, ?, ?, ?, ?)
                """, (patient_name, location, latitude, longitude, datetime.now(), "Pending"))
                conn.commit()
                patient_id = c.lastrowid
                
                # Save image
                image_bytes = uploaded_file.getvalue()
                image = Image.open(io.BytesIO(image_bytes))
                
                # Create directory if it doesn't exist
                os.makedirs("uploads", exist_ok=True)
                image_path = f"uploads/sample_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                image.save(image_path)
                
                # Process image and get prediction
                processed_image = preprocess_image(image)
                diagnosis, confidence = predict(model, processed_image)
                
                # Save sample info
                c.execute("""
                INSERT INTO samples (patient_id, image_path, ai_diagnosis, ai_confidence, date_diagnosed)
                VALUES (?, ?, ?, ?, ?)
                """, (patient_id, image_path, diagnosis, confidence, datetime.now()))
                conn.commit()
                
                st.success(f"Sample uploaded successfully! AI Diagnosis: {diagnosis} (Confidence: {confidence:.2f})")
                
                # Flag low confidence predictions
                if confidence < 0.7:
                    st.warning("Low confidence prediction. Flagged for doctor review.")
    
    # Doctor Portal
    elif page == "Doctor Portal":
        st.title("Doctor Portal - Review Diagnoses")
        
        # Query unverified samples
        samples = pd.read_sql_query("""
        SELECT s.id, p.name, p.location, s.image_path, s.ai_diagnosis, s.ai_confidence, s.doctor_verified
        FROM samples s
        JOIN patients p ON s.patient_id = p.id
        ORDER BY s.ai_confidence ASC, s.date_diagnosed DESC
        """, conn)
        
        if samples.empty:
            st.info("No samples to review.")
        else:
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
                        
                        # Get patient data for GenAI insights
                        patient_data = {
                            'location': row['location']
                        }
                        
                        doctor_insights = generate_doctor_insights(row['ai_diagnosis'], row['ai_confidence'], patient_data)
                        st.markdown("### Clinical Assessment")
                        st.markdown(doctor_insights)
                        
                        st.markdown("### Your Diagnosis")
                        # Verification form
                        with st.form(f"verify_form_{row['id']}"):
                            doctor_diagnosis = st.text_input("Enter diagnosis", value=row['ai_diagnosis'], key=f"diag_{row['id']}")
                            verified = st.checkbox("Verify diagnosis", value=bool(row['doctor_verified']), key=f"verify_{row['id']}")
                            
                            if st.form_submit_button("Submit"):
                                c.execute("""
                                UPDATE samples 
                                SET doctor_diagnosis = ?, doctor_verified = ? 
                                WHERE id = ?
                                """, (doctor_diagnosis, 1 if verified else 0, row['id']))
                                conn.commit()
                                st.success("Diagnosis updated successfully!")
    
    # Patient Portal
    elif page == "Patient Portal":
        st.title("Patient Portal - View Results")
        
        patient_name = st.text_input("Enter patient name to view results")
        
        if patient_name:
            # Query patient results
            results = pd.read_sql_query("""
            SELECT p.id, p.name, p.location, s.ai_diagnosis, s.doctor_diagnosis, s.doctor_verified, s.ai_confidence
            FROM patients p
            JOIN samples s ON p.id = s.patient_id
            WHERE p.name LIKE ?
            ORDER BY s.date_diagnosed DESC
            """, conn, params=[f"%{patient_name}%"])
            
            if results.empty:
                st.info(f"No results found for patient: {patient_name}")
            else:
                for _, row in results.iterrows():
                    with st.expander(f"Results for {row['name']}"):
                        # Use doctor diagnosis if verified, otherwise use AI diagnosis
                        final_diagnosis = row['doctor_diagnosis'] if row['doctor_verified'] else row['ai_diagnosis']
                        verification_status = "Verified by doctor" if row['doctor_verified'] else "Awaiting doctor verification"
                        
                        st.write(f"Diagnosis: {final_diagnosis}")
                        st.write(f"Status: {verification_status}")
                        
                        if row['doctor_verified']:
                            # Generate patient-friendly insights
                            insights = generate_insights(final_diagnosis, row['ai_confidence'])
                            st.markdown("### What this means for you:")
                            st.markdown(insights)
                        else:
                            st.info("Your results are still being reviewed by a doctor. Please check back later.")
    
    # Outbreak Map
    elif page == "Outbreak Map":
        st.title("Parasite Outbreak Map")
        
        # Create map
        m = create_heatmap(conn)
        folium_static(m)
        
        # Show statistics
        st.subheader("Outbreak Statistics")
        
        stats = pd.read_sql_query("""
        SELECT 
            CASE 
                WHEN s.doctor_verified = 1 THEN s.doctor_diagnosis 
                ELSE s.ai_diagnosis 
            END as diagnosis,
            COUNT(*) as count
        FROM samples s
        GROUP BY diagnosis
        ORDER BY count DESC
        """, conn)
        
        st.bar_chart(stats.set_index('diagnosis'))
    
    conn.close()

if __name__ == "__main__":
    main()