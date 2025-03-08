import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import database
import time
import auth
from styles import formatted_header, info_box
from utils import create_heatmap, add_sample_data

@auth.role_required(["doctor", "technician", "patient", "admin"])
def display_outbreak_map():
    """Display the parasite outbreak map and statistics"""
    formatted_header("Parasite Outbreak Map")
    info_box("Geographic distribution of parasitic diseases in your region")
    
    # Create tabs for Map and Statistics
    tab1, tab2 = st.tabs(["Map View", "Statistics"])
    
    with tab1:
        # Get data for the outbreak map
        outbreak_data = database.get_outbreak_data()
        
        # Create map
        m = create_heatmap(outbreak_data)
        
        if m:
            folium_static(m, width=800)
            
            # Add map legend and explanation
            st.markdown("""
            ### Map Legend
            
            - Red circles indicate detected parasite cases
            - Larger circles represent areas with more cases
            - Click on circles to see details about the parasites detected
            """)
        else:
            st.info("No geographic data available yet to create the map")
            
            # Add sample data button for demonstration
            if st.button("Add Sample Data for Demonstration"):
                if add_sample_data():
                    st.success("Sample data added for demonstration")
                    time.sleep(1)
                    st.rerun()
    
    with tab2:
        # Show statistics
        st.subheader("Outbreak Statistics")
        
        stats = database.get_outbreak_statistics()
        
        if not stats.empty:
            # Create a horizontal bar chart
            st.bar_chart(stats.set_index('diagnosis'))
            
            # Show a data table with percentages
            total_cases = stats['count'].sum()
            stats['percentage'] = (stats['count'] / total_cases * 100).round(1)
            
            st.dataframe(
                stats,
                column_config={
                    "diagnosis": "Parasite",
                    "count": "Number of Cases",
                    "percentage": st.column_config.NumberColumn(
                        "% of Total",
                        format="%.1f%%"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Add educational information for patients
            if st.session_state.user_role == "patient":
                st.subheader("Understanding the Data")
                st.markdown("""
                This chart shows the distribution of parasites in your region. The information can help you:
                
                * Understand which parasites are common in your area
                * Take appropriate preventive measures
                * Stay informed about potential outbreak risks
                
                ### Preventive Measures
                
                * **Mosquito protection**: Use bed nets and repellents
                * **Safe water**: Drink only treated or bottled water
                * **Food safety**: Wash fruits and vegetables thoroughly
                * **Personal hygiene**: Wash hands frequently
                """)
        else:
            st.info("No diagnostic data available yet")