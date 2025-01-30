import streamlit as st
import geopy.distance
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import joblib

# Load the dataset (replace with your actual dataset)
@st.cache_resource
def load_model():
    return joblib.load('crime_zone_model.pkl')

model = load_model()

# Define zone classification based on crime count
def classify_zone(crime_count):
    if crime_count >= 400:  # Threshold for "Red"
        return "Red"
    elif crime_count >= 200:  # Threshold for "Yellow"
        return "Yellow"
    else:
        return "Green"

# Get user's current geolocation
@st.cache_data
def get_user_location():
    try:
        # Automatically detect location using device IP
        from geocoder import ip
        user_location = ip("me").latlng  # [latitude, longitude]
        if not user_location:
            st.error("Unable to fetch location. Please check your connection or permissions.")
            return None
        return user_location
    except Exception as e:
        st.error(f"Error fetching location: {e}")
        return None

# Load data
zone_data = load_model()

# Add a "Zone" column based on crime classification
zone_data["Zone"] = zone_data["Total_Crime"].apply(classify_zone)

# Page title
st.title("Crime Safety Zone Detector")

# Get the user's geolocation
user_location = get_user_location()

if user_location:
    st.write(f"Your Current Location: Latitude: {user_location[0]}, Longitude: {user_location[1]}")

    # Calculate distances to all predefined zones
    zone_data["Distance"] = zone_data.apply(
        lambda row: geopy.distance.geodesic(user_location, (row["Latitude"], row["Longitude"])).kilometers, axis=1
    )

    # Find the closest zone
    closest_zone = zone_data.loc[zone_data["Distance"].idxmin()]

    # Display results
    st.write(f"Closest Zone: **{closest_zone['Area']}**")
    st.write(f"Zone Type: **{closest_zone['Zone']}**")
    st.write(f"Crime Count in the Zone: **{closest_zone['Total_Crime']}**")

    if closest_zone["Zone"] == "Red":
        st.error("You are in an **UNSAFE (Red)** zone. Please take precautions.")
    elif closest_zone["Zone"] == "Yellow":
        st.warning("You are in a **Moderate Risk (Yellow)** zone. Stay alert.")
    else:
        st.success("You are in a **SAFE (Green)** zone.")

    # Display map with the user's location and closest zone
    crime_map = folium.Map(location=user_location, zoom_start=13)

    # Add user's location
    folium.Marker(user_location, popup="Your Location", icon=folium.Icon(color="blue")).add_to(crime_map)

    # Add all predefined zones to the map
    for _, row in zone_data.iterrows():
        folium.Marker(
            [row["Latitude"], row["Longitude"]],
            popup=f"{row['Area']} ({row['Zone']}) - {row['Total_Crime']} Crimes",
            icon=folium.Icon(color="red" if row["Zone"] == "Red" else
                             "yellow" if row["Zone"] == "Yellow" else "green")
        ).add_to(crime_map)

    # Show the map
    st_folium(crime_map, width=700)

# Note: To test locally, your browser must allow location access.
