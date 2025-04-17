import pandas as pd
import streamlit as st
import geocoder  # For getting the user's current location
from sklearn.tree import DecisionTreeClassifier
import folium
from folium.plugins import HeatMap
import joblib

# Load the dataset and preprocess
df = pd.read_csv('crime_data_delhi_with_coordinates.csv')

# Sum up the relevant crime columns to get the total crime score
crime_columns = [
    'Crime on Streets', 'Street Theft', 'Molestation on Roads',
    'Harassment Cases', 'Eve Teasing', 'Kidnapping & Abduction of Women',
    'Acid Attack', 'Attempt to Acid Attack', 'Murder with Rape/Gang Rape'
]
df['Total_Crime'] = df[crime_columns].sum(axis=1)

# Define the thresholds for the zones based on crime data
red_zone_threshold = df['Total_Crime'].quantile(0.66)  # Top 33% (Red)
green_zone_threshold = df['Total_Crime'].quantile(0.33)  # Bottom 33% (Green)

# Create a new column 'Zone' based on crime thresholds
def categorize_zone(row):
    if row['Total_Crime'] >= red_zone_threshold:
        return 'Red'      # Red Zone: Top 33%
    elif row['Total_Crime'] <= green_zone_threshold:
        return 'Green'    # Green Zone: Bottom 33%
    else:
        return 'Yellow'   # Yellow Zone: Middle 34%

# Apply the categorization function to the dataframe
df['Zone'] = df.apply(categorize_zone, axis=1)

# Train a Decision Tree Classifier
X = df[['Total_Crime']]
y = df['Zone']
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Save the model for later use in the app
joblib.dump(model, 'crime_zone_model.pkl')

# Streamlit App UI
st.title("Crime Zone Detection Based on Your Location")

# Hardcoded user location for demonstration
user_lat = 28.7496585 # Latitude
user_lng = 77.111702  # Longitude

# Display user's location in latitude/longitude format
st.write(f"Your current location (Latitude, Longitude): {user_lat}, {user_lng}")

# Get a human-readable address using reverse geocoding
reverse = geocoder.osm([user_lat, user_lng], method='reverse')
user_address = reverse.address if reverse and reverse.address else "Address not found"


# Create a map centered around the user's location
delhi_map = folium.Map(location=[user_lat, user_lng], zoom_start=11)

# Add marker for the user's location
folium.Marker(
    location=[user_lat, user_lng],
    popup="Your Current Location",
    icon=folium.Icon(color="blue", icon="user", prefix="fa"),
    draggable=False
).add_to(delhi_map)

# Find the nearest data point based on latitude and longitude
def find_nearest_zone(user_lat, user_lng, df):
    min_distance = float('inf')
    nearest_zone = ''
    
    for index, row in df.iterrows():
        # Calculate the distance (using simple Euclidean distance for simplicity)
        distance = ((row['Latitude'] - user_lat)**2 + (row['Longitude'] - user_lng)**2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            nearest_zone = row['Zone']
            
    return nearest_zone

# Get the predicted zone based on the nearest point
predicted_zone = find_nearest_zone(user_lat, user_lng, df)

# Display the result
st.write(f"The zone based on your location is: **{predicted_zone}**")

# Add the heatmap to the map
zone_intensity = {
    'Red': 3,       # Higher intensity for Red zone
    'Yellow': 2,    # Medium intensity for Yellow zone
    'Green': 1      # Lower intensity for Green zone
}

heatmap_data = []

# Add heatmap data based on the zone intensity
for index, row in df.iterrows():
    # Ensure both Latitude and Longitude are not NaN
    if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
        intensity = zone_intensity.get(row['Zone'], 1)  # Default to Green if zone not found
        heatmap_data.append([row['Latitude'], row['Longitude'], intensity])

# Check if heatmap data is empty or contains NaNs
if len(heatmap_data) > 0:
    HeatMap(heatmap_data).add_to(delhi_map)
else:
    st.write("No valid heatmap data available due to missing latitude/longitude values.")

# Save and display the map
delhi_map.save('delhi_crime_heatmap_user_location.html')
st.write("Crime Zone Heatmap:")
st.components.v1.html(open('delhi_crime_heatmap_user_location.html').read(), height=600)

