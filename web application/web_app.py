import streamlit as st


import pandas as pd
import geopandas as gpd
import osmnx as ox
import folium
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import nearest_points
import branca.colormap as cm
import joblib


##
st.title('Bike share')

place = st.text_input(label='Input location, e.g. Portland, Oregon, USA', value='Portland, Oregon, USA')

city = ox.gdf_from_place(place)

folder_name = 'example_data/'
portland_grid = gpd.read_file(folder_name + 'portland_grid.geojson')
portland_ft = gpd.read_file(folder_name + 'portland_feature_target_table.geojson')
portland_bike_rentals = gpd.read_file(folder_name + 'portland_bike_rentals.geojson')

st.write('The current location is', place)

# Load saved RF model
loaded_model = joblib.load('best_RF_model.sav')

top_features = ['cafe_density', 'bar_density', 'bicycle_parking_density', 'fast_food_density',
                'bank_density', 'pharmacy_density', 'pub_density', 'atm_density', 'car_sharing_density',
                'theatre_density', 'post_office_density', 'drinking_water_density', 'school_density',
                'cinema_density', 'bench_density', 'motorcycle_parking_density', 'ice_cream_density',
                'recycling_density', 'college_density', 'toilets_density', 'arts_centre_density',
                'nightclub_density', 'library_density', 'taxi_density', 'marketplace_density',
                'community_centre_density', 'place_of_worship_density', 'waste_basket_density',
                'clinic_density', 'social_facility_density', 'fountain_density', 'bureau_de_change_density',
                'kindergarten_density', 'police_density', 'veterinary_density', 'parking_density',
                'university_density', 'parking_entrance_density', 'childcare_density', 'hospital_density',
                'car_rental_density', 'vending_machine_density', 'dentist_density', 'bus_station_density']

# predict optimal bike share locations
portland_predict = loaded_model.predict(portland_ft[top_features])
portland_comparison = portland_grid.copy()
portland_comparison['bike_rental_density'] = portland_ft['bike_rental_density'] 
portland_comparison['RF_prediction'] = portland_predict
portland_comparison['bike_rental_diff'] = portland_comparison['RF_prediction'] - portland_comparison['bike_rental_density']

scale_factor = max(max(portland_predict),max(portland_comparison.bike_rental_density))

portland_comparison['scaled_actual_density'] = portland_ft['bike_rental_density'] / scale_factor
portland_comparison['scaled_pred_density'] = portland_comparison['RF_prediction'] / scale_factor


# define dictionaries for opacity and colormaps
pred_dict = portland_comparison['scaled_pred_density']
actual_dict = portland_comparison['scaled_actual_density']
diff_dict = portland_comparison['bike_rental_diff']

pred_opacity = {str(key): pred_dict[key] for key in pred_dict.keys()}
actual_opacity = {str(key): actual_dict[key] for key in actual_dict.keys()}
diff_opacity = {str(key): abs(diff_dict[key]) for key in diff_dict.keys()}

colormap = cm.linear.RdBu_11.scale(-scale_factor,scale_factor)

diff_color = {str(key): colormap(diff_dict[key]) for key in diff_dict.keys()}

# Generate folium plot
m = folium.Map([city.geometry.centroid.y, city.geometry.centroid.x],
               zoom_start=11,
               tiles="CartoDb positron")

style_city = {'color':'#ebc923 ', 'fillColor': '#ebc923 ', 'weight':'1', 'fillOpacity' : 0.1}
folium.GeoJson(city,
               style_function=lambda x: style_city,
               name='City Limit').add_to(m)

# Plot actual bike share density
folium.GeoJson(
    portland_comparison['geometry'],
    name='Actual bike share density',
    style_function=lambda feature: {
        'fillColor': '#0675c4',
        'color': 'black',
        'weight': 0,
        'fillOpacity': actual_opacity[feature['id']],
    }
).add_to(m)

# plot predictions of bike share density
folium.GeoJson(
    portland_comparison['geometry'],
    name='Prediction: bike share density',
    style_function=lambda feature: {
        'fillColor': '#0675c4',
        'color': 'black',
        'weight': 0,
        'fillOpacity': pred_opacity[feature['id']],
    }
).add_to(m)

# add difference
folium.GeoJson(
    portland_comparison['geometry'],
    name='Difference: bike share density',
    style_function=lambda feature: {
        'fillColor': diff_color[feature['id']],
        'color': 'black',
        'weight': 0,
#         'fillOpacity': 0.5,
        'fillOpacity': diff_opacity[feature['id']],
    }
).add_to(m)

colormap.caption = 'Difference in actual vs predicted bike demand'
colormap.add_to(m)

folium.LayerControl().add_to(m)

st.markdown(m._repr_html_(), unsafe_allow_html=True)

# def heat_map(df):
#     locs = zip(df.Y, df.X)
#     m = folium.Map([38.8934, -76.9470], tiles=’stamentoner’,   zoom_start=12)
#     HeatMap(locs).add_to(m)
#     return st.markdown(m._repr_html_(), unsafe_allow_html=True)




# HtmlFile = open('portland_validation.html', 'r', encoding='utf-8')
# html_code = HtmlFile.read() 
# 
# html_text = html2text.html2text(html_code)
# 
# # st.write(html_text)
# st.markdown(html_text, unsafe_allow_html=True)