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
import bike_share_funcs as bsf
from folium import GeoJsonTooltip
##########################################################################################
#
##########################################################################################

    
# amenity_names = ['animal_shelter', 'archive', 'arts_centre', 'atm', 'bank', 'bar', 'bench', 'bench;waste_basket',
#                  'bicycle_parking', 'bicycle_repair_station', 'biergarten', 'bureau_de_change', 'bus_station', 'cafe',
#                  'car_rental', 'car_sharing', 'car_wash', 'casino', 'charging_station', 'childcare', 'cinema',
#                  'circus_school', 'clinic', 'clock', 'club', 'college', 'community_centre', 'compressed_air',
#                  'conference_centre', 'courthouse', 'crematorium', 'dentist', 'device_charging_station', 'doctors',
#                  'doctors_offices', 'drinking_water', 'embassy', 'events_venue', 'fast_food', 'ferry_terminal',
#                  'fire_station', 'fountain', 'fuel', 'gallery', 'gambling', 'garden', 'grave_yard', 'grit_bin',
#                  'hospital', 'ice_cream', 'jobcentre', 'kindergarten', 'language_school', 'left_luggage',
#                  'library', 'life_boats', 'luggage_locker', 'marketplace', 'monastery', 'money_transfer',
#                  'money_transfer; post_office', 'motorcycle_parking', 'music_school', 'music_venue', 'nightclub',
#                  'nursing_home', 'parcel_lockers', 'parking', 'parking_entrance', 'parking_space', 'pharmacy',
#                  'photo_booth', 'place_of_worship', 'place_of_worship;monastery', 'police', 'post_box', 'post_depot',
#                  'post_office', 'prep_school', 'preschool', 'prison', 'pub', 'public_bath', 'public_bookcase',
#                  'public_building', 'recycling', 'restaurant', 'restaurant;cafe', 'school', 'shelter',
#                  'social_centre', 'social_facility', 'sport', 'stripclub', 'studio', 'swimming_pool', 'swingerclub',
#                  'taxi', 'telephone', 'theatre', 'toilets', 'townhall', 'trailer_park', 'trade_school', 'university',
#                  'vending_machine', 'venue', 'veterinary', 'waste_basket', 'water', 'water_fountain','yacht_club']
#                  
                 
                 
amenity_names = ['cafe', 'bar', 'bicycle_parking', 'fast_food',
                'bank', 'pharmacy', 'pub', 'atm', 'car_sharing',
                'theatre', 'post_office', 'drinking_water', 'school',
                'cinema', 'bench', 'motorcycle_parking', 'ice_cream',
                'recycling', 'college', 'toilets', 'arts_centre',
                'nightclub', 'library', 'taxi', 'marketplace',
                'community_centre', 'place_of_worship', 'waste_basket',
                'clinic', 'social_facility', 'fountain', 'bureau_de_change',
                'kindergarten', 'police', 'veterinary']
##########################################################################################
# Start primary script
##########################################################################################
st.title('Better Bike Share')

st.write('Predict optimal bike share locations using machine learning model trained on global Open Street Map (OSM) data')
place = st.text_input(label='Input location, e.g. Portland, Oregon, USA', value='Portland, Oregon, USA')


top_features = ['cafe_density', 'bar_density', 'bicycle_parking_density', 'fast_food_density',
                'bank_density', 'pharmacy_density', 'pub_density', 'atm_density', 'car_sharing_density',
                'theatre_density', 'post_office_density', 'drinking_water_density', 'school_density',
                'cinema_density', 'bench_density', 'motorcycle_parking_density', 'ice_cream_density',
                'recycling_density', 'college_density', 'toilets_density', 'arts_centre_density',
                'nightclub_density', 'library_density', 'taxi_density', 'marketplace_density',
                'community_centre_density', 'place_of_worship_density', 'waste_basket_density',
                'clinic_density', 'social_facility_density', 'fountain_density', 'bureau_de_change_density',
                'kindergarten_density', 'police_density', 'veterinary_density']


status = st.empty()
status2 = st.empty()
if place == 'Portland, Oregon, USA':
  city = ox.gdf_from_place(place)
  (city_grid, city_ft, city_bike_rentals) = bsf.load_example_data()
else:
  status.text('STATUS: Gathering city data from OSM')
  city = ox.gdf_from_place(place)
  status.text('STATUS: Generating city grid')
  city_grid = bsf.gridify_polygon(city,0.01)
  status.text('STATUS: Finding existing bike rentals')
  city_bike_rentals = ox.pois_from_place(place, amenities=['bicycle_rental'])
  status.text('STATUS: Gathering local geographic features from Open Street Map (OSM).')
  status2.text('This may take longer for large cities.')
  all_amenities = ox.pois_from_place(place, amenities=amenity_names)
  all_amenities['geometry'] = all_amenities.apply(lambda row: row['geometry'].centroid 
                                                if (type(row['geometry']) == Polygon) or (type(row['geometry']) == MultiPolygon)
                                                else row['geometry'], axis=1)
  status.text('STATUS: Engineering features. This may take longer for large cities.')
  status2.text('')
  city_ft = bsf.features_density(city_grid,all_amenities,city_bike_rentals)
  
  status.text('STATUS: Data collection complete')


# Load saved RF model
loaded_model = joblib.load('best_RF_model.sav')





##########################################################################################
#
##########################################################################################

status.text('STATUS: Predicting bike share distribution')

# predict optimal bike share locations
city_predict = loaded_model.predict(city_ft[top_features])
city_comparison = city_grid.copy()
city_comparison['bike_rental_density'] = city_ft['bike_rental_density'] 
city_comparison['RF_prediction'] = city_predict
city_comparison['bike_rental_diff'] = city_comparison['RF_prediction'] - city_comparison['bike_rental_density']

scale_factor = max(max(city_predict),max(city_comparison.bike_rental_density))
diff_factor = max(abs(city_comparison['bike_rental_diff']))


city_comparison['scaled_actual_density'] = city_ft['bike_rental_density'] / scale_factor
city_comparison['scaled_pred_density'] = city_comparison['RF_prediction'] / scale_factor



# define dictionaries for opacity and colormaps
pred_dict = city_comparison['scaled_pred_density']
actual_dict = city_comparison['scaled_actual_density']
diff_dict = city_comparison['bike_rental_diff']

pred_opacity = {str(key): pred_dict[key]*0.5 for key in pred_dict.keys()}
actual_opacity = {str(key): actual_dict[key]*0.5 for key in actual_dict.keys()}
diff_opacity = {str(key): abs(diff_dict[key])/(diff_factor*2) for key in diff_dict.keys()}

colormap = cm.linear.RdBu_09.scale(-diff_factor,diff_factor)

diff_color = {str(key): colormap(diff_dict[key]) for key in diff_dict.keys()}


##########################################################################################
#
##########################################################################################

status.text('STATUS: Generating map')

m = folium.Map([city.geometry.centroid.y, city.geometry.centroid.x],
               zoom_start=11,
               tiles="CartoDb positron")

style_city = {'color':'#ebc923 ', 'fillColor': '#ebc923 ', 'weight':'2', 'fillOpacity' : 0}
folium.GeoJson(city,
               style_function=lambda x: style_city,
               name='City Limit').add_to(m)

# Plot actual bike share density
folium.GeoJson(
    city_comparison['geometry'],
    name='Actual bike share density',
    show = False,
    style_function=lambda feature: {
        'fillColor': '#04d45b',
        'color': 'black',
        'weight': 0,
        'fillOpacity': actual_opacity[feature['id']],
    }
).add_to(m)

# plot predictions of bike share density
folium.GeoJson(
    city_comparison['geometry'],
    name='Prediction: bike share density',
    show = False,
    style_function=lambda feature: {
        'fillColor': '#04d45b',
        'color': 'black',
        'weight': 0,
        'fillOpacity': pred_opacity[feature['id']],
    }
).add_to(m)

# add difference
tooltip=GeoJsonTooltip(
    fields=["bike_rental_density", "RF_prediction"],
    aliases=["Bike share density:", "ML prediction:"],
    localize=True,
    sticky=False,
    labels=True,
#     style="""
#         background-color: #ffffff;
#         border: 2px solid black;
#         border-radius: 3px;
#         box-shadow: 3px;
#     """,
#     max_width=800,
)

folium.GeoJson(
    city_comparison,
    name='Difference: bike share density',
    tooltip=tooltip,
    style_function=lambda feature: {
        'fillColor': diff_color[feature['id']],
        'color': 'black',
        'weight': 0,
#         'fillOpacity': 0.75,
        'fillOpacity': diff_opacity[feature['id']],
    }
).add_to(m)

colormap.caption = 'Difference in actual vs predicted bike share density'
colormap.add_to(m)

folium.LayerControl().add_to(m)

st.markdown(m._repr_html_(), unsafe_allow_html=True)

status.text('STATUS: Analysis complete')