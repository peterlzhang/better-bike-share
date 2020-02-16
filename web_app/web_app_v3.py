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
# Initialize amenity names
##########################################################################################
                 
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

st.subheader('About')
st.write('Better Bike Share is a tool that uses machine learning to predict optimal bike share locations. To use this tool, please input a city to evaluate.')
# st.write('Please input an city to evaluate bike share locations')
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
# Generate comparison of actual and predicted
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
# Mapping
##########################################################################################

status.text('STATUS: Generating map')

num_bs = (city_comparison.poly_area_km*city_comparison.bike_rental_density).sum()
capacity = np.floor(city_comparison.poly_area_km*city_comparison.RF_prediction).sum()

msg = f"The predicted maximum bike share capacity is {capacity:.0f}. There are currently {num_bs:.0f} bike shares. "
st.markdown(msg)



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

st.write('The map above shows how the distribution of bikes shares can be optimized. \
Red regions indicate an over saturated market and blue regions indicated an under \
saturated market. Select different layers to explore actual and predicted bike share distributions.')


##########################################################################################
# Contact
##########################################################################################
st.subheader('Contact')

msg1 = 'Email: <a href = "mailto: peter.li.zhang.com">peter.li.zhang@gmail.com</a> <br>\
Github: <a href="https://github.com/peterlzhang" target="_blank" >github.com/peterlzhang</a> <br>\
LinkedIn: <a href="https://www.linkedin.com/in/peter-zhang-ds" target="_blank">www.linkedin.com/in/peter-zhang-ds/</a>'
st.markdown(msg1,  unsafe_allow_html=True)