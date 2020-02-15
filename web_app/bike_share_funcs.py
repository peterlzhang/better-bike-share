import streamlit as st
import pandas as pd
import geopandas as gpd
import osmnx as ox
import folium
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
# from shapely.ops import nearest_points
# import branca.colormap as cm

def load_example_data():
  folder_name = 'example_data/'
  portland_grid = gpd.read_file(folder_name + 'portland_grid.geojson')
  portland_ft = gpd.read_file(folder_name + 'portland_feature_target_table.geojson')
  portland_bike_rentals = gpd.read_file(folder_name + 'portland_bike_rentals.geojson')
  return (portland_grid,portland_ft,portland_bike_rentals)

def gridify_polygon(poly,grid_spacing):
    # creates a cartesian grid inside polygon with the input grid_spacing
    # poly: polygon which we want a grid inside
    # grid_spacing: spaceing in lattitude/longitude degrees
    poly_xmin,poly_ymin,poly_xmax,poly_ymax = poly.geometry.total_bounds

    cols = list(np.arange(poly_xmin,poly_xmax+grid_spacing,grid_spacing))
    rows = list(np.arange(poly_ymin,poly_ymax+grid_spacing,grid_spacing))
    rows.reverse()

    polygons = []
    for x in cols:
        for y in rows:
            polygons.append( Polygon([(x,y), (x+grid_spacing, y), (x+grid_spacing, y-grid_spacing), (x, y-grid_spacing)]) )

    grid = gpd.GeoDataFrame({'geometry':polygons})

    grid['isin_poly'] = grid.apply(lambda row: row['geometry'].centroid.within(poly.geometry[0]), axis=1)
    poly_grid = grid[grid.isin_poly == True]
    poly_grid.crs = {'init': 'epsg:4326', 'no_defs': True}
    poly_grid = poly_grid.drop(['isin_poly'], axis = 1)
    
    # Calculate the polygon areas in km
    poly_grid_cart = poly_grid.copy()
    poly_grid_cart = poly_grid_cart.to_crs({'init': 'epsg:3857'})
    poly_grid_cart['poly_area_km'] = poly_grid_cart['geometry'].area/ 10**6
    # Store polygon area
    poly_grid['poly_area_km'] = poly_grid_cart['poly_area_km']
    
    # 
    poly_grid = poly_grid.reset_index()
    return poly_grid

def amenity_in_polygon(amenity_points,poly):
    # returns the amenities that are inside the given polygon
    # When there are zero amenities within the interrogation region, the function returns an empty dataframe as
    # as expected, but also prints out a lot of errors. not a huge issue but annoying.
    # Maybe implement a test for if empty, return 0
    # Example use:
    #         amenity_in_polygon(food_amenities,city_grid.geometry.iloc[38])
    
    # Generate boolean list of whether amenity is in polygon
    indices = amenity_points.apply(lambda row: row['geometry'].within(poly), axis=1)
    if not any(indices): # If all indices are false
        return pd.DataFrame(columns=['A']) # return empty dataframe (not sure what is best to output here )
    else:
        return amenity_points[amenity_points.apply(lambda row: row['geometry'].within(poly), axis=1)]

    
# Generate features dataframe by finding the count of each unique amenity in each region
def features_density(interrogation_grid,osm_features,targets):
    # Calculate feature and target density inside a series of polygons
    # INPUTS
    # ------
    # Interrogation grid: list of polygons in which to calculate density of features and targets
    # osm_features: gdf of amenities in area retrieved from OSM
    # targest: gdf of target locations
    # OUTPUTS
    # -------
    # cleaned_df: contains the density of features and targets in the interrogation grid
    # create new cleaned df that will store features and target data
    amenity_names = ['animal_shelter', 'archive', 'arts_centre', 'atm', 'bank', 'bar', 'bench', 'bench;waste_basket',
                 'bicycle_parking', 'bicycle_repair_station', 'biergarten', 'bureau_de_change', 'bus_station', 'cafe',
                 'car_rental', 'car_sharing', 'car_wash', 'casino', 'charging_station', 'childcare', 'cinema',
                 'circus_school', 'clinic', 'clock', 'club', 'college', 'community_centre', 'compressed_air',
                 'conference_centre', 'courthouse', 'crematorium', 'dentist', 'device_charging_station', 'doctors',
                 'doctors_offices', 'drinking_water', 'embassy', 'events_venue', 'fast_food', 'ferry_terminal',
                 'fire_station', 'fountain', 'fuel', 'gallery', 'gambling', 'garden', 'grave_yard', 'grit_bin',
                 'hospital', 'ice_cream', 'jobcentre', 'kindergarten', 'language_school', 'left_luggage',
                 'library', 'life_boats', 'luggage_locker', 'marketplace', 'monastery', 'money_transfer',
                 'money_transfer; post_office', 'motorcycle_parking', 'music_school', 'music_venue', 'nightclub',
                 'nursing_home', 'parcel_lockers', 'parking', 'parking_entrance', 'parking_space', 'pharmacy',
                 'photo_booth', 'place_of_worship', 'place_of_worship;monastery', 'police', 'post_box', 'post_depot',
                 'post_office', 'prep_school', 'preschool', 'prison', 'pub', 'public_bath', 'public_bookcase',
                 'public_building', 'recycling', 'restaurant', 'restaurant;cafe', 'school', 'shelter',
                 'social_centre', 'social_facility', 'sport', 'stripclub', 'studio', 'swimming_pool', 'swingerclub',
                 'taxi', 'telephone', 'theatre', 'toilets', 'townhall', 'trailer_park', 'trade_school', 'university',
                 'vending_machine', 'venue', 'veterinary', 'waste_basket', 'water', 'water_fountain','yacht_club']
    
    cleaned_df = interrogation_grid.copy()
    cleaned_df = cleaned_df.reset_index()
    cleaned_df['bike_rental_density'] = 0
    cleaned_df = cleaned_df.reindex(cleaned_df.columns.tolist() + amenity_names, axis=1) 

    
    bar = st.progress(0)
    num_rows = len(cleaned_df)
    # loop through grid points and populate features.
    for index, row in cleaned_df.iterrows():
        bar.progress(int((index/num_rows)*100))
        grid_pt = cleaned_df.geometry.iloc[index]
        amenities_in_grid = amenity_in_polygon(osm_features,grid_pt)

        # fill amenity rows with counts inside each polygon
        if len(amenities_in_grid) > 0:
            amenity_counts = amenities_in_grid['amenity'].value_counts()
            for val, cnt in amenity_counts.iteritems():
                # test if value is in list of features that are selected for ML model
                if val in amenity_names:
                    cleaned_df[val].iloc[index] = cnt / cleaned_df.poly_area_km.iloc[index]

        # add target column for bike rentals
        bike_rentals_in_grid = amenity_in_polygon(targets,grid_pt)
        if len(bike_rentals_in_grid) > 0:
            cleaned_df['bike_rental_density'].iloc[index] = len(bike_rentals_in_grid) / cleaned_df.poly_area_km.iloc[index]
        else:
            cleaned_df['bike_rental_density'].iloc[index] = 0
    bar.progress(int(100))
    # remove nan values
    cleaned_df[amenity_names] = cleaned_df[amenity_names].fillna(0)
    # remove unecessary columns
    cleaned_df = cleaned_df.drop(columns = ['level_0','index'])
    # relable as density 
    new_names = [name + '_density' for name in amenity_names]
    cleaned_df.rename(columns = dict(zip(amenity_names, new_names)), inplace=True)
    
    return cleaned_df