# Workaround to fix chrome issue where folium won't plot maps with a large number of layers
# See comment by dstein64 at: https://github.com/python-visualization/folium/issues/812


import base64
import folium


def _repr_html_(self, **kwargs):
    html = base64.b64encode(self.render(**kwargs).encode('utf8')).decode('utf8')
    onload = (
        'this.contentDocument.open();'
        'this.contentDocument.write(atob(this.getAttribute(\'data-html\')));'
        'this.contentDocument.close();'
    )
    if self.height is None:
        iframe = (
            '<div style="width:{width};">'
            '<div style="position:relative;width:100%;height:0;padding-bottom:{ratio};">'
            '<iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;'
            'border:none !important;" '
            'data-html={html} onload="{onload}" '
            'allowfullscreen webkitallowfullscreen mozallowfullscreen>'
            '</iframe>'
            '</div></div>').format
        iframe = iframe(html=html, onload=onload, width=self.width, ratio=self.ratio)
    else:
        iframe = ('<iframe src="about:blank" width="{width}" height="{height}"'
                  'style="border:none !important;" '
                  'data-html={html} onload="{onload}" '
                  '"allowfullscreen" "webkitallowfullscreen" "mozallowfullscreen">'
                  '</iframe>').format
        iframe = iframe(html=html, onload=onload, width=self.width, height=self.height)
    return iframe

folium.branca.element.Figure._repr_html_ = _repr_html_


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

def avg_dist_to_amenities(interrogation_point,amenity_df,n):
    # calculates the mean distance of the n nearest amenities to the interrogation point
    # If there are less than n amenities in the search it'll just return the average of the known amenities.
    # Example: avg_dist_to_amenities(city_grid.geometry.iloc[39],food_amenities,5)
    dist_to_amenity = amenity_df['geometry'].apply(lambda x: x.distance(interrogation_point))
    dist_to_amenity.sort_values(inplace=True)
    dist_to_amenity[:5]
    if len(dist_to_amenity) >= n:
        return dist_to_amenity[:n].mean()
    elif len(dist_to_amenity) == 0:
        return np.nan
    else:
        return dist_to_amenity.mean()
    
    
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
    cleaned_df = interrogation_grid.copy()
    cleaned_df = cleaned_df.reset_index()
    cleaned_df['bike_rental_density'] = 0
    cleaned_df = cleaned_df.reindex(cleaned_df.columns.tolist() + amenity_names, axis=1) 


    # loop through grid points and populate features.
    for index, row in cleaned_df.iterrows():
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

    # remove nan values
    cleaned_df[amenity_names] = cleaned_df[amenity_names].fillna(0)
    # remove unecessary columns
    cleaned_df = cleaned_df.drop(columns = ['level_0','index'])
    # relable as density 
    new_names = [name + '_density' for name in amenity_names]
    cleaned_df.rename(columns = dict(zip(amenity_names, new_names)), inplace=True)
    
    return cleaned_df