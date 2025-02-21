import os
import glob
from glob import glob
import pandas as pd
import rasterio as rio
from rasterio import features
from rasterio.crs import CRS
from rasterstats import zonal_stats as zs
import geopandas as gpd
import shutil
import numpy as np
from shapely.geometry import shape
import rtree
import re
from shapely.geometry import shape, Point
import datetime
from operator import itemgetter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Set basic data paths
data_path = os.getenv('DATA_PATH', '/data')
inputs_path = os.path.join(data_path, 'inputs')
outputs_path = os.path.join(data_path, 'outputs')
if not os.path.exists(outputs_path):
    os.mkdir(outputs_path)

#Set model specific data paths
buildings_path = os.path.join(inputs_path, 'buildings')
#dd_curves_path = os.path.join(inputs_path, 'dd-curves')
#grid_path = os.path.join(inputs_path,'grid')
run_path = os.path.join(inputs_path, 'run')
#uprn_lookup = glob(os.path.join(inputs_path, 'uprn', '*.csv'))
parameters_path=os.path.join(inputs_path,'parameters')
#udm_para_in_path = os.path.join(inputs_path, 'udm_parameters')

categorys_path = os.path.join(outputs_path, 'impacts')
if not os.path.exists(categorys_path):
    os.mkdir(categorys_path)

def process_data(ev):
    
    T_start = datetime.datetime.now()
    builds_file = all_buildings
    #input_folder = run_path
    
    print(builds_file)
        
    #Creating the spatial index with the rtree module is only done for ine depth file using X,Y only
    print('..creating spatial index..')
    
    #Find the csv file:
    #csv_lookup = glob(os.path.join(run_path, '*.csv',recursive = True))
    csv_lookup = glob(run_path + '/**/*max_depth.csv',recursive = True)
     
    #first get the resolution of the grid:
    df_res = pd.read_csv(csv_lookup[0], nrows = 3)
    xdiff = df_res.iloc[2, 0] - df_res.iloc[1, 0]
    ydiff = df_res.iloc[2, 1] - df_res.iloc[1, 1]
    if xdiff != 0:
        dx = xdiff
    elif xdiff == 0:
        dx = ydiff
    del(df_res)
    #buffer_distance = ((buffer_value)/100) * dx # in % grid resolution
    buffer_distance = dx # in % grid resolution
    
    x = []
    y = []
    with open(csv_lookup[0], 'r') as t:
        aline = t.readline().strip()
        aline = t.readline()
        while aline != '':
            column = re.split('\s|\s\s|\t|,',str(aline))
            x.append(float(column[0]))
            y.append(float(column[1]))
            aline = t.readline()
    t.close()
    
    cell_idx = []
    for idx, xi in enumerate(x):
        cell_idx.append(idx)
        
    index = rtree.index.Index() #create the spatial index
    for pt_idx, xi, yi in zip(cell_idx, x, y):
        index.insert(pt_idx, (xi, yi))
        
    del(cell_idx)
    
    cell_index = []
    buffer_list = []
    builds = builds_file
    builds_n = len(builds)
    #which column is the buildings field in? If it is not in id or toid_number then the index is used
    if builds_field1 == 'index_column':
        builds_df = gpd.GeoDataFrame(builds.assign(index_column=builds.index)[['index_column', 'geometry']])
    else:
        builds_df = gpd.GeoDataFrame(builds[[str(builds_field1), 'geometry']])
    del(builds)
    
    for b_id, b_geom in zip(builds_df[str(builds_field1)], builds_df['geometry']):
        buffer = shape(b_geom.buffer(float(buffer_distance), resolution=10)) #create a buffer polygon for the building polygons from resolution 10 to 16
        for cell in list(index.intersection(buffer.bounds)): #first check if the point is within the bounding box of a building buffer
            cell_int = Point(x[cell], y[cell])  
            if cell_int.intersects(buffer): #then check if the point intersects with buffer polygon
                buffer_list.append(b_id) #store the building ID
                cell_index.append(cell) #store the line inedex of the intersecting points
                
    df_b = pd.DataFrame(list(zip(buffer_list, cell_index)), columns = [str(builds_field1), 'cell'])
    df_b = df_b.sort_values(by = ['cell'])
    print('spatial index created')
    
    #------------------------------------------------------------------------------reading depth files 
        
    files = csv_lookup
    print('files:',files)
    
    for i, filename in enumerate(files):
        f = open(filename)
        print('processing file: ' + str(filename))
        Z=[]
        aline = f.readline().strip()       
        aline = f.readline()
        while aline != '':
            column = re.split('\s|\s\s|\t|,',str(aline))
            Z.append(float(column[2]))
            aline = f.readline()
        f.close()
        
        #--------------------------------------------------------------------------spatial intersection and classification
        #the next line reads the depth values from the file according to cell index from above and stores the depth with the intersecting building ID
        df = pd.DataFrame(list(zip(itemgetter(*cell_index)(Z),buffer_list)), columns=['depth',str(builds_field1)])
        del(Z)
        
        #based on the building ID the mean and maximum depth are established and stored in a new data frame:
        mean_depth = pd.DataFrame(df.groupby([str(builds_field1)])['depth'].mean().astype(float)).round(3).reset_index(level=0).rename(columns={'depth':'mean_depth'}) 
        p90ile_depth = pd.DataFrame(df.groupby([str(builds_field1)])['depth'].quantile(0.90).astype(float)).round(3).reset_index(level=0).rename(columns={'depth':'p90ile_depth'})
        damages_df = pd.merge(mean_depth, p90ile_depth)
        del(mean_depth, p90ile_depth)
        
        #calculate the damages according to the water depth in buffer zone and the type of the building
        damages_df['Class'] = 'A) Low'
        damages_df['Class'][(damages_df['mean_depth'] >= 0) & (damages_df['mean_depth'] < 0.10) & (damages_df['p90ile_depth'] < 0.30)] = 'A) Low'
        damages_df['Class'][(damages_df['mean_depth'] >= 0) & (damages_df['mean_depth'] < 0.10) & (damages_df['p90ile_depth'] >= 0.30)] = 'B) Medium'
        damages_df['Class'][(damages_df['mean_depth'] >= 0.10) & (damages_df['mean_depth'] < 0.30) & (damages_df['p90ile_depth'] < 0.30)] = 'B) Medium' 
        damages_df['Class'][(damages_df['mean_depth'] >= 0.10) & (damages_df['p90ile_depth'] >= 0.30)] = 'C) High'  
        
        #------------------------------------------------------------------------------merge results with a copy of the building layer and create output files
        finalf = builds_df.merge(damages_df, on = str(builds_field1), how = 'left') #the merging of the building shapefile
        
        finalf['Area'] = (finalf.area).astype(int)#calculate the area for each building
        finalf.to_file(os.path.join(categorys_path, 
                        location + '_exposure.gkpg'),driver="GPKG")

        #plot mean exposure depth in finalf dataframe as a png
        dpi = 300
        print('dpi:',dpi)
        fig, ax = plt.subplots(1, 1, dpi = dpi)
        plt.subplots_adjust(left = 0.10 , bottom = 0, right = 0.90 , top =1)
        cmap=mpl.cm.Reds
        bounds_depth =  [0.01, 0.05, 0.10, 0.15, 0.30, 0.50, 0.80, 1.00] #you could change here the water depth of your results
        norm = mpl.colors.BoundaryNorm(bounds_depth, cmap.N)
        ax.set_title("Mean Exposure Depth (m)")
        axins = inset_axes(ax,
                   width="2%", # width of colorbar in % of plot width
                   height="45%", # height of colorbar in % of plot height
                   loc=2, #topright location
                   bbox_to_anchor=(1.01, 0, 1, 1), #first number: space relative to plot (1.0 = no space between cb and plot)
                   bbox_transform=ax.transAxes,
                   borderpad=0) 
        finalf.plot(column='mean_depth', ax = ax,  cmap = 'Reds', norm = norm,  edgecolor='black', linewidth=0.1)
        plt.colorbar(mpl.cm.ScalarMappable(cmap = cmap, norm = norm),
             ax = ax,
             cax = axins,
             extend = 'both',
             format='%.2f',
             ticks = bounds_depth,
             spacing = 'uniform',
             orientation = 'vertical',
             label = 'Mean Exposure Depth (m)')
        plt.savefig(os.path.join(categorys_path, 'exposure_plot.png'), dpi=dpi, bbox_inches='tight')

        class_low = (finalf['Class'] == 'A) Low').sum()
        class_medium = (finalf['Class'] == 'B) Medium').sum()        
        class_high = (finalf['Class'] == 'C) High').sum()
        del(damages_df)
        
        del(finalf['geometry'])
        finalf_csv = pd.DataFrame(finalf)
        # finalf_csv.to_csv(os.path.join(categorys_path, 
        #                     location + '-' + ssp + '-' + year + '-' + depth1 + '_exposure.csv'))
        finalf_csv.to_csv(os.path.join(categorys_path, 
                            location + '_exposure.csv'))
        
        #del(builds_data, builds_df, finalf, finalf_csv, df)
        del(builds_df, finalf, finalf_csv, df)
        
        with open(os.path.join(categorys_path,
                    #location + '-' + ssp + '-' + year + '-' + depth1 + '_exposure_summary.txt'), 'w') as sum_results:
                    location + '_exposure_summary.txt'), 'w') as sum_results:
            sum_results.write('Summary of Exposure Analysis for: ' + str(filename) + '\n\n'
                        + 'Number of Buildings: ' + str(builds_n) + '\n'
                        + 'Grid Resolution: ' + str(dx) + 'm' + '\n'
                        #+ 'Buffer Distance: ' +str(buffer_value) + '% or' + str(buffer_distance) + 'm' + '\n\n'
                        + 'Buffer Distance: ' + str(buffer_distance) + 'm' + '\n\n'
                        + 'Low: ' + str(class_low) + '\n'
                        + 'Medium: ' +str(class_medium) + '\n'
                        + 'High: ' +str(class_high) + '\n\n')
            sum_results.close()
            
    del(x, y)
    del(buffer_list, cell_index, df_b)
    print('The Exposure Analysis is Finished. Time required: ' + str(datetime.datetime.now() - T_start)[:-4])


# Identify the CityCat output raster
archive = glob(run_path + "/*.tif", recursive = True)


# Identify the building files for the buildings
buildings = glob(os.path.join(buildings_path + "/*.gpkg"))

# Search for a parameter file which outline the input parameters defined by the user
parameter_file = glob(parameters_path + "/*.csv", recursive = True)
print('parameter_file:', parameter_file)

if len(parameter_file) != 0 :
    all_parameters = pd.concat(map(pd.read_csv,parameter_file),ignore_index=True)
    print(all_parameters)
    if 'LOCATION' in all_parameters.values:
        location_row = all_parameters[all_parameters['PARAMETER']=='LOCATION']
        location=str(location_row['VALUE'].values[0])
        print('location:',location)
    if 'PROJECTION' in all_parameters.values:
        projection_row = all_parameters[all_parameters['PARAMETER']=='PROJECTION']
        projection=projection_row['VALUE'].values[0]
        print('projection:',projection)       
else:
    location = os.getenv('LOCATION')
    projection = os.getenv('PROJECTION')



# Read in the baseline builings
with rio.open(archive[0],'r+') as max_depth :
    # Set crs of max_depth raster
    max_depth.crs = CRS.from_epsg(projection)
    all_buildings = gpd.read_file(buildings[0], bbox=max_depth.bounds)    # Redefine the toid number to include osgb
    columns=list(all_buildings.columns)
    #which column is the buildings field in? If it is not in id or toid_number then the index is used
    if 'id' in columns:
        builds_field1 = 'id'
    elif 'toid_number' in columns:
        builds_field1 = 'toid_number'
    elif 'toid_numbe' in columns:
        builds_field1 = 'toid_numbe'
    else:
        builds_field1='index_column'
   
    # Create a list of all of the column headers in the buildings file:
    cols_list = []
    print('cols_list:',cols_list)
    
    process_data('Risk_Levels')
    
