# Flood Impacts 
This model takes the CityCAT output and determines the flood impacts on the buildings

## Description
The CityCAT model produces the maximum flood depth in each cell. This information together with the buildings polygons is used to caculate the flood damage to the buildings.

## Input Parameters
*Location: The city/location of interest.
*Projection: the numerical code for the required projection (e.g. 27700). 


## Input Files (data slots)
* Buildings
  * Description: A .gpkg of the buildings
  * Location: /data/inputs/buildings
* Parameters
  * Description: location and projection
  * Location: /data/inputs/parameters
* Maximum Depth Raster
  * Description: A Geotiff file giving the maxmimum flooding depth for each cell
  * Location: /data/inputs/run
* Maximum Depth CSV
  * Description: A CSV file giving the maxmimum flooding depth for each cell
  * Location: /data/inputs/run

## Outputs
* The model should output 4 files, which give the flooding depths of the buildings.
  * Location: /data/outputs/impacts

