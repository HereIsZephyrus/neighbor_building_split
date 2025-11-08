import os
import argparse
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description="Merge shapefiles")
    parser.add_argument("--input-directory", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    return parser.parse_args()

def merge_shapefiles(input_dir, output_path):
    """
    Merge all shapefiles in the input directory into a single shapefile.
    :param input_dir: The path to the input directory containing .shp files
    :param output_path: The path to the output shapefile (including filename)
    """
    # Collect all .shp file paths
    shp_files = []
    for file in os.listdir(input_dir):
        if file.endswith(".shp") and not file.startswith("~"):
            shp_path = os.path.join(input_dir, file)
            shp_files.append(shp_path)

    if not shp_files:
        print("ERROR: No shapefiles found in the input directory")
        return

    # Batch read and merge
    gdf_list = []
    first_gdf = gpd.read_file(shp_files[0])
    target_crs = first_gdf.crs
    gdf_list.append(first_gdf)
    for shp in tqdm(shp_files[1:], desc="Merging progress"):
        try:
            gdf = gpd.read_file(shp)
            if gdf.crs != target_crs:
                gdf = gdf.to_crs(target_crs)
            gdf_list.append(gdf)
        except Exception as e:
            print(f"ERROR: Failed to read file {shp}: {e}")
            continue

    # Concatenate all data
    merged_gdf = pd.concat(gdf_list, ignore_index=True)
    merged_gdf.to_file(output_path)
    print(f"INFO: Merged {len(gdf_list)} files and saved to {output_path}")

if __name__ == "__main__":
    args = parse_arguments()
    input_directory = args.input_directory
    output_file = args.output_file
    merge_shapefiles(input_directory, output_file)
