import geopandas as gpd

class DistrictDataset:
    def __init__(self, shapefile_path: str):
        self.gdf = gpd.read_file(shapefile_path)

    def get_district_ids(self):
        return self.gdf['FID'].unique().tolist()

    def get_district_geometry(self, district_id: int):
        return self.gdf[self.gdf['FID'] == district_id].geometry.iloc[0]

    def __len__(self):
        return len(self.gdf)
