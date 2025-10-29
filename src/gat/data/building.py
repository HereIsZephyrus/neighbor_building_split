import geopandas as gpd


class BuildingDataset:
    """"load buildings from shapefile"""
    def __init__(self, shapefile_path: str):
        self.gdf = gpd.read_file(shapefile_path)

    def get_district_ids(self):
        return self.gdf['district_id'].unique().tolist()

    def get_buildings(self, district_id: int):
        return self.gdf[self.gdf['district_id'] == district_id]

    def __len__(self):
        return len(self.gdf)

    def __getitem__(self, index: int):
        return self.gdf.iloc[index]
