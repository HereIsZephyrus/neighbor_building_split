import geopandas as gpd
from .district import DistrictDataset

class BuildingDataset:
    """"load buildings from shapefile"""
    def __init__(self, shapefile_path: str):
        self.gdf = gpd.read_file(shapefile_path)

    def get_buildings(self, district_dataset: DistrictDataset, district_id: int):
        district_geometry = district_dataset.get_district_geometry(district_id)
        return self.gdf[self.gdf.intersects(district_geometry)]

    def __len__(self):
        return len(self.gdf)

    def __getitem__(self, index: int):
        return self.gdf.iloc[index]
