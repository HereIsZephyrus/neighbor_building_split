"""Generate Voronoi diagrams using morphological dilation."""

from typing import Tuple, Optional
import numpy as np
import geopandas as gpd
from scipy import ndimage
from shapely.geometry import LineString, Polygon
from shapely.ops import linemerge
from affine import Affine
import cv2
from rasterio import features
from ..utils.logger import get_logger

logger = get_logger(__name__)


class VoronoiGenerator:
    """Generate Voronoi partitions using dilation-based approach."""

    def __init__(self, simplify_tolerance: float = 0.5):
        """Initialize with line simplification tolerance in meters."""
        self.simplify_tolerance = simplify_tolerance

    def generate_voronoi_from_raster(
        self,
        building_raster: np.ndarray,
        district_mask: np.ndarray,
        visualize: bool = False,
        viz_interval: int = 1,
        debug_mode: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Voronoi partition using morphological dilation.
        
        Returns (voronoi_partition, buildings_mask) where unclassified pixels are -999.
        """
        building_binary = (building_raster > 0).astype(np.uint8)
        original_buildings_mask = building_binary.copy()
        voronoi = building_raster.copy()

        unique_ids = np.unique(voronoi[voronoi > 0])
        num_features = len(unique_ids)

        if num_features == 0:
            logger.warning("No buildings, returning empty diagram")
            return np.zeros_like(building_raster, dtype=np.int32), original_buildings_mask

        logger.debug("Dilating %d building regions", num_features)

        voronoi = self._dilate_labels(
            voronoi,
            district_mask,
            max_iterations=30,
            visualize=visualize,
            viz_interval=viz_interval,
            debug_mode=debug_mode,
            buildings_mask=original_buildings_mask
        )

        unclassified_mask = (voronoi == 0) & (district_mask == 1)
        unclassified_count = unclassified_mask.sum()
        if unclassified_count > 0:
            logger.warning("%d unclassified pixels marked as NoData", unclassified_count)
            voronoi[unclassified_mask] = -999

        return voronoi, original_buildings_mask

    def _generate_color_map(self, num_labels: int) -> np.ndarray:
        """Generate random BGR colors for visualization."""
        np.random.seed(42)
        colors = np.zeros((num_labels + 1, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]

        for i in range(1, num_labels + 1):
            colors[i] = np.random.randint(0, 256, size=3, dtype=np.uint8)

        return colors

    def _visualize_labels(
        self,
        labeled_array: np.ndarray,
        color_map: np.ndarray,
        district_mask: np.ndarray,
        iteration: int,
        buildings_mask: Optional[np.ndarray] = None,
        debug_mode: bool = False,
        window_name: str = "Voronoi Dilation",
    ) -> None:
        """Visualize labeled array with iteration info and legend."""
        vis_img = np.zeros((*labeled_array.shape, 3), dtype=np.uint8)

        for label in range(len(color_map)):
            mask = (labeled_array == label)
            vis_img[mask] = color_map[label]

        outside_mask = (district_mask == 0)
        vis_img[outside_mask] = [40, 40, 40]

        unlabeled_mask = (labeled_array == 0) & (district_mask == 1)
        vis_img[unlabeled_mask] = [0, 0, 255]

        if buildings_mask is not None:
            building_pixels = (buildings_mask > 0)
            vis_img[building_pixels] = [255, 255, 255]

        h, w = vis_img.shape[:2]
        min_size = 1200
        if h < min_size or w < min_size:
            scale = max(min_size / h, min_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            vis_img = cv2.resize(vis_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        display_img = vis_img.copy()

        text = f"Iteration: {iteration}"
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        cv2.rectangle(display_img, (5, 5), (text_w + 15, text_h + baseline + 15), (0, 0, 0), -1)
        cv2.putText(display_img, text, (10, text_h + 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (255, 255, 255), 2, cv2.LINE_AA)

        unlabeled = ((labeled_array == 0) & (district_mask == 1)).sum()
        text_color = (0, 0, 255) if unlabeled > 0 else (0, 255, 255)
        progress_text = f"Unlabeled: {unlabeled} pixels" + (" (RED)" if unlabeled > 0 else "")
        text_y = text_h + baseline + 35
        (prog_w, prog_h), prog_baseline = cv2.getTextSize(progress_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(display_img, (5, text_y - prog_h - 5), 
                     (prog_w + 15, text_y + prog_baseline + 5), (0, 0, 0), -1)
        cv2.putText(display_img, progress_text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, text_color, 2, cv2.LINE_AA)

        legend_y = text_y + prog_h + prog_baseline + 20
        legend_texts = [
            ("White = Buildings", (255, 255, 255)),
            ("Colors = Voronoi regions", (100, 255, 100)),
            ("Red = Unlabeled pixels", (0, 0, 255)),
            ("Dark gray = Outside district", (100, 100, 100))
        ]

        for i, (text, color) in enumerate(legend_texts):
            legend_y_pos = legend_y + i * 22
            cv2.putText(display_img, text, (10, legend_y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color, 1, cv2.LINE_AA)

        if debug_mode:
            hint_text = "DEBUG: Press SPACE to continue, 'q' to quit"
            hint_y = legend_y + len(legend_texts) * 22 + 15
            (hint_w, hint_h), hint_baseline = cv2.getTextSize(hint_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display_img, (5, hint_y - hint_h - 5), 
                         (hint_w + 15, hint_y + hint_baseline + 5), (0, 0, 128), -1)
            cv2.putText(display_img, hint_text, (10, hint_y), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(window_name, display_img)

    def _dilate_labels(
        self,
        labeled_array: np.ndarray,
        district_mask: np.ndarray,
        max_iterations: int = 30,
        visualize: bool = False,
        viz_interval: int = 1,
        debug_mode: bool = False,
        buildings_mask: Optional[np.ndarray] = None,
        window_name: str = "Voronoi Dilation Progress",
    ) -> np.ndarray:
        """
        Dilate labeled regions using 8-connectivity until district is filled.
        
        Returns labeled array with all district pixels assigned to nearest building.
        """
        result = labeled_array.copy()
        structure = ndimage.generate_binary_structure(2, 2)

        color_map = None
        if visualize:
            num_labels = len(np.unique(result[result > 0]))
            color_map = self._generate_color_map(num_labels)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            if debug_mode:
                logger.debug("Visualization: DEBUG mode (press SPACE to step)")
            else:
                logger.debug("Visualization: press 'q' to quit, 'p' to pause")

        paused = False

        for iteration in range(max_iterations):
            unlabeled_mask = (result == 0) & (district_mask == 1)

            if not unlabeled_mask.any():
                if visualize:
                    self._visualize_labels(
                        result, color_map, district_mask, iteration, 
                        buildings_mask=buildings_mask, debug_mode=debug_mode, 
                        window_name=window_name
                    )
                    cv2.waitKey(2000)
                break

            changed = False
            for label in np.unique(result[result > 0]):
                label_mask = (result == label)
                dilated = ndimage.binary_dilation(label_mask, structure=structure)
                new_pixels = dilated & unlabeled_mask

                if new_pixels.any():
                    result[new_pixels] = label
                    changed = True

            if visualize and (iteration % viz_interval == 0 or not changed):
                self._visualize_labels(
                    result, color_map, district_mask, iteration,
                    buildings_mask=buildings_mask, debug_mode=debug_mode,
                    window_name=window_name
                )

                if debug_mode:
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord('q'):
                            logger.debug("Visualization interrupted")
                            cv2.destroyWindow(window_name)
                            return result
                        elif key == ord(' '):
                            break
                else:
                    while True:
                        key = cv2.waitKey(1 if not paused else 0) & 0xFF
                        if key == ord('q'):
                            logger.debug("Visualization interrupted")
                            cv2.destroyWindow(window_name)
                            return result
                        elif key == ord('p'):
                            paused = not paused
                        elif not paused:
                            break

            if not changed:
                break

        remaining = ((result == 0) & (district_mask == 1)).sum()
        if remaining > 0:
            logger.warning("%d pixels remain unlabeled after %d iterations", remaining, max_iterations)

            if visualize:
                self._visualize_labels(
                    result, color_map, district_mask, max_iterations,
                    buildings_mask=buildings_mask, debug_mode=debug_mode,
                    window_name=window_name
                )
                if debug_mode:
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord(' ') or key == ord('q'):
                            break
                else:
                    cv2.waitKey(3000)

        if visualize:
            cv2.destroyWindow(window_name)

        return result

    def extract_boundaries(
        self,
        voronoi: np.ndarray,
        district_mask: np.ndarray,
        original_buildings_mask: np.ndarray,
    ) -> np.ndarray:
        """Extract centerline boundaries between Voronoi regions using skeletonization."""
        from scipy.ndimage import grey_dilation, grey_erosion
        from skimage.morphology import skeletonize

        dilated = grey_dilation(voronoi, size=(3, 3))
        eroded = grey_erosion(voronoi, size=(3, 3))

        boundaries_thick = (dilated != eroded).astype(np.uint8)

        valid_mask = (district_mask == 1) & (original_buildings_mask == 0)
        boundaries_thick = boundaries_thick & valid_mask

        if boundaries_thick.any():
            boundaries = skeletonize(boundaries_thick).astype(np.uint8)
        else:
            boundaries = boundaries_thick

        logger.debug("Extracted %d boundary pixels", boundaries.sum())

        return boundaries

    def vectorize_voronoi_polygons(
        self,
        voronoi: np.ndarray,
        transform: Affine,
        crs: str = "EPSG:32650",
        district_attrs: Optional[dict] = None,
    ) -> gpd.GeoDataFrame:
        """Convert Voronoi partition raster to vector polygons."""
        unique_ids = np.unique(voronoi[voronoi > 0])
        num_regions = len(unique_ids)

        if num_regions == 0:
            logger.warning("No regions to vectorize")
            return gpd.GeoDataFrame(columns=['geometry', 'building_id', 'area'], crs=crs)

        polygons = []
        building_ids = []

        for geom, value in features.shapes(voronoi.astype(np.int32), transform=transform):
            if value > 0:
                poly = Polygon(geom['coordinates'][0])
                if poly.is_valid and not poly.is_empty:
                    polygons.append(poly)
                    building_ids.append(int(value))

        if not polygons:
            logger.warning("No valid polygons extracted")
            return gpd.GeoDataFrame(columns=['geometry', 'building_id', 'area'], crs=crs)

        gdf = gpd.GeoDataFrame({
            'geometry': polygons,
            'building_id': building_ids
        }, crs=crs)

        gdf['area'] = gdf.geometry.area

        if district_attrs:
            for key, value in district_attrs.items():
                if key not in gdf.columns:
                    gdf[key] = value

        logger.debug("Vectorized %d polygons (%.2f m²)", len(gdf), gdf['area'].sum())

        return gdf

    def vectorize_boundaries(
        self,
        boundaries: np.ndarray,
        transform: Affine,
        crs: str = "EPSG:32650",
        district_attrs: Optional[dict] = None,
    ) -> gpd.GeoDataFrame:
        """Convert boundary skeleton raster to vector line features using pixel tracing."""
        if not boundaries.any():
            logger.warning("No boundaries to vectorize")
            return gpd.GeoDataFrame(columns=['geometry', 'length'], crs=crs)

        lines = []
        visited = np.zeros_like(boundaries, dtype=bool)

        def pixel_to_coords(row, col):
            """Convert pixel coordinates to world coordinates."""
            x, y = transform * (col + 0.5, row + 0.5)
            return (x, y)

        def get_neighbors(row, col):
            """Get 8-connected neighbors."""
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    r, c = row + dr, col + dc
                    if 0 <= r < boundaries.shape[0] and 0 <= c < boundaries.shape[1]:
                        if boundaries[r, c] == 1 and not visited[r, c]:
                            neighbors.append((r, c))
            return neighbors

        def trace_line(start_row, start_col):
            """Trace a line from a starting pixel."""
            path = [(start_row, start_col)]
            visited[start_row, start_col] = True

            current = (start_row, start_col)
            while True:
                neighbors = get_neighbors(current[0], current[1])
                if not neighbors:
                    break
                # Choose the first unvisited neighbor
                next_pixel = neighbors[0]
                visited[next_pixel[0], next_pixel[1]] = True
                path.append(next_pixel)
                current = next_pixel

            return path

        boundary_pixels = np.argwhere(boundaries == 1)

        for row, col in boundary_pixels:
            if not visited[row, col]:
                path = trace_line(row, col)

                if len(path) >= 2:
                    coords = [pixel_to_coords(r, c) for r, c in path]
                    line = LineString(coords)

                    if line.is_valid and line.length > 0:
                        line_simplified = line.simplify(
                            self.simplify_tolerance,
                            preserve_topology=True
                        )
                        if not line_simplified.is_empty and line_simplified.length > 0:
                            lines.append(line_simplified)

        if not lines:
            logger.warning("No lines extracted")
            return gpd.GeoDataFrame(columns=['geometry', 'length'], crs=crs)

        merged = linemerge(lines)
        if merged.geom_type == 'LineString':
            lines = [merged]
        elif merged.geom_type == 'MultiLineString':
            lines = list(merged.geoms)

        gdf = gpd.GeoDataFrame({'geometry': lines}, crs=crs)
        gdf['length'] = gdf.geometry.length

        if district_attrs:
            for key, value in district_attrs.items():
                if key not in gdf.columns:
                    gdf[key] = value

        logger.debug("Traced %d lines (%.2f m total)", len(gdf), gdf['length'].sum())

        return gdf

    def generate_voronoi_polygons(
        self,
        building_raster: np.ndarray,
        district_mask: np.ndarray,
        transform: Affine,
        crs: str = "EPSG:32650",
        district_attrs: Optional[dict] = None,
        visualize: bool = False,
        viz_interval: int = 1,
        debug_mode: bool = False,
    ) -> Tuple[gpd.GeoDataFrame, np.ndarray]:
        """
        Generate Voronoi diagram and convert to polygon features.
        
        Returns (voronoi_polygons_gdf, voronoi_partition_array).
        """
        voronoi, _ = self.generate_voronoi_from_raster(
            building_raster, district_mask, visualize=visualize, 
            viz_interval=viz_interval, debug_mode=debug_mode
        )

        voronoi_gdf = self.vectorize_voronoi_polygons(
            voronoi, transform, crs, district_attrs
        )

        return voronoi_gdf, voronoi

    def generate_voronoi_boundaries(
        self,
        building_raster: np.ndarray,
        district_mask: np.ndarray,
        transform: Affine,
        crs: str = "EPSG:32650",
        district_attrs: Optional[dict] = None,
        visualize: bool = False,
        viz_interval: int = 1,
        debug_mode: bool = False,
    ) -> Tuple[gpd.GeoDataFrame, np.ndarray]:
        """
        Generate Voronoi diagram and extract boundary lines.
        
        DEPRECATED: Use generate_voronoi_polygons for polygon output.
        """
        voronoi, original_buildings_mask = self.generate_voronoi_from_raster(
            building_raster, district_mask, visualize=visualize, 
            viz_interval=viz_interval, debug_mode=debug_mode
        )

        boundaries = self.extract_boundaries(voronoi, district_mask, original_buildings_mask)

        boundary_gdf = self.vectorize_boundaries(
            boundaries, transform, crs, district_attrs
        )

        return boundary_gdf, voronoi

