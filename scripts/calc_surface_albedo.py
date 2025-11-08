# -*- coding: utf-8 -*-
"""
Calculate surface albedo from Landsat 8 Surface Reflectance product.

This script implements the broad-band surface albedo calculation using the
formula proposed by Liang (2000) with USGS recommended parameters for Landsat 8.

Reference:
    Liang, S. (2000). Narrowband to broadband conversions of land surface albedo I:
    Algorithms. Remote Sensing of Environment, 76(2), 213-238.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import rasterio


def calc_surface_albedo_landsat8(band_data: dict, nodata_value=-9999) -> np.ndarray:
    """
    Calculate surface albedo using Liang (2000) formula for Landsat 8.

    Formula:
        α = 0.356*Blue + 0.130*Red + 0.373*NIR + 0.085*SWIR1 + 0.072*SWIR2 - 0.0018

    where α is the broad-band surface albedo.

    Args:
        band_data: Dictionary containing band arrays with keys:
                   'SR_B2' (Band 2 - Blue), 'SR_B4' (Band 4 - Red), 
                   'SR_B5' (Band 5 - NIR), 'SR_B6' (Band 6 - SWIR1), 
                   'SR_B7' (Band 7 - SWIR2)
                   Values should be surface reflectance (0-1 range)
        nodata_value: Value to use for nodata pixels

    Returns:
        Surface albedo array with values in range [0, 1]
    """
    # Liang (2000) coefficients for Landsat
    coefficients = {
        'SR_B2': 0.356,   # Band 2 (Blue)
        'SR_B4': 0.130,   # Band 4 (Red)
        'SR_B5': 0.373,   # Band 5 (NIR)
        'SR_B6': 0.085,   # Band 6 (SWIR1)
        'SR_B7': 0.072    # Band 7 (SWIR2)
    }
    constant = -0.0018

    # Check if all required bands are present
    required_bands = ['SR_B2', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
    for band in required_bands:
        if band not in band_data:
            raise ValueError(f"Missing required band: {band}")

    # Create mask for valid pixels (all bands must be valid)
    valid_mask = np.ones(band_data['SR_B2'].shape, dtype=bool)
    for band in required_bands:
        valid_mask &= (band_data[band] != nodata_value)
        valid_mask &= np.isfinite(band_data[band])

    # Calculate albedo
    albedo = np.full(band_data['SR_B2'].shape, nodata_value, dtype=np.float32)

    if np.any(valid_mask):
        albedo[valid_mask] = (
            coefficients['SR_B2'] * band_data['SR_B2'][valid_mask] +
            coefficients['SR_B4'] * band_data['SR_B4'][valid_mask] +
            coefficients['SR_B5'] * band_data['SR_B5'][valid_mask] +
            coefficients['SR_B6'] * band_data['SR_B6'][valid_mask] +
            coefficients['SR_B7'] * band_data['SR_B7'][valid_mask] +
            constant
        )

        # Clip values to valid albedo range [0, 1]
        albedo[valid_mask] = np.clip(albedo[valid_mask], 0.0, 1.0)

    return albedo


def read_landsat8_bands(image_path: str, scale_factor=0.0000275, offset=-0.2) -> tuple:
    """
    Read Landsat 8 bands from a multi-band GeoTIFF file and convert to surface reflectance.

    Args:
        image_path: Path to the Landsat 8 SR product GeoTIFF
        scale_factor: Scale factor for converting DN to reflectance (default: 0.0000275 for Collection 2)
        offset: Offset for converting DN to reflectance (default: -0.2 for Collection 2)

    Returns:
        Tuple of (band_data dict, profile dict, nodata_value)
        
    Note:
        Landsat 8 Collection 2: reflectance = DN * 0.0000275 - 0.2
        Landsat 8 Collection 1: reflectance = DN * 0.0001
    """
    with rasterio.open(image_path) as src:
        profile = src.profile.copy()

        # Landsat 8 band mapping (assuming standard band order)
        # Band 1: Coastal Aerosol
        # Band 2: Blue
        # Band 3: Green
        # Band 4: Red
        # Band 5: NIR
        # Band 6: SWIR1
        # Band 7: SWIR2

        band_mapping = {
            'SR_B2': 2,   # Band 2 (Blue)
            'SR_B4': 4,   # Band 4 (Red)
            'SR_B5': 5,   # Band 5 (NIR)
            'SR_B6': 6,   # Band 6 (SWIR1)
            'SR_B7': 7    # Band 7 (SWIR2)
        }

        # Get nodata value from metadata
        original_nodata = src.nodata if src.nodata is not None else 0
        
        band_data = {}
        for band_name, band_idx in band_mapping.items():
            if band_idx <= src.count:
                # Read band as float32
                dn_values = src.read(band_idx).astype(np.float32)
                
                # Create mask for valid data (not nodata)
                valid_mask = (dn_values != original_nodata) if original_nodata is not None else np.ones_like(dn_values, dtype=bool)
                
                # Convert DN to surface reflectance: reflectance = DN * scale_factor + offset
                reflectance = np.full_like(dn_values, -9999.0, dtype=np.float32)
                reflectance[valid_mask] = dn_values[valid_mask] * scale_factor + offset
                
                band_data[band_name.upper()] = reflectance
                
                print(f"  {band_name}: DN range [{np.min(dn_values[valid_mask]):.0f}, {np.max(dn_values[valid_mask]):.0f}] "
                      f"-> Reflectance range [{np.min(reflectance[valid_mask]):.4f}, {np.max(reflectance[valid_mask]):.4f}]")
            else:
                raise ValueError(f"Band {band_idx} not found in image (total bands: {src.count})")

        # Use -9999 as nodata value for output
        nodata_value = -9999

    return band_data, profile, nodata_value


def save_albedo(albedo: np.ndarray, output_path: str, profile: dict, nodata_value=-9999):
    """
    Save surface albedo to a GeoTIFF file.

    Args:
        albedo: Surface albedo array
        output_path: Output file path
        profile: Rasterio profile from input image
        nodata_value: NoData value
    """
    # Update profile for single-band output
    profile.update({
        'count': 1,
        'dtype': 'float32',
        'nodata': nodata_value,
        'compress': 'lzw'
    })

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(albedo, 1)
        dst.set_band_description(1, 'Surface Albedo')

    print(f"Surface albedo saved to: {output_path}")


def process_landsat8_albedo(input_path: str, output_path: str = None, 
                            scale_factor: float = 0.0000275, offset: float = -0.2):
    """
    Process Landsat 8 SR product to calculate and save surface albedo.

    Args:
        input_path: Path to input Landsat 8 SR GeoTIFF
        output_path: Path to output albedo GeoTIFF (optional)
        scale_factor: Scale factor for DN to reflectance conversion
        offset: Offset for DN to reflectance conversion
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Generate output path if not provided
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_albedo.tif"
    else:
        output_path = Path(output_path)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading Landsat 8 bands from: {input_path}")
    print(f"Using scale_factor={scale_factor}, offset={offset}")
    band_data, profile, nodata_value = read_landsat8_bands(str(input_path), scale_factor, offset)

    print("Calculating surface albedo...")
    albedo = calc_surface_albedo_landsat8(band_data, nodata_value)

    # Print statistics
    valid_mask = (albedo != nodata_value) & np.isfinite(albedo)
    if np.any(valid_mask):
        print("Albedo statistics:")
        print(f"  Min:  {np.min(albedo[valid_mask]):.4f}")
        print(f"  Max:  {np.max(albedo[valid_mask]):.4f}")
        print(f"  Mean: {np.mean(albedo[valid_mask]):.4f}")
        print(f"  Std:  {np.std(albedo[valid_mask]):.4f}")

    print(f"Saving surface albedo to: {output_path}")
    save_albedo(albedo, str(output_path), profile, nodata_value)

    return albedo, output_path


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Calculate surface albedo from Landsat 8 Surface Reflectance product.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (Collection 2)
    python calc_surface_albedo.py --input input_landsat8_sr.tif --output output_albedo.tif

    # Collection 1 data (use different scale factor)
    python calc_surface_albedo.py --input input_landsat8_sr.tif --output output_albedo.tif \\
        --scale-factor 0.0001 --offset 0.0

Note:
    Input file should be a Landsat 8 Surface Reflectance (SR) product with DN values.
    The script converts DN to reflectance using: reflectance = DN * scale_factor + offset
    
    Landsat 8 Collection 2 (default): scale_factor=0.0000275, offset=-0.2
    Landsat 8 Collection 1: scale_factor=0.0001, offset=0.0
    
    The script uses Liang (2000) formula with USGS recommended parameters.
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input Landsat 8 SR GeoTIFF file'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output albedo GeoTIFF file (default: input_albedo.tif)'
    )

    parser.add_argument(
        '--scale-factor',
        type=float,
        default=0.0000275,
        help='Scale factor for DN to reflectance conversion (default: 0.0000275 for Collection 2)'
    )

    parser.add_argument(
        '--offset',
        type=float,
        default=-0.2,
        help='Offset for DN to reflectance conversion (default: -0.2 for Collection 2)'
    )

    args = parser.parse_args()

    try:
        process_landsat8_albedo(args.input, args.output, args.scale_factor, args.offset)
        print("Processing completed successfully!")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, IOError) as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
