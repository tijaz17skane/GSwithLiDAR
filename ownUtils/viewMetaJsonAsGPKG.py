import json
import pandas as pd
import argparse
import geopandas as gpd
from shapely.geometry import Point

parser = argparse.ArgumentParser(description="View meta.json as a table and save to GPKG for QGIS.")
parser.add_argument("--json_path", type=str, default="/mnt/data/tijaz/data/section_3useful/meta.json", help="Path to meta.json")
parser.add_argument("--out_gpkg", type=str, required=True, help="Output GPKG file path")
args = parser.parse_args()

with open(args.json_path, "r") as f:
    data = json.load(f)

rows = []
for img in data.get("images", []):
    sensor_id = img.get("sensor_id", "")
    path = img.get("path", "")
    pose = img.get("pose", {})
    translation = pose.get("translation", ["", "", ""])
    orientation = pose.get("orientation_xyzw", ["", "", "", ""])
    # Parse translation as floats if possible
    if len(translation) == 3:
        try:
            x, y, z = map(float, translation)
        except Exception:
            x, y, z = None, None, None
    else:
        x, y, z = None, None, None
    orientation_str = ", ".join(str(x) for x in orientation)
    rows.append({
        "sensor_id": sensor_id,
        "path": path,
        "x": x,
        "y": y,
        "z": z,
        "orientation_xyzw": orientation_str
    })

df = pd.DataFrame(rows, columns=["sensor_id", "path", "x", "y", "z", "orientation_xyzw"])
gdf = gpd.GeoDataFrame(
    df,
    geometry=[Point(x, y, z) if x is not None and y is not None and z is not None else None for x, y, z in zip(df.x, df.y, df.z)],
    crs="EPSG:4326"  # Change CRS if your coordinates are not WGS84
)
gdf.to_file(args.out_gpkg, driver="GPKG", layer="camera_positions", index=False)
print(f"GPKG written to {args.out_gpkg}")