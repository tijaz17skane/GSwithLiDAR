import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
import ast

parser = argparse.ArgumentParser(description="Convert COLMAP images.txt to GPKG, PLY, and TXT for QGIS/3D viz, with camera-to-world conversion before writing.")
parser.add_argument("--txt_path", type=str, help="Path to images.txt", default="/mnt/data/tijaz/data/section3ready/sparse/0/images.txt")
parser.add_argument("--out_gpkg", type=str, help="Output GPKG file path", default="/mnt/data/tijaz/data/section3ready/sparse/0/colmapCams.gpkg")
parser.add_argument("--out_ply", type=str, help="Output PLY file path", default="/mnt/data/tijaz/data/section3ready/sparse/0/colmapCams.ply")
parser.add_argument("--out_txt", type=str, help="Output TXT file path", default="/mnt/data/tijaz/data/section3ready/sparse/0/colmapCams.txt")
args = parser.parse_args()

# Find header and read data
with open(args.txt_path, "r") as f:
    lines = f.readlines()

header_line = None
for i, line in enumerate(lines):
    if line.strip().startswith("#") and "IMAGE_ID" in line:
        header_line = i
        break

if header_line is None:
    raise ValueError("Header line with column names not found.")

columns = [col.strip() for col in lines[header_line].replace("#", "").split(",")]
data_lines = [l for l in lines[header_line+1:] if l.strip() and not l.strip().startswith("#")]

rows = []
for l in data_lines:
    parts = l.strip().split()
    if len(parts) < len(columns):
        continue  # skip incomplete lines
    row = dict(zip(columns, parts))
    rows.append(row)

df = pd.DataFrame(rows)

# Always convert TX, TY, TZ to float for geometry
for col in ["TX", "TY", "TZ"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Camera-to-world conversion: get camera position in world coordinates
if all(col in df.columns for col in ["QW", "QX", "QY", "QZ", "TX", "TY", "TZ"]):
    def colmap_to_world(qw, qx, qy, qz, tx, ty, tz):
        q_cam = [qx, qy, qz, qw]  # Quaternion (x, y, z, w)
        t_cam = np.array([tx, ty, tz])  # Translation vector
        R_cam = R.from_quat(q_cam).as_matrix()
        R_world = R_cam.T  # Inverse of rotation matrix
        t_world = -R_world @ t_cam  # Inverse translation
        T_world = np.eye(4)
        T_world[:3, :3] = R_world
        T_world[:3, 3] = t_world

        rot_world = R_world
        pos = t_world 

        return pos, rot_world
    positions, rotations = [], []
    for idx, row in df.iterrows():
        pos, rot = colmap_to_world(
            row["QW"], row["QX"], row["QY"], row["QZ"], row["TX"], row["TY"], row["TZ"]
        )
        positions.append(pos)
        rotations.append(rot)
    df["position_world"] = positions
    df["rotation_world"] = rotations
    # Use world position for geometry
    geometry = [Point(x, y, z) for x, y, z in positions]
else:
    geometry = [Point(x, y, z) for x, y, z in zip(df["TX"], df["TY"], df["TZ"])]

gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
gdf.to_file(args.out_gpkg, driver="GPKG", layer="camera_positions", index=False)
print(f"GPKG written to {args.out_gpkg}")

# Output as PLY (world positions)
def write_ply(positions, ply_path):
    with open(ply_path, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(len(positions)))
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in positions:
            f.write("{:.6f} {:.6f} {:.6f}\n".format(p[0], p[1], p[2]))

write_ply(positions, args.out_ply)
print(f"PLY written to {args.out_ply}")

# Output as TXT (world positions)
def write_txt(positions, txt_path):
    with open(txt_path, "w") as f:
        f.write("x y z\n")
        for p in positions:
            f.write("{:.6f} {:.6f} {:.6f}\n".format(p[0], p[1], p[2]))

write_txt(positions, args.out_txt)
print(f"TXT written to {args.out_txt}")