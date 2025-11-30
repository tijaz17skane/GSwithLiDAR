import argparse
import sqlite3
import struct
import os

CAMERA_MODEL_MAP = {
    'SIMPLE_PINHOLE': 0,
    'PINHOLE': 1,
    'SIMPLE_RADIAL': 2,
    'RADIAL': 3,
    'OPENCV': 4,
    'OPENCV_FISHEYE': 5,
    'FULL_OPENCV': 6,
    'FOV': 7,
    'SIMPLE_RADIAL_FISHEYE': 8,
    'RADIAL_FISHEYE': 9,
    'THIN_PRISM_FISHEYE': 10,
}

def parse_cameras_txt(path):
    cameras = []
    with open(path, 'r') as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith('#'): continue
            parts=line.split()
            if len(parts) < 5: continue
            cam_id = int(parts[0])
            model_str = parts[1]
            width = int(parts[2]); height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            model = CAMERA_MODEL_MAP.get(model_str, None)
            cameras.append((cam_id, model_str, model, width, height, params))
    return cameras

def parse_images_txt(path):
    images = []
    with open(path,'r') as f:
        lines = f.readlines()
    idx=0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line or line.startswith('#'):
            idx += 1
            continue
        parts=line.split()
        if len(parts) < 10:
            idx += 1
            continue
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        camera_id = int(parts[8])
        name = parts[9]
        images.append((image_id, name, camera_id, qw, qx, qy, qz, tx, ty, tz))
        idx += 2  # skip potential POINTS2D line
    return images

def update_database(db_path, cameras, images):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Update cameras
    for cam_id, model_str, model, width, height, params in cameras:
        # Pack params as doubles
        params_blob = struct.pack('<' + 'd'*len(params), *params)
        # Determine model fallback if unknown
        if model is None:
            # Keep existing model if row exists
            cur.execute('SELECT model FROM cameras WHERE camera_id=?', (cam_id,))
            row = cur.fetchone()
            if row is not None:
                model = row[0]
            else:
                # Default to PINHOLE
                model = CAMERA_MODEL_MAP['PINHOLE']
        # prior_focal_length heuristic: use first param if available else keep existing or set 0
        prior_focal = 0
        if params:
            prior_focal = int(round(params[0]))
        cur.execute('SELECT prior_focal_length FROM cameras WHERE camera_id=?', (cam_id,))
        existing = cur.fetchone()
        if existing is not None:
            # Update
            cur.execute('UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=? WHERE camera_id=?',
                        (model, width, height, params_blob, prior_focal, cam_id))
        else:
            # Insert
            cur.execute('INSERT INTO cameras(camera_id, model, width, height, params, prior_focal_length) VALUES(?,?,?,?,?,?)',
                        (cam_id, model, width, height, params_blob, prior_focal))

    # Update images
    for (image_id, name, camera_id, qw, qx, qy, qz, tx, ty, tz) in images:
        # If name already exists under different ID, remove that row to free the unique constraint
        cur.execute('SELECT image_id FROM images WHERE name=?', (name,))
        existing_name_row = cur.fetchone()
        if existing_name_row is not None and existing_name_row[0] != image_id:
            cur.execute('DELETE FROM images WHERE image_id=?', (existing_name_row[0],))

        cur.execute('SELECT image_id FROM images WHERE image_id=?', (image_id,))
        if cur.fetchone() is not None:
            cur.execute('UPDATE images SET name=?, camera_id=?, prior_qw=?, prior_qx=?, prior_qy=?, prior_qz=?, prior_tx=?, prior_ty=?, prior_tz=? WHERE image_id=?',
                        (name, camera_id, qw, qx, qy, qz, tx, ty, tz, image_id))
        else:
            cur.execute('INSERT INTO images(image_id, name, camera_id, prior_qw, prior_qx, prior_qy, prior_qz, prior_tx, prior_ty, prior_tz) VALUES(?,?,?,?,?,?,?,?,?,?)',
                        (image_id, name, camera_id, qw, qx, qy, qz, tx, ty, tz))

    conn.commit()
    conn.close()


def main():
    ap = argparse.ArgumentParser(description='Populate/overwrite COLMAP database cameras and images tables from text files.')
    ap.add_argument('--db', required=True, help='Path to COLMAP database.db')
    ap.add_argument('--cameras', required=True, help='Path to cameras.txt')
    ap.add_argument('--images', required=True, help='Path to images.txt')
    args = ap.parse_args()

    if not os.path.isfile(args.db):
        raise FileNotFoundError(f'Database not found: {args.db}')
    cams = parse_cameras_txt(args.cameras)
    imgs = parse_images_txt(args.images)
    update_database(args.db, cams, imgs)
    print(f'Updated {len(cams)} cameras and {len(imgs)} images in {args.db}')

if __name__ == '__main__':
    main()
