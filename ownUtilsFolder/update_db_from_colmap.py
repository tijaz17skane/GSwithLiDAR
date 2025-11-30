#!/usr/bin/env python3
"""
Update or create an SQLite `database.db` using COLMAP `cameras.txt` and `images.txt`.

This script will:
 - parse `cameras.txt` and `images.txt` (COLMAP text format)
 - backup the existing database (if present) to `database.db.bak.TIMESTAMP`
 - replace the `cameras` and `images` tables with the parsed contents

Usage:
    python update_db_from_colmap.py --cameras cameras.txt --images images.txt --database database.db

The script creates simple schemas for cameras and images. `params` in cameras is stored as JSON.
"""
import argparse
import sqlite3
import shutil
import os
import time
import json


def parse_cameras(path):
    cams = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            cams.append({
                'camera_id': cam_id,
                'model': model,
                'width': width,
                'height': height,
                'params': params,
            })
    return cams


def parse_images(path):
    imgs = []
    with open(path, 'r') as f:
        lines = [l.rstrip('\n') for l in f]
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        i += 1
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        if len(parts) < 9:
            # malformed
            continue
        img_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        name = parts[9] if len(parts) > 9 else ''
        imgs.append({
            'image_id': img_id,
            'qw': qw,
            'qx': qx,
            'qy': qy,
            'qz': qz,
            'tx': tx,
            'ty': ty,
            'tz': tz,
            'camera_id': cam_id,
            'name': name,
        })
        # skip optional POINTS2D line following an image line (if present)
        if i < n and lines[i].strip() and not lines[i].strip().startswith('#'):
            # If it looks like POINTS2D (triplets) we skip it; else keep going
            # We assume the next line is POINTS2D[] and skip it.
            # Heuristic: POINTS2D lines usually contain integers and floats; but safe to skip one line.
            # Many COLMAP exports keep POINTS2D empty (no tokens) so this is safe.
            i += 1
    return imgs


def backup_db(db_path):
    if os.path.exists(db_path):
        ts = time.strftime('%Y%m%dT%H%M%S')
        bak = db_path + '.bak.' + ts
        shutil.copy2(db_path, bak)
        print(f'Backed up existing database to: {bak}')


def recreate_tables(conn):
    cur = conn.cursor()
    # Ensure foreign keys are enforced
    cur.execute('PRAGMA foreign_keys = ON')
    # Simple cameras table
    cur.execute('DROP TABLE IF EXISTS cameras')
    cur.execute('''
    CREATE TABLE cameras (
        camera_id INTEGER PRIMARY KEY,
        model TEXT,
        width INTEGER,
        height INTEGER,
        params TEXT
    )
    ''')

    # Simple images table with foreign key to cameras.camera_id
    cur.execute('DROP TABLE IF EXISTS images')
    cur.execute('''
    CREATE TABLE images (
        image_id INTEGER PRIMARY KEY,
        name TEXT,
        camera_id INTEGER,
        qw REAL, qx REAL, qy REAL, qz REAL,
        tx REAL, ty REAL, tz REAL,
        FOREIGN KEY(camera_id) REFERENCES cameras(camera_id) ON DELETE SET NULL
    )
    ''')
    # index on camera_id for faster joins
    cur.execute('CREATE INDEX IF NOT EXISTS idx_images_camera_id ON images(camera_id)')
    conn.commit()


def insert_cameras(conn, cams):
    cur = conn.cursor()
    for c in cams:
        cur.execute('INSERT INTO cameras(camera_id, model, width, height, params) VALUES (?, ?, ?, ?, ?)',
                    (c['camera_id'], c['model'], c['width'], c['height'], json.dumps(c['params'])))
    conn.commit()


def insert_images(conn, imgs):
    cur = conn.cursor()
    # collect existing camera ids to avoid FK errors
    cur.execute('SELECT camera_id FROM cameras')
    existing = set(r[0] for r in cur.fetchall())
    for im in imgs:
        cam_id = im['camera_id'] if im['camera_id'] in existing else None
        if cam_id is None:
            print(f"Warning: image {im['image_id']} references unknown camera_id {im['camera_id']}; setting camera_id=NULL")
        cur.execute('''INSERT INTO images(image_id, name, camera_id, qw, qx, qy, qz, tx, ty, tz)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (im['image_id'], im['name'], cam_id, im['qw'], im['qx'], im['qy'], im['qz'], im['tx'], im['ty'], im['tz']))
    conn.commit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cameras', required=True, help='Path to cameras.txt')
    parser.add_argument('--images', required=True, help='Path to images.txt')
    parser.add_argument('--database', required=True, help='Path to database.db to create/update')
    args = parser.parse_args()

    cams = parse_cameras(args.cameras)
    imgs = parse_images(args.images)

    print(f'Parsed {len(cams)} cameras and {len(imgs)} images')

    backup_db(args.database)

    conn = sqlite3.connect(args.database)
    # Ensure the connection enforces foreign key constraints
    conn.execute('PRAGMA foreign_keys = ON')
    try:
        recreate_tables(conn)
        insert_cameras(conn, cams)
        insert_images(conn, imgs)
        print('Database updated successfully.')
    finally:
        conn.close()


if __name__ == '__main__':
    main()
