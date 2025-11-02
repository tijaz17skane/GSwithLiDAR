import sqlite3

def create_empty_database(db_path: str):
    """
    Create an empty COLMAP-style SQLite database at the given path.

    Args:
        db_path (str): Path to the SQLite database file to create.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS cameras (
        camera_id INTEGER PRIMARY KEY,
        model INTEGER NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        params BLOB,
        prior_focal_length INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS images (
        image_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        camera_id INTEGER NOT NULL,
        prior_qw REAL,
        prior_qx REAL,
        prior_qy REAL,
        prior_qz REAL,
        prior_tx REAL,
        prior_ty REAL,
        prior_tz REAL
    );

    CREATE TABLE IF NOT EXISTS keypoints (
        image_id INTEGER PRIMARY KEY,
        rows INTEGER NOT NULL,
        cols INTEGER NOT NULL,
        data BLOB
    );

    CREATE TABLE IF NOT EXISTS descriptors (
        image_id INTEGER PRIMARY KEY,
        rows INTEGER NOT NULL,
        cols INTEGER NOT NULL,
        data BLOB
    );

    CREATE TABLE IF NOT EXISTS matches (
        pair_id INTEGER PRIMARY KEY,
        rows INTEGER NOT NULL,
        cols INTEGER NOT NULL,
        data BLOB
    );

    CREATE TABLE IF NOT EXISTS two_view_geometries (
        pair_id INTEGER PRIMARY KEY,
        rows INTEGER NOT NULL,
        cols INTEGER NOT NULL,
        data BLOB,
        config INTEGER NOT NULL,
        F BLOB,
        E BLOB,
        H BLOB,
        qvec BLOB,
        tvec BLOB
    );

    CREATE TABLE IF NOT EXISTS pose_priors (
        image_id INTEGER PRIMARY KEY,
        position BLOB,
        coordinate_system INTEGER NOT NULL,
        position_covariance BLOB
    );
    """)

    conn.commit()
    conn.close()

    #print(f"✅ Empty COLMAP-style database created at: {db_path}")
