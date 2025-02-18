# # // SPDX-License-Identifier: blessing
# # // d a r k s h a p e s

from sqlite3 import connect
from contextlib import contextmanager


@contextmanager
def managed_connection(db_name):
    active_conn = connect(db_name)
    try:
        yield active_conn
    finally:
        active_conn.close()


def create_table(open_conn):
    cursor = open_conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS parameters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status INTEGER CHECK (status IN (1, 0)),
    type TEXT,
    prompt TEXT,
    negative_prompt TEXT,
    active_gpu TEXT,
    noise_seed INTEGER,
    num_inference_steps INTEGER,
    guidance_scale REAL,
    width INTEGER,
    height INTEGER,
    denoising_end REAL,
    safety_checker BOOLEAN,
    eta REAL,
    timesteps TEXT[],
    _diffusers_version TEXT,
    force_zeros_for_empty_prompt BOOLEAN,
    class_name TEXT,
    output_type TEXT,
    model TEXT,
    vae_model TEXT,
    lora_model TEXT,
    model_hash TEXT,
    vae_hash TEXT,
    lora_hash TEXT
    )""")


def insert_into_db(conn, data):
    columns = ", ".join(data.keys())
    placeholders = ":" + ", :".join(data.keys())

    # cursor.execute(f"INSERT INTO parameters ({columns}) VALUES ({placeholders})", values_tuple)
    query = f"INSERT INTO parameters ({columns}) VALUES ({placeholders})"

    with conn:
        cursor = conn.cursor()
        cursor.execute(query, data)


# Use managed_connection and insert
def save_user_set_to_db(user_set, database_name="combo_history.db"):
    with managed_connection(database_name) as conn:
        create_table(conn)
        insert_into_db(conn, user_set)


def retrieve_from_db(database_name="combo_history.db"):
    with managed_connection(database_name) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM parameters")
        rows = cursor.fetchall()
        for row in rows:
            print(row)
