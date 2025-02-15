# // SPDX-License-Identifier: blessing
# // d a r k s h a p e s

from sqlite3 import connect
from contextlib import contextmanager


@contextmanager
def managed_connection(db_name):
    active_conn = connect(db_name)
    try:
        yield active_conn
    finally:
        active_conn.close()


def create_table(end_conn):
    cursor = end_conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS parameters (
    id TEXT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    status INTEGER CHECK (status IN (1, 0)),
    type TEXT,
    timesteps TEXT[],
    noise_seed INTEGER,
    output_type TEXT,
    denoising_end REAL,
    num_inference_steps INTEGER,
    guidance_scale REAL,
    eta REAL,
    width INTEGER,
    height INTEGER,
    safety_checker BOOLEAN,
    model TEXT,
    vae_file TEXT,
    lora_file TEXT,  -- Fixed syntax here
    active_gpu TEXT,
    prompt TEXT,
    negative_prompt TEXT,
    _diffusers_version TEXT,
    force_zeros_for_empty_prompt BOOLEAN,
    class_name TEXT
  )""")


with managed_connection("combo_history.db") as start_conn:
    create_table(start_conn)
