import { sqlite3 } from 'sqlite3';

const db = new sqlite3.Database(':memory:', (err) => {
  if (err) console.error(err.message);
  else console.log('Connected to the SQLite database.');
});

// Create table for storing requests
db.run(`
  CREATE TABLE IF NOT EXISTS requests (
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
    lora_file TEXT?,
    active_gpu TEXT,
    prompt TEXT,
    negative_prompt TEXT,
    _diffusers_version TEXT,
    force_zeros_for_empty_prompt BOOLEAN,
    class_name TEXT
  )
`);