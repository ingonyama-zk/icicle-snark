use icicle_bn254::curve::ScalarField;
use std::env;
use std::fs::File;
use std::path::Path;
use std::io::{self, Write, Read, Result};
use std::mem;
use std::slice;

pub mod prove;
pub mod verify;
pub mod zkey;

pub fn compute_keys(
  mut key: ScalarField,
  inc: ScalarField,
  size: usize,
) -> Result<Vec<ScalarField>> {
  // Get temporary directory from environment variable
  let tmp_dir = env::var("TMPDIR").unwrap_or_else(|_| String::from("."));
  let file_path = Path::new(&tmp_dir).join(format!("precomputed_{}_{}.bin", size, inc));

  if file_path.exists() {
      let keys = load_from_binary_file(&file_path)?;
      println!("Loaded keys from binary file");
      return Ok(keys);
  }

  let mut keys = Vec::with_capacity(size);
  unsafe {
      keys.set_len(size);
  }
  for key_ref in keys.iter_mut().take(size) {
      *key_ref = key;
      key = key * inc;
  }

  save_to_binary_file(&keys, &file_path)?;

  Ok(keys)
}

fn save_to_binary_file(keys: &[ScalarField], file_path: &Path) -> io::Result<()> {
  let mut file = File::create(file_path)?;

  let bytes = unsafe {
      slice::from_raw_parts(keys.as_ptr() as *const u8, std::mem::size_of_val(keys))
  };

  file.write_all(bytes)?;

  Ok(())
}

fn load_from_binary_file(file_path: &Path) -> io::Result<Vec<ScalarField>> {
  let mut file = File::open(file_path)?;
  let mut buffer = Vec::new();
  file.read_to_end(&mut buffer)?;

  let scalar_size = mem::size_of::<ScalarField>();
  let num_scalars = buffer.len() / scalar_size;

  let scalars: Vec<ScalarField> = unsafe {
      slice::from_raw_parts(buffer.as_ptr() as *const ScalarField, num_scalars).to_vec()
  };

  Ok(scalars)
}
