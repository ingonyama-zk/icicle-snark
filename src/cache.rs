use icicle_bn254::curve::ScalarField;
use icicle_core::ntt::{get_root_of_unity, initialize_domain, release_domain, NTTInitDomainConfig};
use icicle_core::traits::{FieldImpl, MontgomeryConvertible};
use icicle_runtime::memory::{DeviceVec, HostOrDeviceSlice, HostSlice};
use icicle_runtime::stream::IcicleStream;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::Arc;
use std::{mem, slice};

use crate::conversions::from_u8;
use crate::file_wrapper::FileWrapper;
use crate::zkey::ZKey;
use crate::{F, G1, G2};

const W: [&str; 30] = [
    "0x0000000000000000000000000000000000000000000000000000000000000001",
    "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000",
    "0x30644e72e131a029048b6e193fd841045cea24f6fd736bec231204708f703636",
    "0x2b337de1c8c14f22ec9b9e2f96afef3652627366f8170a0a948dad4ac1bd5e80",
    "0x21082ca216cbbf4e1c6e4f4594dd508c996dfbe1174efb98b11509c6e306460b",
    "0x09c532c6306b93d29678200d47c0b2a99c18d51b838eeb1d3eed4c533bb512d0",
    "0x1418144d5b080fcac24cdb7649bdadf246a6cb2426e324bedb94fb05118f023a",
    "0x16e73dfdad310991df5ce19ce85943e01dcb5564b6f24c799d0e470cba9d1811",
    "0x07b0c561a6148404f086204a9f36ffb0617942546750f230c893619174a57a76",
    "0x0f1ded1ef6e72f5bffc02c0edd9b0675e8302a41fc782d75893a7fa1470157ce",
    "0x06fd19c17017a420ebbebc2bb08771e339ba79c0a8d2d7ab11f995e1bc2e5912",
    "0x027a358499c5042bb4027fd7a5355d71b8c12c177494f0cad00a58f9769a2ee2",
    "0x0931d596de2fd10f01ddd073fd5a90a976f169c76f039bb91c4775720042d43a",
    "0x006fab49b869ae62001deac878b2667bd31bf3e28e3a2d764aa49b8d9bbdd310",
    "0x2d965651cdd9e4811f4e51b80ddca8a8b4a93ee17420aae6adaa01c2617c6e85",
    "0x2d1ba66f5941dc91017171fa69ec2bd0022a2a2d4115a009a93458fd4e26ecfb",
    "0x00eeb2cb5981ed45649abebde081dcff16c8601de4347e7dd1628ba2daac43b7",
    "0x1bf82deba7d74902c3708cc6e70e61f30512eca95655210e276e5858ce8f58e5",
    "0x19ddbcaf3a8d46c15c0176fbb5b95e4dc57088ff13f4d1bd84c6bfa57dcdc0e0",
    "0x2260e724844bca5251829353968e4915305258418357473a5c1d597f613f6cbd",
    "0x26125da10a0ed06327508aba06d1e303ac616632dbed349f53422da953337857",
    "0x1ded8980ae2bdd1a4222150e8598fc8c58f50577ca5a5ce3b2c87885fcd0b523",
    "0x1ad92f46b1f8d9a7cda0ceb68be08215ec1a1f05359eebbba76dde56a219447e",
    "0x0210fe635ab4c74d6b7bcf70bc23a1395680c64022dd991fb54d4506ab80c59d",
    "0x0c9fabc7845d50d2852e2a0371c6441f145e0db82e8326961c25f1e3e32b045b",
    "0x2a734ebb326341efa19b0361d9130cd47b26b7488dc6d26eeccd4f3eb878331a",
    "0x1067569af1ff73b20113eff9b8d89d4a605b52b63d68f9ae1c79bd572f4e9212",
    "0x049ae702b363ebe85f256a9f6dc6e364b4823532f6437da2034afc4580928c44",
    "0x2a3c09f0a58a7e8500e0a7eb8ef62abc402d111e41112ed49bd61b6e725b19f0",
    "0x2260e724844bca5251829353968e4915305258418357473a5c1d597f613f6cbd",
];

#[derive(Clone)]
pub struct ZKeyCache {
    pub s_values: Vec<usize>,
    pub c_values: Vec<usize>,
    pub m_values: Vec<usize>,
    pub first_slice: Arc<DeviceVec<F>>,
    pub points_a: Arc<DeviceVec<G1>>,
    pub points_b1: Arc<DeviceVec<G1>>,
    pub points_b: Arc<DeviceVec<G2>>,
    pub points_h: Arc<DeviceVec<G1>>,
    pub points_c: Arc<DeviceVec<G1>>,
    pub keys: Arc<DeviceVec<F>>,
    pub zkey: ZKey,
}

#[derive(Default)]
pub struct CacheManager {
    cache: HashMap<String, ZKeyCache>,
    last_key: String,
}

impl CacheManager {
    pub fn compute(&mut self, zkey_path: &str) -> Result<ZKeyCache, Box<dyn std::error::Error>> {
        let mut stream = IcicleStream::create().unwrap();

        let (fd_zkey, sections_zkey) = FileWrapper::read_bin_file(zkey_path, "zkey", 2).unwrap();

        let mut zkey_file = FileWrapper::new(fd_zkey).unwrap();

        let zkey = zkey_file.read_zkey_header(&sections_zkey[..]).unwrap();

        let buff_coeffs = zkey_file.read_section(&sections_zkey[..], 4).unwrap();

        let s_coef = 4 * 3 + zkey.n8r;
        let n_coef = (buff_coeffs.len() - 4) / s_coef;

        let mut first_slice = Vec::with_capacity(n_coef);
        let mut s_values = Vec::with_capacity(n_coef);
        let mut c_values = Vec::with_capacity(n_coef);
        let mut m_values = Vec::with_capacity(n_coef);

        unsafe {
            first_slice.set_len(n_coef);
            s_values.set_len(n_coef);
            c_values.set_len(n_coef);
            m_values.set_len(n_coef);
        }
        let n8 = 32;

        s_values
            .par_iter_mut()
            .zip(c_values.par_iter_mut())
            .zip(m_values.par_iter_mut())
            .zip(first_slice.par_iter_mut())
            .enumerate()
            .for_each(|(i, (((s_val, c_val), m_val), coef_val))| {
                let start = 4 + i * s_coef;
                let buff_coef = &buff_coeffs[start..start + s_coef];

                let s =
                    u32::from_le_bytes([buff_coef[8], buff_coef[9], buff_coef[10], buff_coef[11]])
                        as usize;
                let c = u32::from_le_bytes([buff_coef[4], buff_coef[5], buff_coef[6], buff_coef[7]])
                    as usize;
                let m = buff_coef[0];
                let coef = ScalarField::from_bytes_le(&buff_coef[12..12 + n8]);

                *s_val = s;
                *c_val = c;
                *m_val = m as usize;
                *coef_val = coef;
            });

        let power = zkey.power + 1;
        let inc = F::from_hex(W[power]);
        let keys = CacheManager::pre_compute_keys(F::one(), inc, zkey.domain_size).unwrap();
        let mut d_keys = DeviceVec::device_malloc_async(zkey.domain_size, &stream).unwrap();
        d_keys
            .copy_from_host_async(HostSlice::from_slice(&keys), &stream)
            .unwrap();

        let points_a = zkey_file.read_section(&sections_zkey, 5).unwrap();
        let points_b1 = zkey_file.read_section(&sections_zkey, 6).unwrap();
        let points_b = zkey_file.read_section(&sections_zkey, 7).unwrap();
        let points_c = zkey_file.read_section(&sections_zkey, 8).unwrap();
        let points_h = zkey_file.read_section(&sections_zkey, 9).unwrap();

        let points_a = from_u8(points_a);
        let points_b1 = from_u8(points_b1);
        let points_b = from_u8(points_b);
        let points_c = from_u8(points_c);
        let points_h = from_u8(points_h);

        let mut d_points_a = DeviceVec::device_malloc_async(points_a.len(), &stream).unwrap();
        let mut d_points_b1 = DeviceVec::device_malloc_async(points_b1.len(), &stream).unwrap();
        let mut d_points_b = DeviceVec::device_malloc_async(points_b.len(), &stream).unwrap();
        let mut d_points_c = DeviceVec::device_malloc_async(points_c.len(), &stream).unwrap();
        let mut d_points_h = DeviceVec::device_malloc_async(points_h.len(), &stream).unwrap();
        let mut d_first_slice = DeviceVec::device_malloc_async(first_slice.len(), &stream).unwrap();

        let points_a = HostSlice::from_slice(points_a);
        let points_b1 = HostSlice::from_slice(points_b1);
        let points_b = HostSlice::from_slice(points_b);
        let points_c = HostSlice::from_slice(points_c);
        let points_h = HostSlice::from_slice(points_h);
        let first_slice = HostSlice::from_slice(&first_slice);

        d_points_a.copy_from_host_async(points_a, &stream).unwrap();
        d_points_b1
            .copy_from_host_async(points_b1, &stream)
            .unwrap();
        d_points_b.copy_from_host_async(points_b, &stream).unwrap();
        d_points_c.copy_from_host_async(points_c, &stream).unwrap();
        d_points_h.copy_from_host_async(points_h, &stream).unwrap();
        d_first_slice
            .copy_from_host_async(first_slice, &stream)
            .unwrap();

        G1::from_mont(&mut d_points_a, &stream);
        G1::from_mont(&mut d_points_b1, &stream);
        G2::from_mont(&mut d_points_b, &stream);
        G1::from_mont(&mut d_points_c, &stream);
        G1::from_mont(&mut d_points_h, &stream);

        ScalarField::from_mont(&mut d_first_slice, &stream);

        stream.synchronize().unwrap();
        stream.destroy().unwrap();

        let cache_entry = ZKeyCache {
            s_values,
            c_values,
            m_values,
            zkey,
            first_slice: Arc::new(d_first_slice),
            points_a: Arc::new(d_points_a),
            points_b1: Arc::new(d_points_b1),
            points_b: Arc::new(d_points_b),
            points_c: Arc::new(d_points_c),
            points_h: Arc::new(d_points_h),
            keys: Arc::new(d_keys),
        };

        Ok(cache_entry)
    }
    pub fn get_cache(&mut self, key: &str) -> &mut ZKeyCache {
        let cache = self.cache.get_mut(key).unwrap();

        if !self.last_key.is_empty() && !key.eq(&self.last_key) {
            release_domain::<F>().unwrap();
        }

        let domain: F = get_root_of_unity(cache.points_a.len() as u64);
        let cfg = NTTInitDomainConfig::default();
        initialize_domain(domain, &cfg).unwrap();

        self.last_key = key.to_string();

        cache
    }
    pub fn insert_cache(&mut self, key: &str, cache: ZKeyCache) {
        self.cache.insert(key.to_string(), cache);
    }
    pub fn contains(&self, key: &str) -> bool {
        self.cache.contains_key(key)
    }
    fn pre_compute_keys(
        mut key: ScalarField,
        inc: ScalarField,
        size: usize,
    ) -> io::Result<Vec<ScalarField>> {
        let file_path = format!("precomputed_{}_{}.bin", size, inc);
        let file: &Path = Path::new(&file_path);

        if file.exists() {
            let keys = CacheManager::load_from_binary_file(file)?;
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

        CacheManager::save_to_binary_file(&keys, file)?;

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
}
