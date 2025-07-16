use serde::{Deserialize, Deserializer};
use crate::{
    groth16::prove::Proof,
    G1, G2, ProjectiveG1,
    utils::{deserialize_g1_affine, deserialize_g2_affine},
};
use icicle_bn254::curve::ScalarField;
use icicle_bn254::pairing::PairingTargetField;
use icicle_core::pairing::pairing;
use icicle_core::traits::FieldImpl;
use num_bigint::BigUint;

#[cfg(feature = "android")]
use std::time::Instant;
#[cfg(feature = "android")]
use log::debug;

#[derive(Debug)]
pub struct VerificationKey {
    pub vk_alpha_1: G1,
    pub vk_beta_2: G2,
    pub vk_gamma_2: G2,
    pub vk_delta_2: G2,
    pub ic: Vec<G1>,
    pub n_public: usize,
}

impl<'de> Deserialize<'de> for VerificationKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            vk_alpha_1: Vec<String>,
            vk_beta_2: Vec<Vec<String>>,
            vk_gamma_2: Vec<Vec<String>>,
            vk_delta_2: Vec<Vec<String>>,
            IC: Vec<Vec<String>>,
            nPublic: usize,
        }
        let helper = Helper::deserialize(deserializer)?;
        Ok(VerificationKey {
            vk_alpha_1: crate::utils::deserialize_g1_affine(&helper.vk_alpha_1),
            vk_beta_2: crate::utils::deserialize_g2_affine(&helper.vk_beta_2),
            vk_gamma_2: crate::utils::deserialize_g2_affine(&helper.vk_gamma_2),
            vk_delta_2: crate::utils::deserialize_g2_affine(&helper.vk_delta_2),
            ic: helper.IC.iter().map(|v| crate::utils::deserialize_g1_affine(v)).collect(),
            n_public: helper.nPublic,
        })
    }
}

pub fn verify(
  proof: &Proof,
  public: &[String],
  verification_key: &VerificationKey,
) -> bool {
    #[cfg(feature = "android")]
    let start = Instant::now();
    
    #[cfg(feature = "android")]
    let start_deserialize = Instant::now();
    let pi_a = deserialize_g1_affine(&proof.pi_a);
    let pi_b = deserialize_g2_affine(&proof.pi_b);
    let pi_c = deserialize_g1_affine(&proof.pi_c);
    #[cfg(feature = "android")]
    let duration = start_deserialize.elapsed();
    #[cfg(feature = "android")]
    debug!("Deserializing proof elements took {:?}", duration);
    
    let n_public = verification_key.n_public;  
    let ic = verification_key.ic.clone();
  
    #[cfg(feature = "android")]
    let start_public = Instant::now();
    let mut public_scalars = Vec::with_capacity(n_public);
    for s in public.iter().take(n_public) {
        let hex = BigUint::parse_bytes(s.as_bytes(), 10).unwrap();
        let scalar = ScalarField::from_bytes_le(&hex.to_bytes_le());
        public_scalars.push(scalar);
    }
    #[cfg(feature = "android")]
    let duration = start_public.elapsed();
    #[cfg(feature = "android")]
    debug!("Processing public inputs took {:?}", duration);
  
    #[cfg(feature = "android")]
    let start_commit = Instant::now();
    let mut cpub = ic[0].to_projective();
    for i in 0..public_scalars.len() {
        cpub = cpub + ic[i + 1].to_projective() * public_scalars[i];
    }
    #[cfg(feature = "android")]
    let duration = start_commit.elapsed();
    #[cfg(feature = "android")]
    debug!("Computing public commitment took {:?}", duration);
  
    let neg_pi_a = ProjectiveG1::zero() - pi_a.to_projective();
  
    #[cfg(feature = "android")]
    let start_pairing = Instant::now();
    let first = pairing(&neg_pi_a.into(), &pi_b).unwrap();
    let second = pairing(&cpub.into(), &verification_key.vk_gamma_2).unwrap();
    let third = pairing(&pi_c, &verification_key.vk_delta_2).unwrap();
    let fourth = pairing(&verification_key.vk_alpha_1, &verification_key.vk_beta_2).unwrap();
    #[cfg(feature = "android")]
    let duration = start_pairing.elapsed();
    #[cfg(feature = "android")]
    debug!("Computing pairings took {:?}", duration);
  
    let result = PairingTargetField::one() == first * second * third * fourth;
    
    #[cfg(feature = "android")]
    let total_duration = start.elapsed();
    #[cfg(feature = "android")]
    if result {
        log::info!("Groth16 verification successful in {:?}", total_duration);
    } else {
        log::error!("Groth16 verification failed in {:?}", total_duration);
    }
    
    result
}