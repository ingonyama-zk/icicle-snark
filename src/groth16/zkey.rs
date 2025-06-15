use crate::zkey::{read_g1, read_g2};
use crate::{F, C2, C1, ProjectiveG1, ProjectiveG2};
use crate::utils::from_affine_mont;
use crate::file_wrapper::{FileWrapper, Section};

#[derive(Clone, Debug)]
pub struct ZKeyHeader {
    pub n8q: usize,
    pub q: F,
    pub n8r: usize,
    pub r: F,
    pub n_vars: usize,
    pub n_public: usize,
    pub domain_size: usize,
    pub power: usize,
    pub vk_alpha_1: ProjectiveG1,
    pub vk_beta_1: ProjectiveG1,
    pub vk_beta_2: ProjectiveG2,
    pub vk_gamma_2: ProjectiveG2,
    pub vk_delta_1: ProjectiveG1,
    pub vk_delta_2: ProjectiveG2,
}

impl ZKeyHeader {
    pub fn read_header(fd: &mut FileWrapper, sections: &[Vec<Section>]) -> Self {
        fd.start_read_unique_section(sections, 2).unwrap();
        let n8q = fd.read_u32_le().unwrap() as usize;
        let q = fd.read_big_int(n8q, None).unwrap();

        let n8r = fd.read_u32_le().unwrap() as usize;
        let r = fd.read_big_int(n8r, None).unwrap();
        let n_vars = fd.read_u32_le().unwrap() as usize;
        let n_public = fd.read_u32_le().unwrap() as usize;
        let domain_size = fd.read_u32_le().unwrap() as usize;
        let power = (domain_size as f32).log2() as usize;

        let vk_alpha_1 = read_g1(fd);
        let vk_beta_1 = read_g1(fd);
        let vk_beta_2 = read_g2(fd);
        let vk_gamma_2 = read_g2(fd);
        let vk_delta_1 = read_g1(fd);
        let vk_delta_2 = read_g2(fd);

        let mut mont_points_g1 = [vk_alpha_1, vk_beta_1, vk_delta_1];
        let mut mont_points_g2 = [vk_beta_2, vk_gamma_2, vk_delta_2];
        from_affine_mont::<C1>(&mut mont_points_g1);
        from_affine_mont::<C2>(&mut mont_points_g2);
        let vk_alpha_1 = mont_points_g1[0].to_projective();
        let vk_beta_1 = mont_points_g1[1].to_projective();
        let vk_beta_2 = mont_points_g2[0].to_projective();
        let vk_gamma_2 = mont_points_g2[1].to_projective();
        let vk_delta_1 = mont_points_g1[2].to_projective();
        let vk_delta_2 = mont_points_g2[2].to_projective();

        Self {
            n8q,
            q,
            n8r,
            r,
            n_vars,
            n_public,
            domain_size,
            power,
            vk_alpha_1,
            vk_beta_1,
            vk_beta_2,
            vk_gamma_2,
            vk_delta_1,
            vk_delta_2,
        }
    }
}
