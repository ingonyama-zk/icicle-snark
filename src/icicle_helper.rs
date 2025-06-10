use crate::F;
use icicle_core::{
    curve::{Affine, Curve, Projective},
    msm::{msm, MSMConfig, MSM},
    ntt::{ntt_inplace, NTTConfig, NTTDir, NTT},
    traits::FieldImpl,
};
use icicle_runtime::{
    memory::HostSlice,
    stream::IcicleStream,
};

pub fn ntt_helper(vec: &mut HostSlice<F>, inverse: bool, stream: &IcicleStream)
where
    <F as FieldImpl>::Config: NTT<F, F>,
{
    let dir = if inverse {
        NTTDir::kInverse
    } else {
        NTTDir::kForward
    };

    let mut cfg1 = NTTConfig::<F>::default();
    cfg1.is_async = true;
    cfg1.batch_size = 3;
    cfg1.stream_handle = stream.into();

    ntt_inplace(vec, dir, &cfg1).unwrap();
}

pub fn msm_helper<C: Curve + MSM<C>>(
    scalars: &[C::ScalarField],
    points: &[Affine<C>],
    stream: &IcicleStream,
) -> Projective<C> {
    let mut msm_result = vec![Projective::<C>::zero()];
    let msm_result_host = HostSlice::from_mut_slice(&mut msm_result);
    let mut msm_config = MSMConfig::default();
    msm_config.stream_handle = stream.into();
    msm_config.is_async = true;

    let h_scalars = HostSlice::from_slice(scalars);
    let h_points = HostSlice::from_slice(points);

    msm(&h_scalars[..], &h_points[..], &msm_config, &mut msm_result_host[..]).unwrap();

    msm_result[0]
}
