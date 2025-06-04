# Define available curve libraries with an index and their supported features
# Format: index:curve:features
set(ICICLE_CURVES
  1:bn254:NTT,MSM,G2,PAIRING
  2:bls12_381:NTT,MSM,G2,PAIRING
  3:bls12_377:NTT,MSM,G2,PAIRING
  4:bw6_761:NTT,MSM,G2,PAIRING
)
