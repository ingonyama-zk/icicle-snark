#!/bin/bash

# === 0. Install dependencies if package.json exists ===
if [ -f "package.json" ]; then
  echo "📦 package.json found. Installing dependencies with npm..."
  npm install
fi

# === 1. Compile the circuit ===
echo "🔧 Compiling the circuit..."
circom --r1cs --wasm --c --sym circuit.circom

# === 2. Calculate the witness ===
echo "🧮 Calculating the witness..."
snarkjs wtns calculate circuit_js/circuit.wasm input.json witness.wtns

# === 3. Get the power of tau from r1cs info ===
echo "📊 Extracting constraint count from R1CS..."
CONSTRAINTS=$(snarkjs r1cs info circuit.r1cs | grep "# of Constraints" | cut -d':' -f3 | xargs)
POWER=$(python3 -c "import math; print(max(8, math.ceil(math.log2(${CONSTRAINTS} + 1))))")
if [ "$POWER" -lt 10 ]; then
    POWER=$(printf "%02d" $POWER)
fi
echo "⚡ Computed power of tau: $POWER"

# === 4. Powers of Tau ceremony ===
echo "🔮 Downloading powers of tau ceremony file..."
wget https://storage.googleapis.com/zkevm/ptau/powersOfTau28_hez_final_$POWER.ptau
mv powersOfTau28_hez_final_$POWER.ptau pot$POWER"_final.ptau"

# === 5. Setup ===
echo "🛠️ Running setup..."
snarkjs groth16 setup circuit.r1cs pot$POWER"_final.ptau" circuit_0000.zkey
mv circuit_0000.zkey circuit_final.zkey

# === 6. Verify final key ===
echo "✅ Verifying final key..."
snarkjs zkey verify circuit.r1cs pot${POWER}_final.ptau circuit_final.zkey

# === 7. Export verification key ===
echo "📤 Exporting verification key..."
snarkjs zkey export verificationkey circuit_final.zkey verification_key.json

# === 8. Clean up unnecessary files ===
echo "🗑️  Cleaning up..."
rm -f pot${POWER}_final.ptau

echo "🎉 All steps completed successfully!"
