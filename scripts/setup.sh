#!/bin/bash

# Get circuit name from command line argument or use default
CIRCUIT_NAME=${1:-circuit}
CIRCUIT_FILE="${CIRCUIT_NAME}.circom"

# === 0. Install dependencies if package.json exists ===
if [ -f "package.json" ]; then
  echo "📦 package.json found. Installing dependencies with npm..."
  npm install
fi

# === 1. Compile the circuit ===
echo "🔧 Compiling the circuit..."
circom --r1cs --wasm --c --sym "$CIRCUIT_FILE"

# === 2. Calculate the witness ===
echo "🧮 Calculating the witness..."
snarkjs wtns calculate "${CIRCUIT_NAME}_js/${CIRCUIT_NAME}.wasm" input.json witness.wtns

# === 3. Get the power of tau from r1cs info ===
echo "📊 Extracting constraint count from R1CS..."
CONSTRAINTS=$(snarkjs r1cs info "${CIRCUIT_NAME}.r1cs" | grep "# of Constraints" | cut -d':' -f3 | xargs)
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
snarkjs groth16 setup "${CIRCUIT_NAME}.r1cs" pot$POWER"_final.ptau" "${CIRCUIT_NAME}_0000.zkey"
mv "${CIRCUIT_NAME}_0000.zkey" "${CIRCUIT_NAME}_final.zkey"

# === 6. Verify final key ===
echo "✅ Verifying final key..."
snarkjs zkey verify "${CIRCUIT_NAME}.r1cs" pot${POWER}_final.ptau "${CIRCUIT_NAME}_final.zkey"

# === 7. Export verification key ===
echo "📤 Exporting verification key..."
snarkjs zkey export verificationkey "${CIRCUIT_NAME}_final.zkey" "${CIRCUIT_NAME}_verification_key.json"

# === 8. Clean up unnecessary files ===
echo "🗑️  Cleaning up..."
rm -f pot${POWER}_final.ptau

echo "🎉 All steps completed successfully!"
