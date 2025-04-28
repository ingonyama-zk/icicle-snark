## reference: run on Mac

Build the project:
```bash
cargo build --release
```

Run with different circuit sizes:

100k circuit:
```bash
prove --witness ./benchmark/100k/witness.wtns --zkey ./benchmark/100k/circuit_final.zkey --proof ./benchmark/100k/proof.json --public ./benchmark/100k/public.json --device CPU
```

200k circuit:
```bash
prove --witness ./benchmark/200k/witness.wtns --zkey ./benchmark/200k/circuit_final.zkey --proof ./benchmark/200k/proof.json --public ./benchmark/200k/public.json --device CPU
```

400k circuit:
```bash
prove --witness ./benchmark/400k/witness.wtns --zkey ./benchmark/400k/circuit_final.zkey --proof ./benchmark/400k/proof.json --public ./benchmark/400k/public.json --device CPU
```

800k circuit:
```bash
prove --witness ./benchmark/800k/witness.wtns --zkey ./benchmark/800k/circuit_final.zkey --proof ./benchmark/800k/proof.json --public ./benchmark/800k/public.json --device CPU
```

1600k circuit:
```bash
prove --witness ./benchmark/1600k/witness.wtns --zkey ./benchmark/1600k/circuit_final.zkey --proof ./benchmark/1600k/proof.json --public ./benchmark/1600k/public.json --device CPU
```

## run on iOS simulator

### Prerequisites

- Xcode 15.0 or later
- iOS 17.0 or later
- Rust toolchain (stable or nightly)
- Cargo
- iOS Simulator
- `.cargo/config.toml` with iOS target configuration.
  
### Using ios.sh Script

The `ios.sh` script provides a convenient way to build, copy files, and run the tool in the iOS simulator. It has four main commands:

1. **Build the tool**:
   ```bash
   ./ios.sh build
   ```
   This will:
   - Clean previous build artifacts
   - Build the project for iOS simulator
   - Copy the binary and dependencies to the simulator

2. **Copy input files**:
   ```bash
   ./ios.sh input <circuit_size>
   ```
   Where `<circuit_size>` is the size of the circuit (e.g., `100k`, `1600k`).
   This will:
   - Copy witness and circuit files from `benchmark/<circuit_size>/` to the simulator
   - Set appropriate file permissions
   - Verify the files are copied correctly

3. **Run the tool**:
   ```bash
   ./ios.sh run
   ```
   This will:
   - Verify input files exist
   - Run the tool in the simulator with the correct library paths
   - Generate a proof using the input files:

   ```txt
   
   ```

4. **Open a shell in simulator**:
   ```bash
   ./ios.sh shell
   ```
   This opens an interactive shell in the simulator where you can:
   - Navigate directories
   - Check file permissions
   - Run commands manually
   - Debug file paths

### Example Workflow

1. Build the tool:
   ```bash
   ./ios.sh build
   ```

2. Copy input files for a specific circuit size:
   ```bash
   ./ios.sh input 100k
   ```

3. Run the tool:
   ```bash
   ./ios.sh run
    prove --witness ./input/witness.wtns --zkey ./input/circuit_final.zkey --proof ./input/proof.json --public ./input/public.json --device CPU
   ```

4. If you need to debug, open a shell:
   ```bash
   ./ios.sh shell
   ```

### Notes

- The script automatically handles simulator device selection (uses iPhone 16 Pro)
- All paths are managed relative to the script's location
- Input files are copied from `benchmark/<circuit_size>/` directory
- The script verifies dependencies and file existence before operations
- Debug mode can be enabled by uncommenting `set -x` at the top of the script



## run on iOS simulator

I use a separate project ingonyama-zk/groth16-ios to run icicle-snark on iphone
