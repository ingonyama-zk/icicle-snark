#!/bin/zsh

# Exit on error
set -e

# Get the directory where the script is located
SCRIPT_DIR=${0:A:h}

# Default circuit size
CIRCUIT=${1:-1600k}

print 'Cleaning previous build artifacts...'
cd $SCRIPT_DIR
cargo clean

print 'Building icicle-snark and its dependencies for iOS Simulator...'
cd $SCRIPT_DIR
RUSTFLAGS='-C debuginfo=2' cargo build --target aarch64-apple-ios-sim

print 'Setting up simulator environment...'
# Get the simulator device ID for iPhone 16 Pro
DEVICE_ID=$(xcrun simctl list devices | grep "iPhone 16 Pro" | grep -E -o -i "([0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12})")

# Boot the simulator if not already running
xcrun simctl boot "$DEVICE_ID" 2>/dev/null || true

# Set up paths in simulator
SIMULATOR_ROOT="/Users/$USER/Library/Developer/CoreSimulator/Devices/$DEVICE_ID/data"
SIMULATOR_TARGET="$SIMULATOR_ROOT/target/aarch64-apple-ios-sim/debug"
SIMULATOR_INPUT="$SIMULATOR_ROOT/input"

# Create necessary directories
mkdir -p "$SIMULATOR_TARGET/deps"
mkdir -p "$SIMULATOR_INPUT"

print 'Copying target files to simulator...'
# Copy the binary and libraries
cp -v "$SCRIPT_DIR/target/aarch64-apple-ios-sim/debug/icicle-snark" "$SIMULATOR_TARGET/"
cp -v "$SCRIPT_DIR/target/aarch64-apple-ios-sim/debug/deps/"*.dylib "$SIMULATOR_TARGET/deps/"

print 'Copying input files...'
# Copy input files from benchmark directory
INPUT_SOURCE_DIR="$SCRIPT_DIR/benchmark/$CIRCUIT"
if [[ -d $INPUT_SOURCE_DIR ]]; then
    cp -v "$INPUT_SOURCE_DIR/witness.wtns" "$SIMULATOR_INPUT/"
    cp -v "$INPUT_SOURCE_DIR/circuit_final.zkey" "$SIMULATOR_INPUT/"
    chmod 755 "$SIMULATOR_INPUT/circuit_final.zkey" "$SIMULATOR_INPUT/witness.wtns"
else
    print "Error: Input directory $INPUT_SOURCE_DIR does not exist"
    exit 1
fi

print 'Running benchmark in simulator...'
# Run the command in simulator with proper library paths
xcrun simctl spawn "$DEVICE_ID" /bin/sh -c "cd $SIMULATOR_TARGET && \
    DYLD_LIBRARY_PATH=./deps \
    ./icicle-snark prove ./input/circuit_final.zkey ./input/witness.wtns ./input/proof.json ./input/public.json --device CPU"

print "Done!" 