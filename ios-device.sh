#!/bin/zsh

# Exit on error
set -e

# Get the directory where the script is located
SCRIPT_DIR=${0:A:h}

# Default circuit size
CIRCUIT=${2:-1600k}

# Check ios-deploy is installed
if ! command -v ios-deploy &> /dev/null; then
    print "Error: ios-deploy is not installed. Install it with: brew install ios-deploy"
    exit 1
fi

# Function to build and sign
build_target() {
    print 'Cleaning previous build artifacts...'
    cd $SCRIPT_DIR
    cargo clean

    print 'Building icicle-snark and its dependencies for iOS device...'
    cd $SCRIPT_DIR
    RUSTFLAGS='-C debuginfo=2' cargo build --target aarch64-apple-ios

    print 'Creating app bundle...'
    BUNDLE_DIR="$SCRIPT_DIR/target/ios-bundle"
    mkdir -p "$BUNDLE_DIR/deps/icicle/lib"

    # Copy binary and libraries
    cp "$SCRIPT_DIR/target/aarch64-apple-ios/debug/icicle-snark" "$BUNDLE_DIR/"
    cp "$SCRIPT_DIR/target/aarch64-apple-ios/debug/deps/icicle/lib/"*.dylib "$BUNDLE_DIR/deps/icicle/lib/"

    # Create entitlements file
    cat > "$BUNDLE_DIR/entitlements.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.get-task-allow</key>
    <true/>
</dict>
</plist>
EOF

    print 'Code signing...'
    # Sign all dylibs
    for lib in "$BUNDLE_DIR/deps/icicle/lib/"*.dylib; do
        codesign --force --sign - --entitlements "$BUNDLE_DIR/entitlements.plist" "$lib"
    done
    # Sign the main binary
    codesign --force --sign - --entitlements "$BUNDLE_DIR/entitlements.plist" "$BUNDLE_DIR/icicle-snark"
}

# Function to copy input files
copy_inputs() {
    print 'Copying input files...'
    BUNDLE_DIR="$SCRIPT_DIR/target/ios-bundle"
    mkdir -p "$BUNDLE_DIR/input"
    
    INPUT_SOURCE_DIR=$SCRIPT_DIR/benchmark/$CIRCUIT
    if [[ -d $INPUT_SOURCE_DIR ]]; then
        print 'Source files:'
        ls -la $INPUT_SOURCE_DIR/witness.wtns $INPUT_SOURCE_DIR/circuit_final.zkey
        cp -vp $INPUT_SOURCE_DIR/witness.wtns "$BUNDLE_DIR/input/"
        cp -vp $INPUT_SOURCE_DIR/circuit_final.zkey "$BUNDLE_DIR/input/"
        chmod 755 "$BUNDLE_DIR/input/circuit_final.zkey" "$BUNDLE_DIR/input/witness.wtns"
    else
        print "Error: Input directory $INPUT_SOURCE_DIR does not exist"
        exit 1
    fi
}

# Function to run on device
run_tool() {
    BUNDLE_DIR="$SCRIPT_DIR/target/ios-bundle"
    
    # Verify input files exist
    if [[ ! -f "$BUNDLE_DIR/input/witness.wtns" ]] || [[ ! -f "$BUNDLE_DIR/input/circuit_final.zkey" ]]; then
        print "Error: Input files not found. Please run './ios-device.sh input $CIRCUIT' first"
        exit 1
    fi

    print 'Installing and running on device...'
    ios-deploy --bundle "$BUNDLE_DIR" --noninteractive --debug \
        --env DYLD_LIBRARY_PATH="./deps/icicle/lib" \
        --args "prove" "./input/circuit_final.zkey" "./input/witness.wtns" "./input/proof.json" "./input/public.json" "--device" "CPU"
}

# Main command handling
case $1 in
    "build")
        build_target
        ;;
    "input")
        copy_inputs
        ;;
    "run")
        run_tool
        ;;
    *)
        print "Usage: $0 {build|input|run} [circuit_size]"
        print "  build: Build and sign the tool for iOS device"
        print "  input: Copy input files (requires circuit_size, e.g. 100k, 1600k)"
        print "  run:   Run the tool on connected iOS device"
        print ""
        print "Prerequisites:"
        print "  - ios-deploy (install with: brew install ios-deploy)"
        print "  - Connected iOS device in developer mode"
        print "  - Valid development certificate"
        print ""
        print "Examples:"
        print "  $0 build"
        print "  $0 input 100k"
        print "  $0 run"
        exit 1
        ;;
esac

print "Done!" 