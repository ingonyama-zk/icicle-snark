#!/bin/zsh

# Exit on error
set -e

# Enable debug mode
# set -x

# Get the directory where the script is located
SCRIPT_DIR=${0:A:h}

# Default circuit size
CIRCUIT=${2:-1600k}

# Function to check dependencies
check_dependencies() {
    local target_dir=$1
    local missing_deps=0
    
    for lib in libicicle_field_bn254.dylib libicicle_curve_bn254.dylib libicicle_hash.dylib libicicle_device.dylib; do
        if [[ ! -f $target_dir/deps/icicle/lib/$lib ]]; then
            print "Missing dependency: $lib"
            missing_deps=1
        fi
    done
    
    return $missing_deps
}

# Function to get simulator device ID
get_simulator_id() {
    local device_id=$(xcrun simctl list devices | grep 'iPhone 16 Pro' | grep -E -o -i '([0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12})' | head -n 1)
    if [[ -z $device_id ]]; then
        print "Error: Could not find iPhone 16 Pro simulator"
        exit 1
    fi
    print $device_id
}

# Function to set up simulator paths
setup_simulator_paths() {
    local device_id=$1
    local host_root=$HOME/Library/Developer/CoreSimulator/Devices/$device_id
    local simulator_root=$host_root/data
    local simulator_target=$simulator_root/target/aarch64-apple-ios-sim/debug
    local simulator_input=$simulator_root/input
    
    print "SIMULATOR_ROOT=$simulator_root"
    print "SIMULATOR_TARGET=$simulator_target"
    print "SIMULATOR_INPUT=$simulator_input"
}

# Function to build and copy target
build_target() {
    print 'Cleaning previous build artifacts...'
    cd $SCRIPT_DIR
    cargo clean

    print 'Building icicle-snark and its dependencies in debug mode...'
    cd $SCRIPT_DIR
    RUSTFLAGS='-C debuginfo=2' cargo build --target aarch64-apple-ios-sim -v

    print 'Booting simulator...'
    xcrun simctl boot 'iPhone 16 Pro' || true

    local device_id=$(get_simulator_id)
    local -A paths
    while IFS='=' read -r key value; do
        paths[$key]=$value
    done < <(setup_simulator_paths $device_id)

    print 'Creating target directory structure in simulator...'
    mkdir -p $paths[SIMULATOR_TARGET]/deps/icicle/lib
    mkdir -p $paths[SIMULATOR_INPUT]

    print 'Copying target files to simulator...'
    cp -Rv $SCRIPT_DIR/target/aarch64-apple-ios-sim/debug/icicle-snark $paths[SIMULATOR_TARGET]/
    cp -Rv $SCRIPT_DIR/target/aarch64-apple-ios-sim/debug/deps/icicle/lib/libicicle_field_bn254.dylib $paths[SIMULATOR_TARGET]/deps/icicle/lib/
    cp -Rv $SCRIPT_DIR/target/aarch64-apple-ios-sim/debug/deps/icicle/lib/libicicle_curve_bn254.dylib $paths[SIMULATOR_TARGET]/deps/icicle/lib/
    cp -Rv $SCRIPT_DIR/target/aarch64-apple-ios-sim/debug/deps/icicle/lib/libicicle_hash.dylib $paths[SIMULATOR_TARGET]/deps/icicle/lib/
    cp -Rv $SCRIPT_DIR/target/aarch64-apple-ios-sim/debug/deps/icicle/lib/libicicle_device.dylib $paths[SIMULATOR_TARGET]/deps/icicle/lib/
}

# Function to copy input files
copy_inputs() {
    local device_id=$(get_simulator_id)
    local -A paths
    while IFS='=' read -r key value; do
        paths[$key]=$value
    done < <(setup_simulator_paths $device_id)

    print 'Copying input files...'
    INPUT_SOURCE_DIR=$SCRIPT_DIR/benchmark/$CIRCUIT
    if [[ -d $INPUT_SOURCE_DIR ]]; then
        print 'Source files:'
        ls -la $INPUT_SOURCE_DIR/witness.wtns $INPUT_SOURCE_DIR/circuit_final.zkey
        cp -vp $INPUT_SOURCE_DIR/witness.wtns $paths[SIMULATOR_INPUT]/
        cp -vp $INPUT_SOURCE_DIR/circuit_final.zkey $paths[SIMULATOR_INPUT]/
        chmod 755 $paths[SIMULATOR_INPUT]/circuit_final.zkey $paths[SIMULATOR_INPUT]/witness.wtns
    else
        print "Error: Input directory $INPUT_SOURCE_DIR does not exist"
        exit 1
    fi

    print 'Verifying files in simulator...'
    xcrun simctl spawn $device_id /bin/zsh -c "[[ -f $paths[SIMULATOR_INPUT]/witness.wtns ]] && print 'witness.wtns exists' || print 'witness.wtns missing'"
    xcrun simctl spawn $device_id /bin/zsh -c "[[ -f $paths[SIMULATOR_INPUT]/circuit_final.zkey ]] && print 'circuit_final.zkey exists' || print 'circuit_final.zkey missing'"
}

# Function to run the tool
run_tool() {
    local device_id=$(get_simulator_id)
    local -A paths
    while IFS='=' read -r key value; do
        paths[$key]=$value
    done < <(setup_simulator_paths $device_id)

    # Verify input files exist
    if [[ ! -f $paths[SIMULATOR_INPUT]/witness.wtns ]] || [[ ! -f $paths[SIMULATOR_INPUT]/circuit_final.zkey ]]; then
        print "Error: Input files not found. Please run './ios.sh input $CIRCUIT' first"
        exit 1
    fi

    print 'Running command in simulator...'
    RUNTIME_PATH=$(xcrun simctl getenv $device_id DYLD_ROOT_PATH)
    xcrun simctl spawn $device_id /bin/zsh -c "DYLD_ROOT_PATH='$RUNTIME_PATH' DYLD_LIBRARY_PATH=$paths[SIMULATOR_TARGET]/deps/icicle/lib:/usr/lib:/usr/lib/system RUST_BACKTRACE=1 $paths[SIMULATOR_TARGET]/icicle-snark prove $paths[SIMULATOR_INPUT]/circuit_final.zkey $paths[SIMULATOR_INPUT]/witness.wtns"
}

# Function to open shell in simulator
open_shell() {
    local device_id=$(get_simulator_id)
    local -A paths
    while IFS='=' read -r key value; do
        paths[$key]=$value
    done < <(setup_simulator_paths $device_id)

    print 'Opening shell in simulator...'
    print "Target directory: $paths[SIMULATOR_TARGET]"
    print "Input directory: $paths[SIMULATOR_INPUT]"
    print "Type 'exit' to leave the shell"
    
    # Spawn shell with minimal configuration
    xcrun simctl spawn $device_id /bin/sh -c "PS1='simulator> ' exec /bin/sh"
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
    "shell")
        open_shell
        ;;
    *)
        print "Usage: $0 {build|input|run|shell} [circuit_size]"
        print "  build: Build and copy the target tool to simulator"
        print "  input: Copy input files to simulator (requires circuit_size, e.g. 100k, 1600k)"
        print "  run:   Run the tool in simulator"
        print "  shell: Open a shell in the simulator"
        print ""
        print "Examples:"
        print "  $0 build"
        print "  $0 input 100k"
        print "  $0 run"
        print "  $0 shell"
        exit 1
        ;;
esac

print "Done!" 