# Cross-compile Icicle app to Android, deploy, and run 

## Cross-compile Icicle/Rust CLI app to Android using Ubuntu container


1. build a container `./build-android-toolchain.sh`
2. Cross-compile for

- For ARM64 (aarch64)
docker run --platform linux/amd64 -v $(pwd):/app rust-android-cross cargo build --target aarch64-linux-android --release

- For ARMv7
docker run -v $(pwd):/app rust-android-cross cargo build --target armv7-linux-androideabi --release

3. Push to Android and run
adb push target/aarch64-linux-android/release/rust_android_cli /data/local/tmp/
adb shell /data/local/tmp/rust_android_cli