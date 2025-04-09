# Cross-compile Icicle app to Android, deploy, and run 

## Cross-compile Icicle/Rust CLI app to Android using Ubuntu container


1. build a container `./build-android-toolchain.sh`
2. Cross-compile for ARM64 (aarch64)

`docker run --platform linux/amd64 -v $(pwd):/app icicle-android-cross cargo build --target aarch64-linux-android --release`

for debug, run the container interactively:

```sh
docker run --platform linux/amd64 -it \
  -v $(pwd):/app \
  -v /Users/stas/Projects/ingonyama-zk/icicle:/opt/icicle \
  icicle-android-cross /bin/bash

export ANDROID_NDK=/opt/android-ndk-r28
export TOOLCHAIN=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64
export PATH=$TOOLCHAIN/bin:$PATH
export CC=aarch64-linux-android21-clang
export CXX=aarch64-linux-android21-clang++
export AR=aarch64-linux-android-ar
export LD=aarch64-linux-android-ld
export RUSTFLAGS="-C linker=$CC -C link-arg=--sysroot=$TOOLCHAIN/sysroot -C link-arg=-L$TOOLCHAIN/sysroot/usr/lib/aarch64-linux-android/21"

cd /app
cargo build --target aarch64-linux-android --release
```

3. Push to Android and run
adb push target/aarch64-linux-android/release/rust_android_cli /data/local/tmp/
adb shell /data/local/tmp/rust_android_cli