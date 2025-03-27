# Use Ubuntu LTS as base
FROM ubuntu:22.04

# Avoid interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    unzip \
    git \
    adb \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust via rustup
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"



# Download and extract Android NDK (version 25b)

ARG NDK_VERSION="25b"
WORKDIR /android

RUN wget -q https://dl.google.com/android/repository/android-ndk-r${NDK_VERSION}-linux.zip \
    && unzip -q android-ndk-r${NDK_VERSION}-linux.zip \
    && rm android-ndk-r${NDK_VERSION}-linux.zip

ENV ANDROID_NDK_HOME="/android/android-ndk-r${NDK_VERSION}"

# Add Android targets (ARM64, ARMv7, x86_64)
RUN rustup target add \
    aarch64-linux-android \
    armv7-linux-androideabi \
    x86_64-linux-android

# Configure Cargo for cross-compilation
RUN mkdir -p /root/.cargo
COPY <<EOF /root/.cargo/config.toml
[target.aarch64-linux-android]
linker = "${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang"
ar = "${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android-ar"

[target.armv7-linux-androideabi]
linker = "${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi21-clang"
ar = "${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"

[target.x86_64-linux-android]
linker = "${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android21-clang"
ar = "${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar"
EOF

# Set working directory for builds
WORKDIR /app
VOLUME /app

# Default command (show Rust/NDK versions)
CMD ["sh", "-c", "rustc --version && cargo --version && ${ANDROID_NDK_HOME}/ndk-build --version"]