# Debug Feature Flag

Controls all debug output (`println!` statements) in the Rust code.

## Usage

**Enable debug output:**
```bash
cargo build --features debug
```

**Disable debug output (default):**
```bash
cargo build
```

## What Gets Controlled

- `src/lib.rs` - Main library functions and FFI interface
- `src/groth16/prove.rs` - Groth16 proof generation functions  
- `src/file_wrapper.rs` - File I/O operations

## Debug Output Examples

When enabled, you'll see output like:
```
[RUST] parallel_prove called with num_proofs: 4
[GROTH16] parallel_prove called with 4 witness paths, max_batch_size: 10
[GROTH16] Processing 4 total witnesses in 1 batches of max size 10
commitments_batched: batch_size=4, n_public=1
Building R1CS
commit_g1_batched: section_idx=9, d_scalars len=4000, points len=2, batch_size=4
[GROTH16] All batches completed with 4 total results
```

## Implementation

```rust
#[cfg(feature = "debug")]
macro_rules! debug_println {
    ($($arg:tt)*) => { println!($($arg)*); };
}

#[cfg(not(feature = "debug"))]
macro_rules! debug_println {
    ($($arg:tt)*) => {};
}
```

## Performance

- **Debug enabled**: Slight overhead from string formatting and I/O
- **Debug disabled**: Zero performance impact - statements removed at compile time

## Build Systems

**iOS/macOS:**
```bash
cargo build --features debug --target aarch64-apple-ios
```

**Android (Gradle):**
```gradle
android {
    buildTypes {
        debug { environment "RUSTFLAGS", "--cfg feature=\"debug\"" }
        release { environment "RUSTFLAGS", "" }
    }
}
``` 