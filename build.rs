use std::env;

fn main() {
    let target = env::var("TARGET").unwrap();
    
    // iOS-specific configuration
    if target.contains("apple-ios") {
        // Add iOS framework linking
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Security");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
        
        // Force static linking for iOS
        println!("cargo:rustc-link-arg=-static");
        
        // Tell cargo to rerun if environment changes
        println!("cargo:rerun-if-env-changed=TARGET");
        println!("cargo:rerun-if-env-changed=CONFIGURATION");
    }
    
    // Handle different build configurations
    if let Ok(config) = env::var("CONFIGURATION") {
        if config == "Release" {
            println!("cargo:rustc-env=OPTIMIZATION=release");
        } else {
            println!("cargo:rustc-env=OPTIMIZATION=debug");
        }
    }
} 