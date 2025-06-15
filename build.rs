fn main() {
  // Add iOS-specific configuration
  let target = std::env::var("TARGET").unwrap();
  if target.contains("apple-ios") {
      // Add iOS framework linking
      println!("cargo:rustc-link-lib=framework=Foundation");
      println!("cargo:rustc-link-lib=framework=Security");
      
      // Force static linking for iOS
      println!("cargo:rustc-link-arg=-static");
  }
} 