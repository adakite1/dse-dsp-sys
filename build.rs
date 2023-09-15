use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    cc::Build::new()
        .cpp(true)
        .include(PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?).join("r8brain-free-src"))
        .file("r8brain-free-src/r8bbase.cpp")
        .file("src/resampler-extern.cpp")
        .compile("resampler");
    println!("cargo:rerun-if-changed=src/resampler-extern.cpp");
    println!("cargo:rerun-if-changed=src/lib.c");
    Ok(())
}