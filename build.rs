use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    cc::Build::new()
        .file("adpcm-xq/adpcm-xq.c")
        .file("adpcm-xq/adpcm-lib.c")
        .compile("adpcm-xq");
    cc::Build::new()
        .cpp(true)
        .include(PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?).join("r8brain-free-src"))
        .file("r8brain-free-src/r8bbase.cpp")
        .file("src/resampler-extern.cpp")
        .compile("resampler");
    // cc::Build::new()
    //     .include(PathBuf::from(std::env::var("CARGO_MANIFEST_DIR")?).join("adpcm-xq"))
    //     .file("src/lib.c")
    //     .compile("dse-dsp-sys");
    println!("cargo:rerun-if-changed=src/resampler-extern.cpp");
    println!("cargo:rerun-if-changed=src/lib.c");
    Ok(())
}