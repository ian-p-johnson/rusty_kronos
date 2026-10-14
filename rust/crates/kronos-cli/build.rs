use std::env;

fn main() {
    // Optional cuDNN link path. When candle is built with its `cudnn`
    // feature the linker needs -lcudnn; the NVIDIA pip wheel ships the
    // soname-suffixed copies only, so point KRONOS_CUDNN_LIB at a directory
    // with unversioned symlinks (see PORTING note or parity README).
    if let Ok(path) = env::var("KRONOS_CUDNN_LIB") {
        println!("cargo:rustc-link-search=native={path}");
    }
    // Trigger rebuild when it changes.
    println!("cargo:rerun-if-env-changed=KRONOS_CUDNN_LIB");
    println!("cargo:rerun-if-changed=build.rs");
}
