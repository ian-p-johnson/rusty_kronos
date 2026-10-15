use std::env;

fn main() {
    // Link against libtorch (pip torch 2.11) when LIBTORCH is set.
    // torch-sys's own -ltorch_cuda gets dropped by the binary's --as-needed
    // linker pass (nothing references torch_cuda symbols from Rust), which
    // leaves ATen's CUDA dispatch unregistered ("could not run aten::empty
    // ... CUDA backend"). Re-emit it with --no-as-needed at the end of the
    // link line so it survives, and carry the rpath from the spike recipe.
    let torch = match env::var("LIBTORCH") {
        Ok(t) => t,
        Err(_) => return,
    };
    let lib = format!("{torch}/lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-Wl,-ltorch_cuda");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
}
