fn main() {
    // Try pkg-config first, fall back to env var
    if pkg_config::probe_library("ipopt").is_ok() {
        return; // pkg-config emitted the link flags already
    }
    if let Ok(dir) = std::env::var("IPOPT_DIR") {
        println!("cargo:rustc-link-search=native={}/lib", dir);
    }
    println!("cargo:rustc-link-lib=dylib=ipopt");
    println!("cargo:rerun-if-env-changed=IPOPT_DIR");
}
