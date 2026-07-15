//! build.rs — locate and link a system IPOPT for the raw-FFI bridge.
//!
//! This pairs with the hand-written `mod ffi` in `ipopt_ffi.rs`: it does no
//! code generation, it only tells the linker where libipopt is. Install IPOPT
//! from a package manager so there's no source build (and no Metis/MUMPS pain):
//!
//!   macOS:          brew install ipopt
//!   Debian/Ubuntu:  sudo apt install coinor-libipopt-dev
//!
//! Cargo.toml needs:
//!   [build-dependencies]
//!   pkg-config = "0.3"

use std::env;

fn main() {
    // Rebuild if the override changes.
    println!("cargo:rerun-if-env-changed=IPOPT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");

    // Preferred path: pkg-config reads the ipopt.pc that brew/apt install and
    // emits the `cargo:rustc-link-search` / `-link-lib` lines automatically,
    // pulling in transitive deps (BLAS/LAPACK, gfortran, etc.) as declared.
    if pkg_config::Config::new()
        .atleast_version("3.14")
        .probe("ipopt")
        .is_ok()
    {
        return;
    }

    // Fallback: point us straight at a lib directory, bypassing pkg-config:
    //   IPOPT_LIB_DIR=$(brew --prefix ipopt)/lib cargo build
    if let Ok(dir) = env::var("IPOPT_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=dylib=ipopt");
        return;
    }

    panic!(
        "\n\
        Could not find libipopt via pkg-config.\n\
        \n\
        Install it:\n\
          macOS:          brew install ipopt\n\
          Debian/Ubuntu:  sudo apt install coinor-libipopt-dev\n\
        \n\
        If it's installed via Homebrew, expose its pkg-config file:\n\
          export PKG_CONFIG_PATH=\"$(brew --prefix ipopt)/lib/pkgconfig:$PKG_CONFIG_PATH\"\n\
        \n\
        Or bypass pkg-config entirely:\n\
          IPOPT_LIB_DIR=$(brew --prefix ipopt)/lib cargo build\n"
    );
}
