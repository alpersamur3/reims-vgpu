//! Sets `cfg(apv_guest_4k)`, a guest with 4 KiB pages, for an x86-64 build without the
//! `guest-arm64` feature (`guest::records::APV_PAGE_SHIFT`). Every other build serves an arm64
//! guest with 16 KiB pages: an aarch64 build, and an x86-64 build with `guest-arm64` (an arm64
//! guest under TCG on an x86-64 host).

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(apv_guest_4k)");
    let x86_64 = std::env::var("CARGO_CFG_TARGET_ARCH").is_ok_and(|arch| arch == "x86_64");
    let guest_arm64 = std::env::var_os("CARGO_FEATURE_GUEST_ARM64").is_some();
    if x86_64 && !guest_arm64 {
        println!("cargo::rustc-cfg=apv_guest_4k");
    }
}
