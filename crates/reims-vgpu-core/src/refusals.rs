//! Ours (ASoftware): whether the guest goes on past what the platform refuses as unsupported,
//! `REIMS_VGPU_REFUSALS=skip`, read once per process. Unset, or any other value, nothing changes:
//! every refusal is decided as without the variable (a death at the send for most of them).
//!
//! With it, the guest loses what was refused and the emulator goes on: a guest whose platform
//! still lacks a feature (visionOS 27.2 on the Vulkan platform) runs, at the cost of what that
//! feature would have drawn or computed. The run is not faithful to Metal, which the first use
//! says on stderr. What it covers is each user's to say: the Vulkan platform's answer layer
//! (`reims_vgpu_vulkan::answer`) and pipeline creators, and the render decoder's draws
//! ([`crate::deserializer::render`]). A guest that meets no refusal runs exactly as without it.

use std::sync::OnceLock;

/// The variable, and the one value that turns the switch on.
pub const ENV_VAR: &str = "REIMS_VGPU_REFUSALS";

/// The value of [`ENV_VAR`] that turns the switch on.
pub const SKIP: &str = "skip";

/// Whether this process skips refusals: [`ENV_VAR`] is [`SKIP`], read once. Never under this
/// crate's own tests.
pub fn skip() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !cfg!(test) && std::env::var_os(ENV_VAR).is_some_and(|v| v == SKIP))
}

/// Says once per process, on stderr, that a refusal at `sel` was skipped under the switch and
/// that the run is not faithful to Metal; later skips are each user's to log.
pub fn first_skip(sel: &str) {
    static SAID: OnceLock<()> = OnceLock::new();
    SAID.get_or_init(|| {
        let line = format!(
            "reims-vgpu: {sel}: refused by the platform and skipped ({ENV_VAR}={SKIP}): the guest \
             goes on without what is refused, and this run is not faithful to Metal\n"
        );
        let _ = std::io::Write::write_all(&mut std::io::stderr(), line.as_bytes());
    });
}
