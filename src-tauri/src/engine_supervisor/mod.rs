//! Process isolation for transcribe.cpp.
//!
//! All transcribe.cpp native code (backend init, device enumeration, model
//! load, inference, streaming) runs in a child process: the Handy executable
//! re-launched with [`WORKER_FLAG`]. A GPU driver fault, `GGML_ASSERT` abort,
//! or hang then kills only the worker.
//!
//! [`EngineSupervisor`] is the single owner of that worker. It loads and
//! unloads models, runs and streams transcriptions, cancels them, and
//! recovers from crashes and hangs (trying another healthy GPU before CPU
//! after a device failure), so the rest of the app never sees processes, restarts
//! or fallback. At most one worker holds a model at a time, and it lives
//! exactly as long as that model, so unloading returns all of its CPU and GPU
//! memory to the OS. Listing compute devices may briefly run a second,
//! model-less worker.
//!
//! A CPU-only worker ([`CPU_ONLY_FLAG`]) restricts transcribe.cpp to its CPU
//! backends before any are registered, so no GPU driver code runs in it at
//! all: a driver that crashes or hangs while initializing can't take the CPU
//! fallback down with it.
//!
//! A failed GPU is excluded while recovery tries another healthy GPU before
//! strict CPU, without changing the saved preference. Discovery can select a
//! new GPU or a failed one observed disappearing and then returning. A failed
//! device that stays present is never retried automatically; restart Handy or
//! explicitly change accelerators to try it again. Checks run between dictations
//! at most once a minute, so idle Handy does no hardware polling and active
//! streams keep their decoder. Recovery cannot repair the underlying driver.
//!
//! Explicit CPU preferences and host GPU restrictions prevent GPU upgrades.
//! CLI device indices are strict diagnostic selections without fallback.
//!
//! - `protocol`: the framed request/response protocol between the two.
//! - `supervisor`: the parent side ([`EngineSupervisor`]).
//! - `worker`: the child process.

mod protocol;
mod supervisor;
mod worker;

pub use protocol::{DeviceInfo, DeviceSelector, LoadedInfo};
pub use supervisor::{
    CancelMark, EngineError, EngineSupervisor, Finalized, LoadSpec, StreamHandle, StreamProgress,
    Unloading,
};

/// Hidden first argument that turns the executable into a worker.
pub const WORKER_FLAG: &str = "--transcribe-worker";
/// Worker argument: register only the CPU backends.
const CPU_ONLY_FLAG: &str = "--cpu-only";
const LOG_LEVEL_ENV: &str = "HANDY_TRANSCRIBE_WORKER_LOG";

/// Whether this process was launched as a transcription worker. Checked in
/// `main` before CLI parsing, Tauri, or single-instance handling.
pub fn is_worker_invocation() -> bool {
    std::env::args_os()
        .nth(1)
        .is_some_and(|arg| arg == WORKER_FLAG)
}

/// Run the worker until the parent closes its stdin. Returns the exit code.
pub fn run_worker() -> i32 {
    worker::run()
}
