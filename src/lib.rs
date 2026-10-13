//! Akis owns model semantics; Power owns the embedded execution boundary.
mod archive;
mod error;
mod math;
mod model;
mod structured;
mod tokenizer;

pub use a3s_power::inference::{DevicePreference, EmbeddedRuntime, InferenceLimits};
pub use archive::{Archive, Geometry};
pub use error::{Error, Result};
pub use model::{Generation, Model};
pub use structured::{Completion, Tool, ToolCall};
pub use tokenizer::Tokenizer;
pub use tokio_util::sync::CancellationToken;

#[cfg(test)]
mod test_support;
