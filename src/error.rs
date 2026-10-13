#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid Needle archive: {0}")]
    Archive(String),
    #[error("invalid tokenizer: {0}")]
    Tokenizer(String),
    #[error("invalid request: {0}")]
    Request(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Power(#[from] a3s_power::error::PowerError),
    #[error(transparent)]
    Tensor(#[from] candle_core::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
