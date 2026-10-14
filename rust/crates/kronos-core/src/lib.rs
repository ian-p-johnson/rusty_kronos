//! Rust port of Kronos — a foundation model for the language of financial
//! candlesticks (arXiv:2508.02739, AAAI 2026).
//!
//! Scope of this port, and why it is tractable: the entire GPU workload of the
//! upstream project is `model/kronos.py` + `model/module.py`, ~1,230 lines of
//! standard PyTorch with no custom CUDA kernels and no custom autograd on the
//! inference path. Everything else in the upstream repo is pandas/matplotlib
//! glue that never touches a device. See `PORTING_FROM_PYTHON.md`.
//!
//! Numerical parity is the governing constraint. Upstream `model/` is frozen as
//! the oracle (`parity/check_frozen.sh`), and every stage here is checked
//! against per-stage activations captured from it.

pub mod ar;
pub mod block;
pub mod config;
pub mod data;
pub mod embed;
pub mod flash;
pub mod npy;
pub mod ops;
pub mod parity;
pub mod predictor;
pub mod sampling;
pub mod tokenizer;
pub mod weights;

pub use config::{BlockGeometry, ConfigError, PredictorConfig, TokenizerConfig};
pub use data::{load_csv, DataError, Kline};
pub use weights::{
    expected_predictor_keys, expected_tokenizer_keys, load_predictor_weights,
    load_tokenizer_weights, LoadError, Weights,
};
