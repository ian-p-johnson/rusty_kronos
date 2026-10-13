//! Configuration schema mirroring the released Hugging Face `config.json` files.
//!
//! Every field is required and validated at construction. The port is driven
//! entirely by these files — nothing about Kronos-small / -base / -mini is
//! hardcoded — because the three released predictors genuinely differ and the
//! two released *tokenizers* differ in a way the paper does not mention:
//!
//! | | Kronos-Tokenizer-base | Kronos-Tokenizer-2k |
//! |---|---|---|
//! | `group_size` | 4 | 5 |
//! | `num_groups` (= codebook_dim / group_size) | 5 | 4 |
//!
//! Kronos-mini pairs with the 2k tokenizer. Note also that the paper describes
//! "3 encoder/decoder layers, ff_dim 1024, group_size 9" — the released
//! configs say 4/4, 512, and 4-or-5. **The config.json is authoritative.**

use serde::{Deserialize, Serialize};
use std::fmt;

/// A validation failure in a config file. These are always fatal: a Kronos
/// model that loads with a mismatched `d_model`/`n_heads` will produce garbage
/// rather than fail, so we refuse to build it.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("{field} must be > 0, got {value}")]
    NonPositive { field: &'static str, value: i64 },
    #[error("d_model ({d_model}) must be divisible by n_heads ({n_heads})")]
    HeadDimNotDivisible { d_model: usize, n_heads: usize },
    #[error("codebook_dim ({codebook_dim}) must be divisible by group_size ({group_size})")]
    GroupSizeNotDivisible {
        codebook_dim: usize,
        group_size: usize,
    },
    #[error("group_size must be > 0")]
    ZeroGroupSize,
    #[error("bits fields must sum to <= 31 to stay within i64/u64 packing: s1_bits={s1_bits}, s2_bits={s2_bits}")]
    TooManyBits { s1_bits: u32, s2_bits: u32 },
    #[error("unrecognised field `{field}` in config.json")]
    UnknownField { field: String },
}

/// The geometry a `TransformerBlock` needs.
///
/// Both the predictor and the tokenizer own transformer blocks, but with
/// *different* dimensions: Kronos-small uses 512/8/1024 while the tokenizer
/// uses 256/4/512. `TransformerBlock` is therefore parameterised on this struct
/// rather than on either config, so neither can accidentally borrow the other's
/// head count or embedding width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockGeometry {
    pub d_model: usize,
    pub n_heads: usize,
    pub ff_dim: usize,
    pub head_dim: usize,
    /// Context length the RoPE cache must cover.
    pub max_len: usize,
}

impl BlockGeometry {
    pub fn new(d_model: usize, n_heads: usize, ff_dim: usize, max_len: usize) -> Self {
        assert!(
            d_model.is_multiple_of(n_heads),
            "d_model must be divisible by n_heads"
        );
        Self {
            d_model,
            n_heads,
            ff_dim,
            head_dim: d_model / n_heads,
            max_len,
        }
    }
    pub fn for_predictor(c: &PredictorConfig, max_len: usize) -> Self {
        Self::new(c.d_model, c.n_heads, c.ff_dim, max_len)
    }
    pub fn for_tokenizer(c: &TokenizerConfig, max_len: usize) -> Self {
        Self::new(c.d_model, c.n_heads, c.ff_dim, max_len)
    }
}

macro_rules! require_positive {
    ($field:ident, $val:expr) => {
        if ($val as i64) <= 0 {
            return Err(ConfigError::NonPositive {
                field: stringify!($field),
                value: $val as i64,
            });
        }
    };
}

/// `Kronos` predictor architecture — matches `model/kronos.py::Kronos.__init__`.
///
/// `deny_unknown_fields` is load-bearing: a config we silently ignore a key from
/// is a config we will eventually compute the wrong architecture for.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PredictorConfig {
    pub attn_dropout_p: f32,
    pub d_model: usize,
    pub ff_dim: usize,
    pub ffn_dropout_p: f32,
    pub learn_te: bool,
    pub n_heads: usize,
    pub n_layers: usize,
    pub resid_dropout_p: f32,
    pub s1_bits: u32,
    pub s2_bits: u32,
    pub token_dropout_p: f32,
}

impl PredictorConfig {
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        require_positive!(d_model, self.d_model);
        require_positive!(n_heads, self.n_heads);
        require_positive!(n_layers, self.n_layers);
        require_positive!(ff_dim, self.ff_dim);
        if !self.d_model.is_multiple_of(self.n_heads) {
            return Err(ConfigError::HeadDimNotDivisible {
                d_model: self.d_model,
                n_heads: self.n_heads,
            });
        }
        if self.s1_bits + self.s2_bits > 31 {
            return Err(ConfigError::TooManyBits {
                s1_bits: self.s1_bits,
                s2_bits: self.s2_bits,
            });
        }
        Ok(())
    }

    #[inline]
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }
    #[inline]
    pub fn vocab_s1(&self) -> usize {
        1usize << self.s1_bits
    }
    #[inline]
    pub fn vocab_s2(&self) -> usize {
        1usize << self.s2_bits
    }
    /// `DependencyAwareLayer` fixes `n_heads=4` (model/module.py:447) regardless
    /// of the predictor's own head count.
    #[inline]
    pub fn dep_heads(&self) -> usize {
        4
    }
    #[inline]
    pub fn dep_head_dim(&self) -> usize {
        self.d_model / self.dep_heads()
    }
}

/// `KronosTokenizer` architecture — matches `model/kronos.py::KronosTokenizer.__init__`.
///
/// Note `n_enc_layers`/`n_dec_layers` are the values in config.json; the module
/// builds `n_enc_layers - 1` / `n_dec_layers - 1` blocks (module.py:60-68), which
/// is why config 4 yields 3 transformer blocks on each side.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerConfig {
    pub attn_dropout_p: f32,
    pub beta: f32,
    pub d_in: usize,
    pub d_model: usize,
    pub ff_dim: usize,
    pub ffn_dropout_p: f32,
    pub gamma: f32,
    pub gamma0: f32,
    pub group_size: usize,
    pub n_dec_layers: usize,
    pub n_enc_layers: usize,
    pub n_heads: usize,
    pub resid_dropout_p: f32,
    pub s1_bits: u32,
    pub s2_bits: u32,
    pub zeta: f32,
}

impl TokenizerConfig {
    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        require_positive!(d_model, self.d_model);
        require_positive!(d_in, self.d_in);
        require_positive!(n_heads, self.n_heads);
        require_positive!(n_enc_layers, self.n_enc_layers);
        require_positive!(n_dec_layers, self.n_dec_layers);
        require_positive!(ff_dim, self.ff_dim);
        if !self.d_model.is_multiple_of(self.n_heads) {
            return Err(ConfigError::HeadDimNotDivisible {
                d_model: self.d_model,
                n_heads: self.n_heads,
            });
        }
        if self.group_size == 0 {
            return Err(ConfigError::ZeroGroupSize);
        }
        // BinarySphericalQuantizer asserts embed_dim group_size.is_multiple_of(0) (module.py:58)
        if !self.codebook_dim().is_multiple_of(self.group_size) {
            return Err(ConfigError::GroupSizeNotDivisible {
                codebook_dim: self.codebook_dim(),
                group_size: self.group_size,
            });
        }
        if self.s1_bits + self.s2_bits > 31 {
            return Err(ConfigError::TooManyBits {
                s1_bits: self.s1_bits,
                s2_bits: self.s2_bits,
            });
        }
        Ok(())
    }

    #[inline]
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }
    /// BSQ operates on s1_bits + s2_bits dimensions.
    #[inline]
    pub fn codebook_dim(&self) -> usize {
        (self.s1_bits + self.s2_bits) as usize
    }
    #[inline]
    pub fn num_groups(&self) -> usize {
        self.codebook_dim() / self.group_size
    }
    /// `nn.ModuleList([... for _ in range(n_enc_layers - 1)])` (module.py:60-63)
    #[inline]
    pub fn n_enc_blocks(&self) -> usize {
        self.n_enc_layers - 1
    }
    #[inline]
    pub fn n_dec_blocks(&self) -> usize {
        self.n_dec_layers - 1
    }
    /// `q_scale = 1/sqrt(embed_dim)` when `l2_norm` (module.py:95)
    #[inline]
    pub fn q_scale(&self) -> f32 {
        1.0 / (self.codebook_dim() as f32).sqrt()
    }
}

impl fmt::Display for PredictorConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Kronos(n_layers={}, d_model={}, d_ff={}, heads={}, head_dim={}, s1_bits={}, s2_bits={}, \
             vocab_s1={}, vocab_s2={}, learn_te={})",
            self.n_layers, self.d_model, self.ff_dim, self.n_heads, self.head_dim(),
            self.s1_bits, self.s2_bits, self.vocab_s1(), self.vocab_s2(), self.learn_te
        )
    }
}

impl fmt::Display for TokenizerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "KronosTokenizer(enc_blocks={}, dec_blocks={}, d_model={}, d_ff={}, heads={}, d_in={}, \
             codebook_dim={}, group_size={}, num_groups={}, q_scale={:.6})",
            self.n_enc_blocks(), self.n_dec_blocks(), self.d_model, self.ff_dim,
            self.n_heads, self.d_in, self.codebook_dim(), self.group_size,
            self.num_groups(), self.q_scale()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three released predictor configs, verbatim.
    const SMALL: &str = r#"{"attn_dropout_p":0.1,"d_model":512,"ff_dim":1024,"ffn_dropout_p":0.25,
        "learn_te":true,"n_heads":8,"n_layers":8,"resid_dropout_p":0.25,"s1_bits":10,
        "s2_bits":10,"token_dropout_p":0.1}"#;
    const BASE: &str = r#"{"attn_dropout_p":0.0,"d_model":832,"ff_dim":2048,"ffn_dropout_p":0.2,
        "learn_te":true,"n_heads":16,"n_layers":12,"resid_dropout_p":0.2,"s1_bits":10,
        "s2_bits":10,"token_dropout_p":0.0}"#;
    const MINI: &str = r#"{"attn_dropout_p":0.0,"d_model":256,"ff_dim":512,"ffn_dropout_p":0.2,
        "learn_te":true,"n_heads":4,"n_layers":4,"resid_dropout_p":0.2,"s1_bits":10,
        "s2_bits":10,"token_dropout_p":0.0}"#;

    /// The two released tokenizer configs — these differ in `group_size`.
    const TOK_BASE: &str = r#"{"attn_dropout_p":0.0,"beta":0.05,"d_in":6,"d_model":256,
        "ff_dim":512,"ffn_dropout_p":0.0,"gamma":1.1,"gamma0":1.0,"group_size":4,
        "n_dec_layers":4,"n_enc_layers":4,"n_heads":4,"resid_dropout_p":0.0,"s1_bits":10,
        "s2_bits":10,"zeta":0.05}"#;
    const TOK_2K: &str = r#"{"attn_dropout_p":0.0,"beta":0.05,"d_in":6,"d_model":256,
        "ff_dim":512,"ffn_dropout_p":0.0,"gamma":1.1,"gamma0":1.0,"group_size":5,
        "n_dec_layers":4,"n_enc_layers":4,"n_heads":4,"resid_dropout_p":0.0,"s1_bits":10,
        "s2_bits":10,"zeta":0.05}"#;

    #[test]
    fn parses_and_validates_all_released_configs() {
        for (name, s) in [("small", SMALL), ("base", BASE), ("mini", MINI)] {
            let c = PredictorConfig::from_json(s).unwrap_or_else(|e| panic!("{name}: {e}"));
            c.validate().unwrap_or_else(|e| panic!("{name}: {e}"));
        }
        let b = TokenizerConfig::from_json(TOK_BASE).unwrap();
        b.validate().unwrap();
        let k = TokenizerConfig::from_json(TOK_2K).unwrap();
        k.validate().unwrap();
    }

    #[test]
    fn derived_values_match_python() {
        let c = PredictorConfig::from_json(SMALL).unwrap();
        assert_eq!(c.head_dim(), 64); // module.py:320
        assert_eq!(c.vocab_s1(), 1024); // 2**s1_bits, module.py:212
        assert_eq!(c.dep_heads(), 4); // module.py:447 hardcodes 4
        assert_eq!(c.dep_head_dim(), 128);

        let b = TokenizerConfig::from_json(TOK_BASE).unwrap();
        assert_eq!(b.codebook_dim(), 20); // module.py:55
        assert_eq!(b.n_enc_blocks(), 3); // range(n_enc_layers - 1)
        assert_eq!(b.n_dec_blocks(), 3);
        assert_eq!(b.num_groups(), 5);
        assert!((b.q_scale() - 0.223_606_8).abs() < 1e-9); // 1/sqrt(20)

        let k = TokenizerConfig::from_json(TOK_2K).unwrap();
        assert_eq!(k.num_groups(), 4); // different tokenizer, different grouping
    }

    #[test]
    fn rejects_unknown_fields() {
        let bad = r#"{"attn_dropout_p":0.1,"d_model":512,"ff_dim":1024,"ffn_dropout_p":0.25,
            "learn_te":true,"n_heads":8,"n_layers":8,"resid_dropout_p":0.25,"s1_bits":10,
            "s2_bits":10,"token_dropout_p":0.1,"surprise_me":1}"#;
        assert!(PredictorConfig::from_json(bad).is_err());
    }

    #[test]
    fn rejects_invalid_geometry() {
        let bad = SMALL.replace("\"d_model\":512", "\"d_model\":500");
        let c = PredictorConfig::from_json(&bad).unwrap();
        assert!(matches!(
            c.validate(),
            Err(ConfigError::HeadDimNotDivisible { .. })
        ));

        let bad = TOK_BASE.replace("\"group_size\":4", "\"group_size\":7"); // 20 % 7 != 0
        let k = TokenizerConfig::from_json(&bad).unwrap();
        assert!(matches!(
            k.validate(),
            Err(ConfigError::GroupSizeNotDivisible { .. })
        ));
    }
}
