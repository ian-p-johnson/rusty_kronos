//! Strict safetensors loading.
//!
//! The exit criterion for Phase 1 is that a *partial* or *mismatched* load is
//! impossible to miss. A quietly-missing `k_proj.bias` produces a model that
//! runs, produces numbers, and is wrong — the worst failure mode available.
//! So this loader is deliberately unforgiving:
//!
//! 1. the expected key set is *derived from the config*, not hardcoded;
//! 2. every expected key must be present;
//! 3. every key present in the file must be expected (catches renamed/typo'd
//!    checkpoints and partial re-saves);
//! 4. shapes are checked against what the architecture will multiply.
//!
//! The key names are exactly the PyTorch module paths, because the released
//! checkpoints are written by `PyTorchModelHubMixin` from the module tree.

use crate::config::{PredictorConfig, TokenizerConfig};
use candle_core::{DType, Device, Tensor};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("could not read safetensors: {0}")]
    Io(String),
    #[error("tensor `{name}` expected shape {expected:?} but found {found:?}")]
    ShapeMismatch {
        name: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    #[error("{n} tensor(s) required by the architecture are missing: {missing:?}")]
    Missing { missing: Vec<String>, n: usize },
    #[error("{n} tensor(s) present in the file are not used by the architecture: {unexpected:?}")]
    Unexpected { unexpected: Vec<String>, n: usize },
    #[error("tensor `{name}` has dtype {found}, expected f32 in the checkpoint")]
    DTypeMismatch { name: String, found: String },
}

/// What one layer's worth of parameters must contain. Used by both the
/// predictor and the tokenizer, which share `TransformerBlock`.
fn transformer_block_keys(prefix: &str, n: usize) -> Vec<String> {
    let mut v = Vec::with_capacity(n * 11);
    for i in 0..n {
        let p = format!("{prefix}.{i}");
        v.push(format!("{p}.norm1.weight"));
        v.push(format!("{p}.norm2.weight"));
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            v.push(format!("{p}.self_attn.{proj}.weight"));
            v.push(format!("{p}.self_attn.{proj}.bias"));
        }
        // FFN is Linear(..., bias=False) (module.py:275-277) -> no biases
        v.push(format!("{p}.ffn.w1.weight"));
        v.push(format!("{p}.ffn.w2.weight"));
        v.push(format!("{p}.ffn.w3.weight"));
        // RoPE inv_freq is a registered buffer, so it IS persisted
        v.push(format!("{p}.self_attn.rotary.inv_freq"));
    }
    v
}

/// Every tensor the `Kronos` predictor requires, derived from its config.
pub fn expected_predictor_keys(c: &PredictorConfig) -> BTreeSet<String> {
    let d = c.d_model;
    let mut k: BTreeSet<String> = BTreeSet::new();
    k.extend(transformer_block_keys("transformer", c.n_layers));

    // HierarchicalEmbedding
    k.insert("embedding.emb_s1.weight".to_string());
    k.insert("embedding.emb_s2.weight".to_string());
    k.insert("embedding.fusion_proj.weight".into());
    k.insert("embedding.fusion_proj.bias".into());

    // TemporalEmbedding: learn_te=True in all released configs, so these are
    // real nn.Embedding tables sized by the calendar vocabularies (module.py:540-545).
    for name in ["minute", "hour", "weekday", "day", "month"] {
        k.insert(format!("time_emb.{name}_embed.weight"));
    }

    k.insert("norm.weight".into());

    // DependencyAwareLayer (module.py:446-462)
    for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
        k.insert(format!("dep_layer.cross_attn.{proj}.weight"));
        k.insert(format!("dep_layer.cross_attn.{proj}.bias"));
    }
    k.insert("dep_layer.cross_attn.rotary.inv_freq".into());
    k.insert("dep_layer.norm.weight".into());

    // DualHead
    k.insert("head.proj_s1.weight".into());
    k.insert("head.proj_s1.bias".into());
    k.insert("head.proj_s2.weight".into());
    k.insert("head.proj_s2.bias".into());

    let _ = d;
    k
}

/// Every tensor the `KronosTokenizer` requires, derived from its config.
pub fn expected_tokenizer_keys(c: &TokenizerConfig) -> BTreeSet<String> {
    let mut k: BTreeSet<String> = BTreeSet::new();
    k.extend(transformer_block_keys("encoder", c.n_enc_blocks()));
    k.extend(transformer_block_keys("decoder", c.n_dec_blocks()));

    k.insert("embed.weight".into());
    k.insert("embed.bias".into());
    k.insert("head.weight".into());
    k.insert("head.bias".into());
    k.insert("quant_embed.weight".into());
    k.insert("quant_embed.bias".into());
    k.insert("post_quant_embed.weight".into());
    k.insert("post_quant_embed.bias".into());
    // Present in the file but UNUSED at inference (only `forward` touches it,
    // module.py:99). Loaded anyway: if it is missing, the checkpoint is not the
    // one we validated against and that is worth knowing.
    k.insert("post_quant_embed_pre.weight".into());
    k.insert("post_quant_embed_pre.bias".into());

    // BinarySphericalQuantizer buffers. `group_codebook` is registered with
    // `persistent=False` (module.py:78) so it is deliberately absent and must
    // be reconstructed from `group_size` at runtime.
    k.insert("tokenizer.bsq.basis".into());
    k.insert("tokenizer.bsq.group_basis".into());
    k
}

/// Shape of a tensor the architecture will actually multiply. `None` means the
/// value is a 1-D buffer (norm weight, embedding table row, inv_freq) that has
/// a length we can check but not a fixed shape.
fn expected_shape(name: &str, p: &PredictorConfig) -> Option<Vec<usize>> {
    let d = p.d_model;
    Some(match name {
        n if n.ends_with(".norm1.weight")
            || n.ends_with(".norm2.weight")
            || n == "norm.weight"
            || n == "dep_layer.norm.weight" =>
        {
            vec![d]
        }
        n if n.ends_with(".self_attn.q_proj.weight")
            || n.ends_with(".self_attn.k_proj.weight")
            || n.ends_with(".self_attn.v_proj.weight") =>
        {
            vec![d, d]
        }
        n if n.ends_with(".self_attn.out_proj.weight") => vec![d, d],
        n if n.ends_with(".self_attn.q_proj.bias")
            || n.ends_with(".self_attn.k_proj.bias")
            || n.ends_with(".self_attn.v_proj.bias")
            || n.ends_with(".self_attn.out_proj.bias") =>
        {
            vec![d]
        }
        n if n.ends_with(".ffn.w1.weight") || n.ends_with(".ffn.w3.weight") => vec![p.ff_dim, d],
        n if n.ends_with(".ffn.w2.weight") => vec![d, p.ff_dim],
        n if n.ends_with(".self_attn.rotary.inv_freq") => vec![p.head_dim() / 2],
        "embedding.emb_s1.weight" => vec![p.vocab_s1(), d],
        "embedding.emb_s2.weight" => vec![p.vocab_s2(), d],
        "embedding.fusion_proj.weight" => vec![d, 2 * d],
        "embedding.fusion_proj.bias" => vec![d],
        "time_emb.minute_embed.weight" => vec![60, d],
        "time_emb.hour_embed.weight" => vec![24, d],
        "time_emb.weekday_embed.weight" => vec![7, d],
        "time_emb.day_embed.weight" => vec![32, d],
        "time_emb.month_embed.weight" => vec![13, d],
        "head.proj_s1.weight" => vec![p.vocab_s1(), d],
        "head.proj_s1.bias" => vec![p.vocab_s1()],
        "head.proj_s2.weight" => vec![p.vocab_s2(), d],
        "head.proj_s2.bias" => vec![p.vocab_s2()],
        // dep_layer projections carry the SAME shapes as a self-attention block
        // even though their module path is `cross_attn` not `self_attn`, so they
        // need explicit arms rather than falling into the `.self_attn.*` ones above.
        n if n.starts_with("dep_layer.cross_attn.") && n.ends_with(".weight") => vec![d, d],
        n if n.starts_with("dep_layer.cross_attn.") && n.ends_with(".bias") => vec![d],
        n if n.starts_with("dep_layer.cross_attn.") => vec![p.dep_head_dim() / 2],
        _ => return None,
    })
}

fn expected_shape_tok(name: &str, t: &TokenizerConfig) -> Option<Vec<usize>> {
    let d = t.d_model;
    Some(match name {
        n if n.ends_with(".norm1.weight") || n.ends_with(".norm2.weight") => vec![d],
        n if n.ends_with(".self_attn.q_proj.weight")
            || n.ends_with(".self_attn.k_proj.weight")
            || n.ends_with(".self_attn.v_proj.weight")
            || n.ends_with(".self_attn.out_proj.weight") =>
        {
            vec![d, d]
        }
        n if n.ends_with(".self_attn.q_proj.bias")
            || n.ends_with(".self_attn.k_proj.bias")
            || n.ends_with(".self_attn.v_proj.bias")
            || n.ends_with(".self_attn.out_proj.bias") =>
        {
            vec![d]
        }
        n if n.ends_with(".ffn.w1.weight") || n.ends_with(".ffn.w3.weight") => vec![t.ff_dim, d],
        n if n.ends_with(".ffn.w2.weight") => vec![d, t.ff_dim],
        n if n.ends_with(".self_attn.rotary.inv_freq") => vec![t.head_dim() / 2],
        "embed.weight" => vec![d, t.d_in],
        "embed.bias" => vec![d],
        "head.weight" => vec![t.d_in, d],
        "head.bias" => vec![t.d_in],
        "quant_embed.weight" => vec![t.codebook_dim(), d],
        "quant_embed.bias" => vec![t.codebook_dim()],
        "post_quant_embed.weight" => vec![d, t.codebook_dim()],
        "post_quant_embed.bias" => vec![d],
        "post_quant_embed_pre.weight" => vec![d, t.s1_bits as usize],
        "post_quant_embed_pre.bias" => vec![d],
        "tokenizer.bsq.basis" => vec![t.codebook_dim()],
        "tokenizer.bsq.group_basis" => vec![t.group_size],
        _ => return None,
    })
}

/// Dtype a checkpoint tensor must have.
///
/// Everything is f32 except the two BSQuantizer basis buffers, which PyTorch
/// builds with `2 ** torch.arange(...)` (module.py:68-69). `torch.arange`
/// without a `dtype=` argument yields int64, and `register_buffer` preserves
/// that, so these are genuinely i64 on disk and we keep them i64 here.
fn expected_dtype(name: &str) -> DType {
    match name {
        "tokenizer.bsq.basis" | "tokenizer.bsq.group_basis" => DType::I64,
        _ => DType::F32,
    }
}

/// A validated, device-resident set of weights.
///
/// `get` is infallible after construction precisely because construction proved
/// the key is present — the model code can then be written without threading
/// `Result` through every `forward`.
pub struct Weights {
    tensors: BTreeMap<String, Tensor>,
    pub dtype: DType,
}

impl Weights {
    /// Validate a checkpoint's key set, dtypes and shapes *without* touching
    /// the GPU. Used by tests and by `kronos info`.
    pub fn validate_file(
        path: &std::path::Path,
        expected: &BTreeSet<String>,
        shape_of: impl Fn(&str) -> Option<Vec<usize>>,
    ) -> Result<usize, LoadError> {
        let raw = read_cpuside(path)?;
        let names: BTreeSet<String> = raw.keys().cloned().collect();

        let missing: Vec<String> = expected.difference(&names).cloned().collect();
        if !missing.is_empty() {
            return Err(LoadError::Missing {
                n: missing.len(),
                missing: missing[..missing.len().min(12)].to_vec(),
            });
        }
        let unexpected: Vec<String> = names.difference(expected).cloned().collect();
        if !unexpected.is_empty() {
            return Err(LoadError::Unexpected {
                n: unexpected.len(),
                unexpected: unexpected[..unexpected.len().min(12)].to_vec(),
            });
        }
        for name in expected {
            let t = &raw[name];
            let want = expected_dtype(name);
            if t.dtype() != want {
                return Err(LoadError::DTypeMismatch {
                    name: name.clone(),
                    found: format!("{:?} (expected {:?})", t.dtype(), want),
                });
            }
            if let Some(exp) = shape_of(name) {
                let found: Vec<usize> = t.shape().dims().to_vec();
                if found != exp {
                    return Err(LoadError::ShapeMismatch {
                        name: name.clone(),
                        expected: exp,
                        found,
                    });
                }
            }
        }
        Ok(expected.len())
    }

    pub fn load(
        path: &std::path::Path,
        expected: &BTreeSet<String>,
        dtype: DType,
        dev: &Device,
    ) -> Result<Self, LoadError> {
        let raw = read_cpuside(path)?;
        let mut tensors = BTreeMap::new();
        for name in expected {
            let t = raw[name].to_device(dev).and_then(|t| t.to_dtype(dtype));
            tensors.insert(
                name.clone(),
                t.map_err(|e| LoadError::Io(format!("{name}: {e}")))?,
            );
        }
        Ok(Self { tensors, dtype })
    }

    /// Infallible by construction — see the type-level note above.
    pub fn get(&self, name: &str) -> &Tensor {
        self.tensors
            .get(name)
            .unwrap_or_else(|| panic!("weight `{name}` absent — loader validation was skipped"))
    }

    pub fn get_cloned(&self, name: &str) -> Tensor {
        self.get(name).clone()
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Total parameter count, for reporting and for cross-checking against the
    /// paper's Table 1 (24.7 M / 102.3 M / 499.2 M).
    pub fn numel(&self) -> usize {
        self.tensors.values().map(|t| t.elem_count()).sum()
    }
}

/// Read a safetensors file into a CPU-side name->Tensor map.
///
/// We deliberately go through `load_buffer` rather than holding a mmap: the
/// files are <=100 MB and load once at startup, and this keeps us on candle's
/// public API instead of pinning the safetensors crate's version.
fn read_cpuside(path: &std::path::Path) -> Result<BTreeMap<String, Tensor>, LoadError> {
    let bytes =
        std::fs::read(path).map_err(|e| LoadError::Io(format!("{}: {e}", path.display())))?;
    let map = candle_core::safetensors::load_buffer(&bytes, &Device::Cpu)
        .map_err(|e| LoadError::Io(format!("{}: {e}", path.display())))?;
    Ok(map.into_iter().collect())
}

pub fn load_predictor_weights(
    path: &std::path::Path,
    cfg: &PredictorConfig,
    dtype: DType,
    dev: &Device,
) -> Result<Weights, LoadError> {
    let expected = expected_predictor_keys(cfg);
    Weights::validate_file(path, &expected, |n| expected_shape(n, cfg))?;
    Weights::load(path, &expected, dtype, dev)
}

pub fn load_tokenizer_weights(
    path: &std::path::Path,
    cfg: &TokenizerConfig,
    dtype: DType,
    dev: &Device,
) -> Result<Weights, LoadError> {
    let expected = expected_tokenizer_keys(cfg);
    Weights::validate_file(path, &expected, |n| expected_shape_tok(n, cfg))?;
    Weights::load(path, &expected, dtype, dev)
}
impl Weights {
    /// Iterate in stable key order. The CLI relies on this for reproducible
    /// manifest output.
    pub fn iter_sorted(&self) -> impl Iterator<Item = (&String, &Tensor)> {
        self.tensors.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_small() -> PredictorConfig {
        PredictorConfig::from_json(
            r#"{"attn_dropout_p":0.1,"d_model":512,"ff_dim":1024,"ffn_dropout_p":0.25,
            "learn_te":true,"n_heads":8,"n_layers":8,"resid_dropout_p":0.25,"s1_bits":10,
            "s2_bits":10,"token_dropout_p":0.1}"#,
        )
        .unwrap()
    }

    #[test]
    fn predictor_key_count_matches_checkpoint() {
        // Kronos-small: 8 layers x 11 + 2 emb + 1 fusion_w + 1 fusion_b + 5 temb + 1 norm
        //                + 4 proj x2 + 1 inv_freq + 1 dep norm + 4 head
        let n = expected_predictor_keys(&cfg_small()).len();
        assert_eq!(
            n, 136,
            "released Kronos-small model.safetensors has 136 tensors"
        );
    }

    #[test]
    fn tokenizer_key_count_matches_checkpoint() {
        let t = TokenizerConfig::from_json(
            r#"{"attn_dropout_p":0.0,"beta":0.05,"d_in":6,"d_model":256,"ff_dim":512,
            "ffn_dropout_p":0.0,"gamma":1.1,"gamma0":1.0,"group_size":4,"n_dec_layers":4,
            "n_enc_layers":4,"n_heads":4,"resid_dropout_p":0.0,"s1_bits":10,"s2_bits":10,
            "zeta":0.05}"#,
        )
        .unwrap();
        let n = expected_tokenizer_keys(&t).len();
        // 3 enc + 3 dec blocks x 11 = 66, + embed(2) head(2) quant_embed(2)
        //   + post_quant_embed(2) + post_quant_embed_pre(2) + bsq buffers(2) = 96
        assert_eq!(
            n, 96,
            "released Kronos-Tokenizer-base model.safetensors has 96 tensors"
        );
    }

    #[test]
    fn ffn_has_no_biases_but_attention_does() {
        let keys = expected_predictor_keys(&cfg_small());
        assert!(keys.contains("transformer.0.ffn.w1.weight"));
        assert!(!keys.contains("transformer.0.ffn.w1.bias"));
        assert!(keys.contains("transformer.0.self_attn.q_proj.bias"));
    }

    #[test]
    fn group_codebook_is_deliberately_absent() {
        let t = TokenizerConfig::from_json(
            r#"{"attn_dropout_p":0.0,"beta":0.05,"d_in":6,"d_model":256,"ff_dim":512,
            "ffn_dropout_p":0.0,"gamma":1.1,"gamma0":1.0,"group_size":4,"n_dec_layers":4,
            "n_enc_layers":4,"n_heads":4,"resid_dropout_p":0.0,"s1_bits":10,"s2_bits":10,
            "zeta":0.05}"#,
        )
        .unwrap();
        let keys = expected_tokenizer_keys(&t);
        // persistent=False (module.py:78) -> reconstructed at runtime from group_size
        assert!(!keys.contains("tokenizer.bsq.group_codebook"));
        assert!(keys.contains("tokenizer.bsq.basis"));
    }

    #[test]
    fn shapes_are_derived_not_hardcoded() {
        let p = cfg_small();
        assert_eq!(
            expected_shape("transformer.0.ffn.w1.weight", &p),
            Some(vec![1024, 512])
        );
        assert_eq!(
            expected_shape("transformer.0.ffn.w2.weight", &p),
            Some(vec![512, 1024])
        );
        assert_eq!(
            expected_shape("embedding.fusion_proj.weight", &p),
            Some(vec![512, 1024])
        );
        assert_eq!(
            expected_shape("head.proj_s1.weight", &p),
            Some(vec![1024, 512])
        );
        assert_eq!(
            expected_shape("transformer.0.self_attn.rotary.inv_freq", &p),
            Some(vec![32])
        );
        // dep_layer uses 4 heads regardless of the predictor's 8
        assert_eq!(
            expected_shape("dep_layer.cross_attn.q_proj.weight", &p),
            Some(vec![512, 512])
        );
        assert_eq!(
            expected_shape("dep_layer.cross_attn.q_proj.bias", &p),
            Some(vec![512])
        );
        assert_eq!(
            expected_shape("dep_layer.cross_attn.rotary.inv_freq", &p),
            Some(vec![64])
        );
    }
}
