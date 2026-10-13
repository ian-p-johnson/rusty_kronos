//! `Kronos` — the autoregressive predictor — `model/kronos.py:180-328`.

use crate::block::{DependencyAwareLayer, TransformerBlock};
use crate::config::BlockGeometry;
use crate::config::PredictorConfig;
use crate::embed::{DualHead, HierarchicalEmbedding, TemporalEmbedding};
#[allow(unused_imports)]
use crate::ops::linear;
use crate::ops::{embedding_lookup, rms_norm};
use crate::weights::Weights;
use candle_core::{DType, Result, Tensor};

/// `(s1_logits, context)` — the two halves of `decode_s1`.
pub struct DecodeS1 {
    pub s1_logits: Tensor,
    pub context: Tensor,
}

/// `Kronos` — decoder-only causal transformer over hierarchical K-line tokens.
///
/// Note the two-phase decode: `decode_s1` runs the full transformer and returns
/// the *unnormalised* logits plus the post-norm hidden state, then `decode_s2`
/// conditions on the sampled coarse token via cross-attention. This is why
/// inference costs **two** forward passes per generated step (paper Table 12).
pub struct Kronos {
    cfg: PredictorConfig,
    embedding: HierarchicalEmbedding,
    time_emb: TemporalEmbedding,
    transformer: Vec<TransformerBlock>,
    norm: Tensor,
    head: DualHead,
    dep_layer: DependencyAwareLayer,
    eps: f32,
}

impl Kronos {
    pub fn load(
        w: &Weights,
        cfg: &PredictorConfig,
        max_len: usize,
        dev: &candle_core::Device,
        dtype: DType,
    ) -> Result<Self> {
        let geo = BlockGeometry::for_predictor(cfg, max_len);
        let rope = crate::ops::RotaryCache::new(geo.head_dim, max_len, dev, dtype)?;
        let mut transformer = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            transformer.push(TransformerBlock::load(
                &format!("transformer.{i}"),
                w,
                geo,
                rope.clone(),
            )?);
        }
        Ok(Self {
            embedding: HierarchicalEmbedding::load(w, cfg)?,
            time_emb: TemporalEmbedding::load(w, cfg)?,
            transformer,
            norm: w.get("norm.weight").clone(),
            head: DualHead::load(w)?,
            dep_layer: DependencyAwareLayer::load(w, cfg, max_len, dev, dtype)?,
            eps: 1e-5,
            cfg: cfg.clone(),
        })
    }

    /// `Kronos.decode_s1` — `model/kronos.py:278-308`.
    ///
    /// Full causal forward over `[s1_ids, s2_ids]` plus the temporal stamp.
    /// Returns both the coarse-subtoken logits and the post-norm context, because
    /// `decode_s2` needs the latter as cross-attention key/value.
    pub fn decode_s1(&self, s1_ids: &Tensor, s2_ids: &Tensor, stamp: &Tensor) -> Result<DecodeS1> {
        let mut x = self.embedding.forward(s1_ids, s2_ids)?;
        x = x.broadcast_add(&self.time_emb.forward(stamp)?)?;
        // token_drop is nn.Dropout and is a no-op at eval (module.py:213)
        for blk in &self.transformer {
            x = blk.forward(&x)?;
        }
        let context = rms_norm(&x, &self.norm, self.eps)?;
        let s1_logits = self.head.forward(&context)?;
        Ok(DecodeS1 { s1_logits, context })
    }

    /// `Kronos.decode_s2` — `model/kronos.py:310-328`.
    ///
    /// `s1_ids` here is the *sampled* coarse token (shape `[B, 1]`), NOT the full
    /// sequence. The dependency-aware layer cross-attends this single query
    /// against the whole context window.
    pub fn decode_s2(&self, context: &Tensor, s1_ids: &Tensor) -> Result<Tensor> {
        // emb_s1 is called directly (module.py:326), so no sqrt(d) scaling here
        let sibling_embed = embedding_lookup(self.embedding.emb_s1(), s1_ids)?;
        let x2 = self.dep_layer.forward(context, &sibling_embed)?;
        self.head.cond_forward(&x2)
    }

    pub fn config(&self) -> &PredictorConfig {
        &self.cfg
    }
}
