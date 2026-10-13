//! Embeddings and output heads — `model/module.py:400-443`, `:486-513`, `:536-562`.

use crate::config::PredictorConfig;
use crate::ops::{embedding_lookup, linear};
use crate::weights::Weights;
use candle_core::{DType, Result, Tensor};

/// `HierarchicalEmbedding` — `model/module.py:400-443`.
///
/// `fusion_proj(cat([emb_s1(s1) * sqrt(d), emb_s2(s2) * sqrt(d)], -1))`.
///
/// The `sqrt(d_model)` scaling is unusual and easy to drop; it is what makes the
/// concatenated embedding have unit-ish scale before the fusion projection.
pub struct HierarchicalEmbedding {
    emb_s1: Tensor,
    emb_s2: Tensor,
    fusion_w: Tensor,
    fusion_b: Tensor,
    scale: f64,
}

impl HierarchicalEmbedding {
    pub fn load(w: &Weights, cfg: &PredictorConfig) -> Result<Self> {
        Ok(Self {
            emb_s1: w.get("embedding.emb_s1.weight").clone(),
            emb_s2: w.get("embedding.emb_s2.weight").clone(),
            fusion_w: w.get("embedding.fusion_proj.weight").clone(),
            fusion_b: w.get("embedding.fusion_proj.bias").clone(),
            scale: (cfg.d_model as f64).sqrt(),
        })
    }

    pub fn emb_s1(&self) -> &Tensor {
        &self.emb_s1
    }

    pub fn forward(&self, s1_ids: &Tensor, s2_ids: &Tensor) -> Result<Tensor> {
        let a = embedding_lookup(&self.emb_s1, s1_ids)?.affine(self.scale, 0.0)?;
        let b = embedding_lookup(&self.emb_s2, s2_ids)?.affine(self.scale, 0.0)?;
        let cat = Tensor::cat(&[&a, &b], a.dims().len() - 1)?;
        linear(&cat, &self.fusion_w, Some(&self.fusion_b))
    }
}

/// `TemporalEmbedding` — `model/module.py:536-562`.
///
/// Five calendar features (minute-of-hour, hour-of-day, weekday, day-of-month,
/// month-of-year) each looked up and **summed** (module.py:562).
///
/// Upstream supports a `FixedEmbedding` (sinusoidal, non-learnable) path when
/// `learn_te` is false. All three released checkpoints set `learn_te: true` and
/// ship real embedding tables, so this port implements the learnable path only
/// and asserts on the flag rather than silently ignoring it.
pub struct TemporalEmbedding {
    minute: Tensor,
    hour: Tensor,
    weekday: Tensor,
    day: Tensor,
    month: Tensor,
}

impl TemporalEmbedding {
    pub fn load(w: &Weights, cfg: &PredictorConfig) -> Result<Self> {
        assert!(
            cfg.learn_te,
            "learn_te=false selects FixedEmbedding (model/module.py:546), which this \
             port does not implement; all released checkpoints use learn_te=true"
        );
        Ok(Self {
            minute: w.get("time_emb.minute_embed.weight").clone(),
            hour: w.get("time_emb.hour_embed.weight").clone(),
            weekday: w.get("time_emb.weekday_embed.weight").clone(),
            day: w.get("time_emb.day_embed.weight").clone(),
            month: w.get("time_emb.month_embed.weight").clone(),
        })
    }

    /// `stamp` is `[B, L, 5]` of calendar features; upstream casts to long.
    pub fn forward(&self, stamp: &Tensor) -> Result<Tensor> {
        let long = stamp.to_dtype(DType::U32)?;
        let mut acc = embedding_lookup(&self.minute, &long.narrow(2, 0, 1)?.squeeze(2)?)?;
        acc = acc.broadcast_add(&embedding_lookup(
            &self.hour,
            &long.narrow(2, 1, 1)?.squeeze(2)?,
        )?)?;
        acc = acc.broadcast_add(&embedding_lookup(
            &self.weekday,
            &long.narrow(2, 2, 1)?.squeeze(2)?,
        )?)?;
        acc = acc.broadcast_add(&embedding_lookup(
            &self.day,
            &long.narrow(2, 3, 1)?.squeeze(2)?,
        )?)?;
        acc.broadcast_add(&embedding_lookup(
            &self.month,
            &long.narrow(2, 4, 1)?.squeeze(2)?,
        )?)
    }
}

/// `DualHead` — `model/module.py:486-513`.
///
/// Two independent Linear layers over vocab sizes `2**s1_bits` and `2**s2_bits`
/// (1,024 each for every released checkpoint).
pub struct DualHead {
    proj_s1_w: Tensor,
    proj_s1_b: Tensor,
    proj_s2_w: Tensor,
    proj_s2_b: Tensor,
}

impl DualHead {
    pub fn load(w: &Weights) -> Result<Self> {
        Ok(Self {
            proj_s1_w: w.get("head.proj_s1.weight").clone(),
            proj_s1_b: w.get("head.proj_s1.bias").clone(),
            proj_s2_w: w.get("head.proj_s2.weight").clone(),
            proj_s2_b: w.get("head.proj_s2.bias").clone(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        linear(x, &self.proj_s1_w, Some(&self.proj_s1_b))
    }

    /// `DualHead.cond_forward` — the fine-subtoken head.
    pub fn cond_forward(&self, x: &Tensor) -> Result<Tensor> {
        linear(x, &self.proj_s2_w, Some(&self.proj_s2_b))
    }
}
