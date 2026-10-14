//! `TransformerBlock`, attention, and the dependency-aware cross-attention layer.
//!
//! Shapes are `[batch, seq, d_model]` on the way in and out; attention
//! internally uses `[batch, heads, seq, head_dim]`.

use crate::config::{BlockGeometry, PredictorConfig};
use crate::ops::{linear, rms_norm, sdpa, RotaryCache};
use crate::weights::Weights;
use candle_core::{DType, Result, Tensor};

/// `FeedForward` (SwiGLU) — `model/module.py:271-281`.
///
/// `w2(silu(w1(x)) * w3(x))`. All three Linear layers are `bias=False`
/// (module.py:275-277), which is why the checkpoints carry no FFN biases.
pub struct FeedForward {
    w1: Tensor,
    w3: Tensor,
    w2: Tensor,
}

impl FeedForward {
    fn load(prefix: &str, w: &Weights) -> Result<Self> {
        Ok(Self {
            w1: w.get(&format!("{prefix}.w1.weight")).clone(),
            w3: w.get(&format!("{prefix}.w3.weight")).clone(),
            w2: w.get(&format!("{prefix}.w2.weight")).clone(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h1 = linear(x, &self.w1, None)?.contiguous()?;
        let h3 = linear(x, &self.w3, None)?.contiguous()?;
        let gated = candle_nn::ops::silu(&h1)?.broadcast_mul(&h3)?;
        if std::env::var("KRONOS_TRACE").is_ok() {
            let g0 = gated
                .narrow(0, 0, 1)?
                .narrow(1, 0, 1)?
                .flatten_all()?
                .to_dtype(candle_core::DType::F64)?
                .to_vec1::<f64>()?;
            eprintln!("    [ffn] gated[0,0,:5]={:?}", &g0[..5.min(g0.len())]);
            let out0 = linear(&gated, &self.w2, None)?
                .narrow(0, 0, 1)?
                .narrow(1, 0, 1)?
                .flatten_all()?
                .to_dtype(candle_core::DType::F64)?
                .to_vec1::<f64>()?;
            eprintln!("    [ffn] out[0,0,:5]={:?}", &out0[..5.min(out0.len())]);
        }
        linear(&gated, &self.w2, None)
    }
}

/// `MultiHeadAttentionWithRoPE` — `model/module.py:315-353`.
///
/// Self-attention is always `is_causal=True` at inference (module.py:349) and
/// the predictor never supplies a `key_padding_mask`, so there is no mask path.
pub struct SelfAttention {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    qb: Tensor,
    kb: Tensor,
    vb: Tensor,
    ob: Tensor,
    n_heads: usize,
    head_dim: usize,
    d_model: usize,
    rope: RotaryCache,
}

impl SelfAttention {
    fn load(prefix: &str, w: &Weights, geo: BlockGeometry, rope: RotaryCache) -> Result<Self> {
        let g = |s: &str| w.get(&format!("{prefix}.{s}")).clone();
        Ok(Self {
            q: g("q_proj.weight"),
            k: g("k_proj.weight"),
            v: g("v_proj.weight"),
            o: g("out_proj.weight"),
            qb: g("q_proj.bias"),
            kb: g("k_proj.bias"),
            vb: g("v_proj.bias"),
            ob: g("out_proj.bias"),
            n_heads: geo.n_heads,
            head_dim: geo.head_dim,
            d_model: geo.d_model,
            rope,
        })
    }

    /// `[B, L, d] -> [B, H, L, hd]`, i.e. `.view(B,L,H,hd).transpose(1,2)`.
    fn split_heads(&self, x: &Tensor) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        x.reshape((b, l, self.n_heads, self.head_dim))?
            .permute((0, 2, 1, 3))
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let q = self.split_heads(&linear(x, &self.q, Some(&self.qb))?.contiguous()?)?;
        let k = self.split_heads(&linear(x, &self.k, Some(&self.kb))?.contiguous()?)?;
        let v = self.split_heads(&linear(x, &self.v, Some(&self.vb))?.contiguous()?)?;
        let (q, k) = self.rope.apply(&q, &k)?;
        let ctx = sdpa(&q, &k, &v, true)?;
        // back to [B, L, d]
        let ctx = ctx
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((b, l, self.d_model))?;
        linear(&ctx, &self.o, Some(&self.ob))
    }
}

/// `TransformerBlock` (pre-LN, residual) — `model/module.py:465-483`.
pub struct TransformerBlock {
    norm1: Tensor,
    attn: SelfAttention,
    norm2: Tensor,
    ffn: FeedForward,
    eps: f32,
}

impl TransformerBlock {
    pub fn load(prefix: &str, w: &Weights, geo: BlockGeometry, rope: RotaryCache) -> Result<Self> {
        Ok(Self {
            norm1: w.get(&format!("{prefix}.norm1.weight")).clone(),
            attn: SelfAttention::load(&format!("{prefix}.self_attn"), w, geo, rope)?,
            norm2: w.get(&format!("{prefix}.norm2.weight")).clone(),
            ffn: FeedForward::load(&format!("{prefix}.ffn"), w)?,
            eps: 1e-5,
        })
    }

    /// Compute dtype, taken from a weight (all block weights share it).
    pub fn dtype_of_norm1(&self) -> DType {
        self.norm1.dtype()
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let n1 = rms_norm(x, &self.norm1, self.eps)?;
        let h = x.broadcast_add(&self.attn.forward(&n1)?)?;
        let n2 = rms_norm(&h, &self.norm2, self.eps)?;
        h.broadcast_add(&self.ffn.forward(&n2)?)
    }
}

/// `MultiHeadCrossAttentionWithRoPE` — `model/module.py:356-397`.
///
/// Head count is **fixed at 4** (module.py:447) regardless of the predictor's
/// own `n_heads`, so Kronos-base's cross-attention uses head_dim = 832/4 = 208,
/// not its 16-way 52. `is_causal` is `self.training` (module.py:387), i.e.
/// **False at inference**: this is a full bidirectional cross-attention, and the
/// single-query form used by `decode_s2` attends to the whole window.
pub struct CrossAttention {
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    qb: Tensor,
    kb: Tensor,
    vb: Tensor,
    ob: Tensor,
    n_heads: usize,
    head_dim: usize,
    rope: RotaryCache,
}

impl CrossAttention {
    pub fn load(
        prefix: &str,
        w: &Weights,
        d_model: usize,
        n_heads: usize,
        max_len: usize,
        dev: &candle_core::Device,
        dtype: DType,
    ) -> Result<Self> {
        let g = |s: &str| w.get(&format!("{prefix}.{s}")).clone();
        let head_dim = d_model / n_heads;
        Ok(Self {
            q: g("q_proj.weight"),
            k: g("k_proj.weight"),
            v: g("v_proj.weight"),
            o: g("out_proj.weight"),
            qb: g("q_proj.bias"),
            kb: g("k_proj.bias"),
            vb: g("v_proj.bias"),
            ob: g("out_proj.bias"),
            n_heads,
            head_dim,
            rope: RotaryCache::new(head_dim, max_len, dev, dtype)?,
        })
    }

    fn split(&self, x: &Tensor) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        x.reshape((b, l, self.n_heads, self.head_dim))?
            .permute((0, 2, 1, 3))
    }

    /// `query` is `[B, Lq, d]`, `key`/`value` are `[B, Lk, d]`; `Lq` may differ.
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        causal: bool,
    ) -> Result<Tensor> {
        let (b, lq, d) = query.dims3()?;
        let q = self.split(&linear(query, &self.q, Some(&self.qb))?.contiguous()?)?;
        let k = self.split(&linear(key, &self.k, Some(&self.kb))?.contiguous()?)?;
        let v = self.split(&linear(value, &self.v, Some(&self.vb))?.contiguous()?)?;
        // RoPE positions run over the query length here (module.py:304 uses
        // `q.shape[-2]`), so a 1-token query gets position 0.
        let (q, k) = self.rope.apply(&q, &k)?;
        let ctx = sdpa(&q, &k, &v, causal)?;
        let ctx = ctx
            .permute((0, 2, 1, 3))?
            .contiguous()?
            .reshape((b, lq, d))?;
        linear(&ctx, &self.o, Some(&self.ob))
    }
}

/// `DependencyAwareLayer` — `model/module.py:446-462`.
///
/// `RMSNorm(hidden_states + CrossAttn(query=sibling, key=hidden, value=hidden))`.
pub struct DependencyAwareLayer {
    attn: CrossAttention,
    norm: Tensor,
    eps: f32,
}

impl DependencyAwareLayer {
    pub fn load(
        w: &Weights,
        cfg: &PredictorConfig,
        max_len: usize,
        dev: &candle_core::Device,
        dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            attn: CrossAttention::load(
                "dep_layer.cross_attn",
                w,
                cfg.d_model,
                cfg.dep_heads(),
                max_len,
                dev,
                dtype,
            )?,
            norm: w.get("dep_layer.norm.weight").clone(),
            eps: 1e-5,
        })
    }

    pub fn forward(&self, hidden: &Tensor, sibling_embed: &Tensor) -> Result<Tensor> {
        let attn = self.attn.forward(sibling_embed, hidden, hidden, false)?;
        rms_norm(&hidden.broadcast_add(&attn)?, &self.norm, self.eps)
    }
}
