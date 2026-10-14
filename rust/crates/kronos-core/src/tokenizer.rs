//! `KronosTokenizer` — the continuous-to-discrete K-line tokenizer.
//!
//! Scope note: this covers the **inference** path (`encode` / `decode`). The
//! training-only quantiser losses (soft entropy, commitment) are deliberately
//! absent — see `PORTING_FROM_PYTHON.md` §4.2 and Phase 7.

use crate::block::TransformerBlock;
use crate::config::{BlockGeometry, TokenizerConfig};
use crate::ops::{
    bits_to_indices, bsq_sign, build_group_codebook, indices_to_bits, l2_normalize, RotaryCache,
};
use crate::weights::Weights;
use candle_core::{DType, Result, Tensor};
use std::collections::BTreeMap;

/// A pair of subtoken id tensors `[B, T]`, one for each half of the binary code.
pub type SubTokens = (Tensor, Tensor);

/// `KronosTokenizer` — `model/kronos.py:13-177`.
///
/// Pipeline: `embed(6 -> d)` -> 3 encoder blocks -> `quant_embed(d -> 20)` ->
/// BSQ -> `post_quant_embed(20 -> d)` -> 3 decoder blocks -> `head(d -> 6)`.
///
/// Note the released configs specify `n_enc_layers: 4` but the module builds
/// `n_enc_layers - 1` blocks (module.py:60-63), i.e. **3** on each side.
pub struct KronosTokenizer {
    cfg: TokenizerConfig,
    embed_w: Tensor,
    embed_b: Tensor,
    encoder: Vec<TransformerBlock>,
    quant_embed_w: Tensor,
    quant_embed_b: Tensor,
    post_quant_w: Tensor,
    post_quant_b: Tensor,
    decoder: Vec<TransformerBlock>,
    head_w: Tensor,
    head_b: Tensor,
    /// Rebuilt at runtime: `group_codebook` is registered `persistent=False`
    /// (module.py:78) so it exists in no checkpoint.
    group_codebook: Tensor,
    dtype: DType,
}

impl KronosTokenizer {
    pub fn load(
        w: &Weights,
        cfg: &TokenizerConfig,
        max_len: usize,
        dev: &candle_core::Device,
        dtype: DType,
    ) -> Result<Self> {
        // Note the tokenizer's OWN geometry (256/4/512) — not the predictor's.
        // Using the predictor's head count here would silently reshape attention
        // into a different partitioning and produce wrong output.
        let geo = BlockGeometry::for_tokenizer(cfg, max_len);
        let rope = RotaryCache::new(geo.head_dim, max_len, dev, dtype)?;
        let mut encoder = Vec::with_capacity(cfg.n_enc_blocks());
        for i in 0..cfg.n_enc_blocks() {
            encoder.push(TransformerBlock::load(
                &format!("encoder.{i}"),
                w,
                geo,
                rope.clone(),
            )?);
        }
        let mut decoder = Vec::with_capacity(cfg.n_dec_blocks());
        for i in 0..cfg.n_dec_blocks() {
            decoder.push(TransformerBlock::load(
                &format!("decoder.{i}"),
                w,
                geo,
                rope.clone(),
            )?);
        }
        Ok(Self {
            embed_w: w.get("embed.weight").clone(),
            embed_b: w.get("embed.bias").clone(),
            encoder,
            quant_embed_w: w.get("quant_embed.weight").clone(),
            quant_embed_b: w.get("quant_embed.bias").clone(),
            post_quant_w: w.get("post_quant_embed.weight").clone(),
            post_quant_b: w.get("post_quant_embed.bias").clone(),
            decoder,
            head_w: w.get("head.weight").clone(),
            head_b: w.get("head.bias").clone(),
            group_codebook: build_group_codebook(cfg.group_size, dev)?,
            dtype,
            cfg: cfg.clone(),
        })
    }

    /// Named intermediate stages, for the per-stage parity harness.
    ///
    /// Mirrors the forward-hook captures in `parity/gen_reference.py`, so the
    /// names here match the committed fixture filenames. Returns a map of
    /// stage name -> tensor, all cast to f64 for comparison.
    pub fn encode_with_stages(&self, x: &Tensor) -> Result<BTreeMap<String, Tensor>> {
        let mut stages = BTreeMap::new();
        let mut z = crate::ops::linear(x, &self.embed_w, Some(&self.embed_b))?;
        stages.insert("tok.embed".to_string(), z.to_dtype(DType::F64)?);
        for (i, blk) in self.encoder.iter().enumerate() {
            z = blk.forward(&z)?;
            stages.insert(format!("tok.encoder.{i}"), z.to_dtype(DType::F64)?);
        }
        let z = crate::ops::linear(&z, &self.quant_embed_w, Some(&self.quant_embed_b))?;
        stages.insert("tok.quant_embed".to_string(), z.to_dtype(DType::F64)?);
        let quantized = self.bsq(&z)?;
        stages.insert(
            "tok.tokenizer.bsq".to_string(),
            quantized.to_dtype(DType::F64)?,
        );
        Ok(stages)
    }

    /// `KronosTokenizer.encode` — `model/kronos.py:142-159`, `half=True`.
    ///
    /// Returns `(s1_ids, s2_ids)` as U32 in `[0, 2**10)`.
    pub fn encode(&self, x: &Tensor) -> Result<SubTokens> {
        let mut z = crate::ops::linear(x, &self.embed_w, Some(&self.embed_b))?;
        for blk in &self.encoder {
            z = blk.forward(&z)?;
        }
        let z = crate::ops::linear(&z, &self.quant_embed_w, Some(&self.quant_embed_b))?;
        let quantized = self.bsq(&z)?;
        self.split_half(&quantized)
    }

    /// `BSQuantizer.forward` with `collect_metrics=False` — `module.py:245-254`
    /// and `BinarySphericalQuantizer.forward` at `module.py:99-100`.
    ///
    /// `F.normalize` -> `sign` -> `* 1/sqrt(codebook_dim)`.
    ///
    /// With `collect_metrics=False` upstream returns immediately after the scale
    /// (`module.py:99-100`), so the entropy and commitment terms — the only
    /// training-only code in this file — never execute here.
    fn bsq(&self, z: &Tensor) -> Result<Tensor> {
        let normalized = l2_normalize(z)?;
        let signed = bsq_sign(&normalized)?;
        signed.affine(self.cfg.q_scale() as f64, 0.0)
    }

    /// Split the 20-bit code into its two 10-bit subtokens — `module.py:248-252`.
    ///
    /// `half=True` takes `quantized[..., :s1_bits]` and `quantized[..., s1_bits:]`,
    /// which is a slice on the **last** axis of a `[B, T, 20]` tensor.
    fn split_half(&self, quantized: &Tensor) -> Result<SubTokens> {
        let last = quantized.dims().len() - 1;
        let s1 = bits_to_indices(&quantized.narrow(last, 0, self.cfg.s1_bits as usize)?)?;
        let s2 = bits_to_indices(&quantized.narrow(
            last,
            self.cfg.s1_bits as usize,
            self.cfg.s2_bits as usize,
        )?)?;
        Ok((s1.to_dtype(DType::U32)?, s2.to_dtype(DType::U32)?))
    }

    /// `KronosTokenizer.decode` — `model/kronos.py:161-177`, `half=True`.
    pub fn decode(&self, tokens: &SubTokens) -> Result<Tensor> {
        let bits = indices_to_bits(
            &tokens.0.to_dtype(DType::F64)?,
            &tokens.1.to_dtype(DType::F64)?,
            self.cfg.codebook_dim(),
            self.cfg.q_scale() as f64,
        )?;
        let mut z = crate::ops::linear(
            &bits.to_dtype(self.dtype)?,
            &self.post_quant_w,
            Some(&self.post_quant_b),
        )?;
        for blk in &self.decoder {
            z = blk.forward(&z)?;
        }
        crate::ops::linear(&z, &self.head_w, Some(&self.head_b))
    }

    /// The compute dtype the weights were loaded in.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// `q_scale = 1/sqrt(codebook_dim)`, exposed for the parity harness.
    pub fn q_scale(&self) -> f64 {
        self.cfg.q_scale() as f64
    }

    /// Read-only access to the decoder blocks (parity diagnostics).
    pub fn decoder_ref(&self) -> &[TransformerBlock] {
        &self.decoder
    }

    /// The reconstructed `group_codebook`, exposed for parity reporting.
    pub fn group_codebook(&self) -> &Tensor {
        &self.group_codebook
    }
}
