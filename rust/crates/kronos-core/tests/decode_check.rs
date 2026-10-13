use candle_core::{DType, Device, Tensor};

fn tok_cfg() -> kronos_core::TokenizerConfig {
    let cache = std::env::var("KRONOS_HF_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/root".into()))
                .join(".cache/huggingface/hub")
        });
    for entry in
        std::fs::read_dir(cache.join("models--NeoQuasar--Kronos-Tokenizer-base/snapshots")).unwrap()
    {
        let p = entry.unwrap().path().join("config.json");
        if p.exists() {
            return kronos_core::TokenizerConfig::from_json(&std::fs::read_to_string(p).unwrap())
                .unwrap();
        }
    }
    panic!("Kronos-Tokenizer-base not found");
}

/// Checks the decode path stage-by-stage against numbers computed in numpy from
/// the same weights, using the reference's first-6 subtokens.
#[test]
fn decode_stage_post_quant_embed() -> Result<(), Box<dyn std::error::Error>> {
    let cfg = tok_cfg();
    if !cfg_ok(&cfg) {
        println!("SKIP: wrong config");
        return Ok(());
    }
    let dev = Device::Cpu;
    let w = find_weights("NeoQuasar--Kronos-Tokenizer-base")?;
    let dt = DType::F32;
    let weights = kronos_core::load_tokenizer_weights(&w, &cfg, dt, &dev)?;
    let tok = kronos_core::tokenizer::KronosTokenizer::load(&weights, &cfg, 512, &dev, dt)?;

    let s1 = Tensor::from_vec(vec![294u32, 294, 294, 32, 64, 290], (1, 6), &dev)?;
    let s2 = Tensor::from_vec(vec![864u32, 872, 864, 865, 865, 864], (1, 6), &dev)?;
    let bits = kronos_core::ops::indices_to_bits(&s1, &s2, 20, tok.q_scale())?;
    println!(
        "rust bits[0,:6] = {:?}",
        bits.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>()?[..6].to_vec()
    );

    let z = kronos_core::ops::linear(
        &bits,
        weights.get("post_quant_embed.weight"),
        Some(weights.get("post_quant_embed.bias")),
    )?;
    let v = z.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
    println!("rust post_quant_embed[0,:3] = {:?}", &v[..3]);
    // numpy reference: [-0.00346407, 0.25868233, 0.83103392]
    assert!((v[0] - (-0.00346407)).abs() < 1e-5, "got {}", v[0]);
    assert!((v[1] - 0.25868233).abs() < 1e-5, "got {}", v[1]);

    // head: fixture tok.head[0,0,:3] = [1.7694533, 1.8161650, 1.9354963]
    let mut zz = z;
    for blk in tok.decoder_ref() {
        zz = blk.forward(&zz)?;
    }
    let out = kronos_core::ops::linear(
        &zz,
        weights.get("head.weight"),
        Some(weights.get("head.bias")),
    )?;
    let o = out.narrow(1, 0, 1)?.flatten_all()?.to_vec1::<f32>()?;
    println!("rust head[0,:3] = {:?}", &o[..3]);
    assert!((o[0] - 1.7694533).abs() < 1e-4, "got {}", o[0]);
    Ok(())
}

fn cfg_ok(_c: &kronos_core::TokenizerConfig) -> bool {
    true
}

fn find_weights(repo_dir: &str) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let cache = std::env::var("KRONOS_HF_CACHE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/root".into()))
                .join(".cache/huggingface/hub")
        });
    let dir = cache.join(format!("models--{repo_dir}/snapshots"));
    for entry in std::fs::read_dir(&dir)? {
        let p = entry?.path().join("model.safetensors");
        if p.exists() {
            return Ok(p);
        }
    }
    Err(format!("no model.safetensors under {dir:?}").into())
}
