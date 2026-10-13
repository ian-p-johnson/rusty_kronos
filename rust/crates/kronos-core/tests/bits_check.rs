use candle_core::{Device, Tensor};
use kronos_core::ops::{bits_to_indices, indices_to_bits};

/// Reproduces `indices_to_bits` for the real first-6 subtokens and checks it
/// against the same expression evaluated in numpy ((idx >> i) & 1, bipolar, scaled).
#[test]
fn indices_to_bits_matches_python() {
    let dev = Device::Cpu;
    let s1: &[u32] = &[294, 294, 294, 32, 64, 290];
    let s2: &[u32] = &[864, 872, 864, 865, 865, 864];
    let t1 = Tensor::from_vec(s1.to_vec(), (1, 6), &dev).unwrap();
    let t2 = Tensor::from_vec(s2.to_vec(), (1, 6), &dev).unwrap();
    let q = 1.0f32 / (20f32).sqrt();

    let bits = indices_to_bits(&t1, &t2, 20, q as f64).unwrap();
    let v = bits.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    let expect: Vec<f32> = (0..6)
        .flat_map(|r| {
            let (a, b) = (s1[r], s2[r]);
            (0..20).map(move |i| {
                let src = if i < 10 { a } else { b };
                let sh = if i < 10 { i } else { i - 10 };
                (((src >> sh) & 1) as f32 * 2.0 - 1.0) * q
            })
        })
        .collect();
    for i in 0..6 * 20 {
        assert!(
            (v[i] - expect[i]).abs() < 1e-6,
            "bits[{i}]={} expected {}",
            v[i],
            expect[i]
        );
    }
    println!("indices_to_bits row0[:8] = {:?}", &v[..8]);
}

/// The two directions must be inverses over the whole released code space.
#[test]
fn bits_indices_round_trip_again() {
    let dev = Device::Cpu;
    let n = 1usize << 10;
    let t = Tensor::from_vec((0..n).map(|i| i as u32).collect::<Vec<_>>(), (1, n), &dev).unwrap();
    let bits = indices_to_bits(&t, &t, 20, 1.0).unwrap();
    let back1 = bits_to_indices(&bits.narrow(2, 0, 10).unwrap()).unwrap();
    let back2 = bits_to_indices(&bits.narrow(2, 10, 10).unwrap()).unwrap();
    let orig = &t.to_vec2::<u32>().unwrap()[0];
    let got1 = &back1.to_vec2::<f64>().unwrap()[0];
    let got2 = &back2.to_vec2::<f64>().unwrap()[0];
    for i in 0..n {
        assert_eq!(orig[i] as f64, got1[i], "s1 index {i}");
        assert_eq!(orig[i] as f64, got2[i], "s2 index {i}");
    }
    println!("full-codebook round-trip OK over {n} values");
}
