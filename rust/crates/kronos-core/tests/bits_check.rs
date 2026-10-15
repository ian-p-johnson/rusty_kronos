use tch::Device;

#[allow(dead_code)]
fn m_device(dev: &tch::Device) -> tch::Device {
    *dev
}

use kronos_core::ops::{bits_to_indices, indices_to_bits};

// ---- tch test helpers (candle-helper compatibility) ----
#[allow(dead_code)]
fn vec_to_tensor<T: tch::kind::Element>(v: &[T], dev: tch::Device, shape: &[i64]) -> tch::Tensor {
    tch::Tensor::from_slice(v).reshape(shape).to_device(dev)
}
#[allow(dead_code)]
fn tensor_to_vecf32(t: &tch::Tensor) -> Vec<f32> {
    let h = t
        .reshape([-1])
        .to_device(tch::Device::Cpu)
        .to_kind(tch::Kind::Float);
    (&h).try_into().unwrap()
}
#[allow(dead_code)]
fn tensor_to_vecf64(t: &tch::Tensor) -> Vec<f64> {
    let h = t
        .reshape([-1])
        .to_device(tch::Device::Cpu)
        .to_kind(tch::Kind::Double);
    (&h).try_into().unwrap()
}
#[allow(dead_code)]
fn tensor_to_veci64(t: &tch::Tensor) -> Vec<i64> {
    let h = t
        .reshape([-1])
        .to_device(tch::Device::Cpu)
        .to_kind(tch::Kind::Int64);
    (&h).try_into().unwrap()
}
#[allow(dead_code)]
fn tensor_to_veci32(t: &tch::Tensor) -> Vec<i32> {
    let h = t
        .reshape([-1])
        .to_device(tch::Device::Cpu)
        .to_kind(tch::Kind::Int);
    (&h).try_into().unwrap()
}
#[allow(dead_code)]
fn tensor_to_vec2f32(t: &tch::Tensor) -> Vec<Vec<f32>> {
    let h = t.to_device(tch::Device::Cpu).to_kind(tch::Kind::Float);
    (&h).try_into().unwrap()
}
#[allow(dead_code)]
fn tensor_to_vec2i32(t: &tch::Tensor) -> Vec<Vec<i32>> {
    let h = t.to_device(tch::Device::Cpu).to_kind(tch::Kind::Int);
    (&h).try_into().unwrap()
}

/// Reproduces `indices_to_bits` for the real first-6 subtokens and checks it
/// against the same expression evaluated in numpy ((idx >> i) & 1, bipolar, scaled).
#[test]
fn indices_to_bits_matches_python() {
    let dev = Device::Cpu;
    let s1: &[i64] = &[294, 294, 294, 32, 64, 290];
    let s2: &[i64] = &[864, 872, 864, 865, 865, 864];
    let t1 = vec_to_tensor(s1, m_device(&dev), &[1, 6]);
    let t2 = vec_to_tensor(s2, m_device(&dev), &[1, 6]);
    let q = 1.0f32 / (20f32).sqrt();

    let bits = indices_to_bits(&t1, &t2, 20, q as f64).unwrap();
    let v = tensor_to_vecf32(&bits.reshape([-1]));

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
    let t = vec_to_tensor(
        &(0..n).map(|i| i as i64).collect::<Vec<_>>(),
        m_device(&dev),
        &[1, n as i64],
    );
    let bits = indices_to_bits(&t, &t, 20, 1.0).unwrap();
    let back1 = bits_to_indices(&bits.narrow(2, 0, 10)).unwrap();
    let back2 = bits_to_indices(&bits.narrow(2, 10, 10)).unwrap();
    let orig = &tensor_to_vec2i32(&t)[0];
    let got1 = tensor_to_vecf64(&back1);
    let got2 = tensor_to_vecf64(&back2);
    for i in 0..n {
        assert_eq!(orig[i] as f64, got1[i], "s1 index {i}");
        assert_eq!(orig[i] as f64, got2[i], "s2 index {i}");
    }
    println!("full-codebook round-trip OK over {n} values");
}
