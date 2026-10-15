use tch::Device;

#[allow(dead_code)]
fn m_device(dev: &tch::Device) -> tch::Device {
    *dev
}

use kronos_core::ops::linear;

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

/// `linear(x, W, b)` must be `x @ W^T + b` with W stored [out, in], using the
/// real first row of the probe input and an identity-plus-mean weight matrix.
#[test]
fn linear_matches_numpy() {
    let dev = Device::Cpu;
    let x0 = [1.9819049f32, 1.9894122, 1.9993943];
    let x = vec_to_tensor(x0.as_ref(), m_device(&dev), &[1, 1, 3]);
    // rows: identity x3, then [1,1,1] (sum), then [1/3,1/3,1/3] (mean), then zero row
    let w = vec_to_tensor(
        &[
            1.0f32,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0 / 3.0,
            1.0 / 3.0,
            1.0 / 3.0,
            0.0,
            0.0,
            0.0,
        ],
        m_device(&dev),
        &[6, 3],
    );
    let b = vec_to_tensor(&[0.0f32, 0.0, 0.0, 0.0, 0.0, 7.5], m_device(&dev), &[6]);

    let y = linear(&x, &w, Some(&b)).unwrap();
    let v = tensor_to_vecf32(&y.reshape([-1]));

    let sum = x0[0] + x0[1] + x0[2];
    let mean = sum / 3.0;
    let exp = [x0[0], x0[1], x0[2], sum, mean, 7.5];
    for i in 0..6 {
        assert!(
            (v[i] - exp[i]).abs() < 1e-4,
            "y[{i}]={} exp={}",
            v[i],
            exp[i]
        );
    }
    println!("linear OK: {:?}", &v[..6]);
}
