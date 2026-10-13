use candle_core::{Device, Tensor};
use kronos_core::ops::linear;

/// `linear(x, W, b)` must be `x @ W^T + b` with W stored [out, in], using the
/// real first row of the probe input and an identity-plus-mean weight matrix.
#[test]
fn linear_matches_numpy() {
    let dev = Device::Cpu;
    let x0 = [1.9819049f32, 1.9894122, 1.9993943];
    let x = Tensor::from_vec(x0.to_vec(), (1, 1, 3), &dev).unwrap();
    // rows: identity x3, then [1,1,1] (sum), then [1/3,1/3,1/3] (mean), then zero row
    let w = Tensor::from_vec(
        vec![
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
        (6, 3),
        &dev,
    )
    .unwrap();
    let b = Tensor::from_vec(vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 7.5], (6,), &dev).unwrap();

    let y = linear(&x, &w, Some(&b)).unwrap();
    let v = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();

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
