//! Minimal `.npy` v1/v2 reader.
//!
//! We parse the header ourselves rather than pulling in the `ndarray` conversion
//! path: the fixtures are C-order little-endian float32 (or int64), the format
//! is trivial, and this keeps parity loading dependency-free.
//!
//! Layout: magic `\x93NUMPY` | major u8 | minor u8 | header_len u16 (v1) or
//! u32 (v2) | python dict header | raw C-order data.

use crate::ops::Result;
use std::path::Path;
use tch::{Kind, Tensor};

fn err(msg: impl std::fmt::Display) -> tch::TchError {
    tch::TchError::Convert(format!("{msg}"))
}

/// Extract `'shape': (1, 2, 3)` from the header dict.
fn parse_shape(header: &str) -> Result<Vec<usize>> {
    let key = "'shape'";
    let start = header
        .find(key)
        .ok_or_else(|| err("npy header has no 'shape'"))?
        + key.len();
    let rest = &header[start..];
    let open = rest
        .find('(')
        .ok_or_else(|| err("npy header shape is not a tuple"))?;
    let close = rest[open..]
        .find(')')
        .ok_or_else(|| err("npy header shape tuple unterminated"))?
        + open;
    rest[open + 1..close]
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| err(format!("bad npy dim {s:?}: {e}")))
        })
        .collect()
}

/// `'fortran_order': True` means column-major. We rescan it to row-major after
/// reading, because every consumer here wants C-order.
fn fortran_order(header: &str) -> bool {
    header.contains("'fortran_order': True") || header.contains("'fortran_order': true")
}

fn dtype_of(header: &str) -> Kind {
    if header.contains("'<f4'") || header.contains("\"<f4\"") {
        Kind::Float
    } else if header.contains("'<f8'") || header.contains("\"<f8\"") {
        Kind::Double
    } else if header.contains("'<i8'") || header.contains("\"<i8\"") {
        Kind::Int64
    } else if header.contains("'|u1'") {
        Kind::Uint8
    } else {
        Kind::Float
    }
}

/// Read a `.npy` file as a CPU tensor.
pub fn read_npy(path: &Path) -> Result<Tensor> {
    let bytes = std::fs::read(path).map_err(|e| err(format!("{}: {e}", path.display())))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(err(format!("{}: not a .npy file", path.display())));
    }
    let major = bytes[6];
    let (hlen, data_start) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10usize)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12usize,
        )
    };
    let header = std::str::from_utf8(&bytes[data_start..data_start + hlen])
        .map_err(|e| err(format!("{}: header not utf-8: {e}", path.display())))?;
    let shape = parse_shape(header)?;
    let dt = dtype_of(header);
    let data = &bytes[data_start + hlen..];
    // element count is the PRODUCT of dims. The earlier `.max(len())` treated
    // a (1,1) array as if it had 2 elements and rejected a valid file.
    let n: usize = shape.iter().product::<usize>();
    let elem = match dt {
        Kind::Float => 4,
        Kind::Double | Kind::Int64 => 8,
        Kind::Uint8 => 1,
        other => return Err(err(format!("unsupported npy dtype {other:?}"))),
    };
    if data.len() < n * elem {
        return Err(err(format!(
            "{}: truncated, want {} bytes have {}",
            path.display(),
            n * elem,
            data.len()
        )));
    }
    let mut shape = shape;
    if shape.is_empty() {
        shape = vec![1];
    }
    let mut fortran = fortran_order(header);
    // numpy collapses an all-ones/1-D array to C-order; only reshape matters
    if fortran && shape.len() == 1 {
        fortran = false;
    }
    let to_shape = if fortran {
        // interpret the buffer column-major, then present it row-major
        let mut rev = shape.clone();
        rev.reverse();
        rev
    } else {
        shape.clone()
    };
    match dt {
        Kind::Float => {
            let v: Vec<f32> = data[..n * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let t = Tensor::from_slice(&v)
                .reshape(to_shape.iter().map(|&d| d as i64).collect::<Vec<_>>());
            if fortran {
                // written column-major: present it row-major
                let perm: Vec<i64> = (0..shape.len())
                    .map(|i| (shape.len() - 1 - i) as i64)
                    .collect();
                Ok(t.f_permute(perm.as_slice())?.contiguous())
            } else {
                Ok(t)
            }
        }
        Kind::Int64 => {
            // '<i8' is 8 bytes per element; earlier code read 4 and widened the
            // low half, which doubled every index past 2^31 and silently
            // shifted the attribute axis.
            const ELEM: usize = 8;
            let v: Vec<i64> = data[..n * ELEM]
                .as_chunks::<ELEM>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();
            let t = Tensor::from_slice(&v)
                .reshape(to_shape.iter().map(|&d| d as i64).collect::<Vec<_>>());
            if fortran {
                let perm: Vec<i64> = (0..shape.len())
                    .map(|i| (shape.len() - 1 - i) as i64)
                    .collect();
                Ok(t.f_permute(perm.as_slice())?.contiguous())
            } else {
                Ok(t)
            }
        }
        other => Err(err(format!("dtype {other:?} not yet supported"))),
    }
}

/// Fixture filename for a stage name: `pred.norm` -> `pred__norm.npy`.
pub fn fixture_path(dir: &Path, name: &str) -> std::path::PathBuf {
    dir.join(format!("{}.npy", name.replace('.', "__")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_npy(path: &Path, shape: &[usize], data: &[f32]) {
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(b"\x93NUMPY\x01\x00").unwrap();
        let dict = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
            shape
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let mut pad = 64 - ((10 + dict.len() + 1) % 64);
        if pad == 64 {
            pad = 0;
        }
        let dict = format!("{dict}{}", " ".repeat(pad));
        let hlen = dict.len() as u16;
        f.write_all(&hlen.to_le_bytes()).unwrap();
        f.write_all(dict.as_bytes()).unwrap();
        for v in data {
            f.write_all(&f32::to_le_bytes(*v)).unwrap();
        }
    }

    #[test]
    fn round_trips_a_npy_file() {
        let p = std::env::temp_dir().join("kronos_npy_test.npy");
        write_npy(&p, &[2, 3], &[1.0, -2.0, 3.0, 4.0, -5.0, 6.0]);
        let t = read_npy(&p).unwrap();
        assert_eq!(t.size(), [2, 3]);
        let v: Vec<Vec<f32>> = (&t).try_into().unwrap();
        assert_eq!(v[0], vec![1.0, -2.0, 3.0]);
        assert_eq!(v[1], vec![4.0, -5.0, 6.0]);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn rejects_garbage() {
        let p = std::env::temp_dir().join("kronos_npy_bad.npy");
        std::fs::write(&p, b"not a numpy file at all").unwrap();
        assert!(read_npy(&p).is_err());
        let _ = std::fs::remove_file(&p);
    }
}
