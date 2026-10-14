//! K-line CSV loading and calendar features.
//!
//! The goldens are driven end-to-end from the raw fixture, so this must parse
//! `tests/data/regression_input.csv` and derive the five calendar features
//! (`minute/hour/weekday/day/month`) exactly as `calc_time_stamps`
//! (`model/kronos.py:472-479`) does, with **weekday = Monday 0 .. Sunday 6**
//! (`x_timestamp.dt.weekday`).
//!
//! Deliberately dependency-free: the file is machine-written with a fixed
//! schema, so a hand-rolled parser is cheap and keeps the port standalone.

use candle_core::{Device, Tensor};

/// One OHLCVA row with its wall-clock timestamp.
#[derive(Debug, Clone, Copy)]
pub struct Kline {
    /// year, month(1-12), day, hour, minute
    pub y: i32,
    pub mo: i32,
    pub d: i32,
    pub h: i32,
    pub mi: i32,
    pub v: [f32; 6], // open, high, low, close, volume, amount
}

#[derive(Debug, thiserror::Error)]
pub enum DataError {
    #[error("CSV `{src}` line {line}: {msg}")]
    CsvLine {
        src: String,
        line: usize,
        msg: String,
    },
    #[error("CSV `{0}`: unexpected column count/header")]
    Schema(String),
}

/// Howard Hinnant's `days_from_civil`: days since 1970-01-01, proleptic Gregorian.
pub fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// `pd.Series.dt.weekday` convention: Monday 0 .. Sunday 6.
///
/// 1970-01-01 was a Thursday (= 3), so `weekday = (days + 3) % 7` with the
/// result normalised into range.
pub fn weekday_of(y: i64, m: i64, d: i64) -> u32 {
    let days = days_from_civil(y, m, d);
    (((days % 7) + 7 + 3) % 7) as u32
}

/// Parse `YYYY-MM-DD HH:MM(:SS[.fraction])`; the seconds part is ignored
/// because `calc_time_stamps` never reads it.
fn parse_ts(
    s: &str,
    src: &str,
    line: usize,
) -> std::result::Result<(i32, i32, i32, i32, i32), DataError> {
    let _bad = |msg: &str| DataError::CsvLine {
        src: src.to_string(),
        line,
        msg: msg.to_string(),
    };
    let (date, time) = s.split_once(' ').ok_or_else(|| {
        DataError::Schema("timestamp must be `YYYY-MM-DD HH:MM[:SS]`".to_string())
    })?;
    let mut di = date.split('-');
    let (y, mo, d) = match (di.next(), di.next(), di.next(), di.next()) {
        (Some(y), Some(mo), Some(d), None) => (y, mo, d),
        _ => return Err(DataError::Schema("bad date part".to_string())),
    };
    let mut ti = time.split(':');
    let (h, mi) = match (ti.next(), ti.next(), ti.next()) {
        (Some(h), Some(m), None) | (Some(h), Some(m), Some(_)) => (h, m),
        _ => return Err(DataError::Schema("bad time part".to_string())),
    };
    let num = |v: &str, what: &str| -> Result<i32, DataError> {
        v.trim()
            .parse::<i32>()
            .map_err(|_| DataError::Schema(format!("bad {what} `{v}` in timestamp `{s}`")))
    };
    Ok((
        num(y, "year")?,
        num(mo, "month")?,
        num(d, "day")?,
        num(h, "hour")?,
        num(mi, "minute")?,
    ))
}

/// Load the K-line schema. Returns `(calendar rows, values)`.
pub fn load_csv(path: &std::path::Path) -> std::result::Result<Vec<Kline>, DataError> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| DataError::Schema(format!("{}: {e}", path.display())))?;
    let src = path.display().to_string();
    let mut lines = text.lines().enumerate();
    let (_i, header) = lines.next().ok_or_else(|| DataError::Schema(src.clone()))?;
    // accept any header ordering, but require the six value columns by name
    let cols: Vec<&str> = header.split(',').map(|c| c.trim()).collect();
    let want = ["open", "high", "low", "close", "volume", "amount"];
    if !want.iter().all(|w| cols.contains(w)) {
        return Err(DataError::Schema(src));
    }
    let mut out = Vec::new();
    for (i, line) in lines {
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(',').map(|c| c.trim()).collect();
        let ts = parse_ts(f.first().copied().unwrap_or(""), &src, i + 1)?;
        let lookup = |name: &str| -> Option<f32> {
            let j = cols.iter().position(|c| *c == name)?;
            f.get(j)?.parse::<f32>().ok()
        };
        let mut v = [0.0f32; 6];
        for (k, w) in WANT.iter().enumerate() {
            v[k] = lookup(w).ok_or_else(|| DataError::CsvLine {
                src: src.clone(),
                line: i + 1,
                msg: format!("bad or missing number in `{w}`"),
            })?;
        }
        out.push(Kline {
            y: ts.0,
            mo: ts.1,
            d: ts.2,
            h: ts.3,
            mi: ts.4,
            v,
        });
    }
    Ok(out)
}

/// The six value columns in the order the model consumes them.
pub const WANT: [&str; 6] = ["open", "high", "low", "close", "volume", "amount"];

/// `[rows, 5]` F32 of `minute, hour, weekday, day, month`.
pub fn stamp_tensor(
    ys: &[i32],
    mos: &[i32],
    ds: &[i32],
    hs: &[i32],
    mins: &[i32],
    dev: &Device,
) -> candle_core::Result<Tensor> {
    let n = ys.len();
    let v: Vec<f32> = (0..n)
        .flat_map(|i| {
            [
                mins[i] as f32,
                hs[i] as f32,
                weekday_of(ys[i] as i64, mos[i] as i64, ds[i] as i64) as f32,
                ds[i] as f32,
                mos[i] as f32,
            ]
        })
        .collect();
    Tensor::from_vec(v, (n, 5), dev)
}

/// Convenience: stamps for a slice of K-lines (a contiguous view of the file).
pub fn stamps(rows: &[Kline], dev: &Device) -> candle_core::Result<Tensor> {
    stamp_tensor(
        &rows.iter().map(|r| r.y).collect::<Vec<_>>(),
        &rows.iter().map(|r| r.mo).collect::<Vec<_>>(),
        &rows.iter().map(|r| r.d).collect::<Vec<_>>(),
        &rows.iter().map(|r| r.h).collect::<Vec<_>>(),
        &rows.iter().map(|r| r.mi).collect::<Vec<_>>(),
        dev,
    )
}

/// `[cols]` values for a contiguous slice, as an F32 tensor.
pub fn values(rows: &[Kline], dev: &Device) -> candle_core::Result<Tensor> {
    let n = rows.len();
    let mut v = Vec::with_capacity(n * 6);
    for r in rows {
        v.extend_from_slice(&r.v);
    }
    Tensor::from_vec(v, (n, 6), dev)?.contiguous()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weekday_matches_known_dates() {
        // 1970-01-01 Thursday(3) / 2024-07-03 Wednesday(2) / 2000-01-01 Saturday(5)
        assert_eq!(weekday_of(1970, 1, 1), 3);
        assert_eq!(weekday_of(2024, 7, 3), 2);
        assert_eq!(weekday_of(2000, 1, 1), 5);
        assert_eq!(weekday_of(2024, 2, 29), 3); // leap day, Thursday (verbatim from datetime.date)
        assert_eq!(weekday_of(2049, 12, 31), 4); // Friday (datetime.date(2049,12,31))
    }

    #[test]
    fn parses_the_fixture_timestamp() {
        let (y, mo, d, h, mi) = parse_ts("2024-07-03 09:55:00", "t", 0).expect("parses");
        assert_eq!((y, mo, d, h, mi), (2024, 7, 3, 9, 55));
        let (y, mo, d, h, mi) = parse_ts("1970-01-01 00:00", "t", 0).expect("no seconds");
        assert_eq!((y, mo, d, h, mi), (1970, 1, 1, 0, 0));
        assert!(parse_ts("bad", "t", 0).is_err());
    }
}
