use nalgebra::SVector;
use std::fmt::Write;

pub fn array_distance<const N: usize>(a1: &[f64; N], a2: &[f64; N]) -> f64 {
    let v1: SVector<f64, N> = unsafe {
        let a1_ptr = a1.as_ptr();
        SVector::from_column_slice(std::slice::from_raw_parts(a1_ptr, N))
    };

    let v2: SVector<f64, N> = unsafe {
        let a2_ptr = a2.as_ptr();
        SVector::from_column_slice(std::slice::from_raw_parts(a2_ptr, N))
    };

    (v2 - v1).norm()
}

pub fn vector_to_string<const N: usize>(v: &SVector<f64, N>) -> String {
    let mut result = String::new();
    write!(result, "[").unwrap();

    for i in 0..N {
        if i > 0 {
            write!(result, ", ").unwrap();
        }
        write!(result, "{}", v[i]).unwrap();
    }

    write!(result, "]").unwrap();
    result
}
