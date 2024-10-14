use nalgebra::SVector;
use std::fmt::Write;

pub fn svector_to_array<const N: usize>(v: SVector<f64, N>) -> [f64; N] {
    v.as_slice()
        .try_into()
        .expect("Failed to convert slice to array.")
}

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

pub fn make_zeroed_matrix<T: bytemuck::Zeroable>() -> Box<T> {
    use std::alloc::Layout;
    let layout = Layout::new::<T>();
    unsafe {
        let ptr = std::alloc::alloc_zeroed(layout);
        assert_ne!(ptr, std::ptr::null_mut(), "Failed to allocate memory");
        // Since we asked for zeroed memory, it is valid matrix.
        Box::from_raw(ptr as _)
    }
}

#[cfg(test)]
mod test {
    use nalgebra::{vector, SVector};

    use super::svector_to_array;

    #[test]
    fn convert_nonempty_svector_to_array() {
        let v = SVector::<f64, 20>::from_fn(|_, _| 0.5);
        let arr = svector_to_array(v);

        assert!(
            v.iter().zip(arr.iter()).all(|(a, b)| a == b),
            "Not all elements are equal?"
        )
    }

    #[test]
    fn convert_single_element_svector_to_array() {
        let v = vector![0.5];
        let arr = svector_to_array(v);

        assert!(
            v.iter().zip(arr.iter()).all(|(a, b)| a == b),
            "Not all elements are equal?"
        )
    }

    #[test]
    fn convert_empty_svector_to_array() {
        let v = vector![];
        let arr = svector_to_array(v);
        assert!(arr.is_empty(), "Not zero?");
    }
}
