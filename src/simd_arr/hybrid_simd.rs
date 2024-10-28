use std::ops::{Index, IndexMut};

use super::{dense_simd::DenseSimd, sparse_simd::VecSparseSimd, SimdArr};

#[derive(Clone, Debug)]
pub enum HybridSimd<const SIZE: usize, const CRITIALITY: usize> {
    Dense(Box<DenseSimd<SIZE>>),
    Sparse(VecSparseSimd<CRITIALITY, SIZE>),
}

impl<const S: usize, const C: usize> SimdArr<S> for HybridSimd<S, C> {
    fn new_from_array(arr: [f32; S]) -> Self {
        match VecSparseSimd::new_from_array(&arr) {
            None => HybridSimd::Dense(Box::new(DenseSimd::new_from_array(arr))),
            Some(sparse) => HybridSimd::Sparse(sparse),
        }
    }

    fn check_nan(&self) {
        match self {
            HybridSimd::Dense(d) => d.check_nan(),
            HybridSimd::Sparse(s) => s.check_nan(),
        }
    }

    fn zero() -> Self {
        Self::Sparse(VecSparseSimd::zero())
    }

    fn to_array(&self) -> [f32; S] {
        match self {
            HybridSimd::Dense(d) => d.to_array(),
            HybridSimd::Sparse(s) => s.to_array(),
        }
    }

    fn neg(&mut self) {
        match self {
            HybridSimd::Dense(d) => {
                d.neg();
            }
            HybridSimd::Sparse(s) => {
                s.neg();
            }
        }
    }

    fn new_from_value_and_pos(val: f32, pos: usize) -> Self {
        HybridSimd::Sparse(VecSparseSimd::new_from_value_and_pos(val, pos))
    }

    fn acumulate(&mut self, rhs: &Self) {
        match (self, rhs) {
            (HybridSimd::Dense(a), HybridSimd::Dense(b)) => a.acumulate(b),
            (HybridSimd::Dense(a), HybridSimd::Sparse(b)) => {
                let transformation = DenseSimd::new_from_array(b.to_array());
                a.acumulate(&transformation);
            }
            (res @ HybridSimd::Sparse(_), HybridSimd::Dense(b)) => {
                let mut transformation = DenseSimd::new_from_array(res.to_array());
                transformation.acumulate(b);

                *res = HybridSimd::Dense(Box::new(transformation));
            }
            (res @ HybridSimd::Sparse(_), HybridSimd::Sparse(b)) => {
                let success = res.unwrap_sparse().acumulate(b);
                if success.is_err() {
                    let mut transformation_self = DenseSimd::new_from_array(res.to_array());
                    let transformation_rhs = DenseSimd::new_from_array(b.to_array());
                    transformation_self.acumulate(&transformation_rhs);
                    *res = HybridSimd::Dense(Box::new(transformation_self))
                }
            }
        }
    }

    fn multiply(&mut self, rhs: f32) {
        match self {
            HybridSimd::Dense(d) => d.multiply(rhs),
            HybridSimd::Sparse(s) => s.multiply(rhs),
        }
    }
}

impl<const S: usize, const C: usize> HybridSimd<S, C> {
    fn unwrap_sparse<'a>(&'a mut self) -> &'a mut VecSparseSimd<C, S> {
        if let Self::Sparse(x) = self {
            x
        } else {
            panic!();
        }
    }
}

impl<const S: usize, const C: usize> Index<usize> for HybridSimd<S, C> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            HybridSimd::Dense(d) => &d[index],
            HybridSimd::Sparse(s) => &s[index],
        }
    }
}

impl<const S: usize, const C: usize> IndexMut<usize> for HybridSimd<S, C> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self {
            HybridSimd::Dense(d) => &mut d[index],
            HybridSimd::Sparse(s) => &mut s[index],
        }
    }
}

#[cfg(test)]
mod hybrid_simd_tests {
    use rand::{self, seq::SliceRandom, Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::array::from_fn;

    use crate::simd_arr::SimdArr;

    use super::HybridSimd;

    fn test_add<const N: usize>(a: [f32; N], b: [f32; N]) {
        let res_vec = a
            .iter()
            .zip(b)
            .map(|(a_elm, b_elm)| a_elm + b_elm)
            .collect::<Vec<_>>();
        let res: [f32; N] = from_fn(|i| res_vec[i]);

        let mut x = HybridSimd::<N, N>::new_from_array(a);
        let y = HybridSimd::<N, N>::new_from_array(b);

        x.acumulate(&y);

        assert_eq!(x.to_array(), res)
    }

    fn test_sub<const N: usize>(a: [f32; N], b: [f32; N]) {
        let res_vec = a
            .iter()
            .zip(b)
            .map(|(a_elm, b_elm)| a_elm - b_elm)
            .collect::<Vec<_>>();
        let res: [f32; N] = from_fn(|i| res_vec[i]);

        let mut x = HybridSimd::<N, N>::new_from_array(a);
        let mut y = HybridSimd::<N, N>::new_from_array(b);
        y.neg();

        x.acumulate(&y);

        assert_eq!(x.to_array(), res)
    }

    #[test]
    fn arithmetic_stress_test() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);

        for _ in 0..1000 {
            let cero_ratio: f32 = rng.gen();
            let a: [f32; 32] = rand::random::<[f32; 32]>().map(|x| (x - cero_ratio).max(0.));
            let b: [f32; 32] = rand::random::<[f32; 32]>().map(|x| (x - cero_ratio).max(0.));

            test_add(a, b);
            test_sub(a, b);
        }
    }

    #[test]
    fn consistency_stress_test() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);

        for _ in 0..1000 {
            let cero_ratio: f32 = rng.gen();
            let a: [f32; 32] = rand::random::<[f32; 32]>().map(|x| (x - cero_ratio).max(0.));

            assert_eq!(a, HybridSimd::<32, 32>::new_from_array(a).to_array())
        }
    }

    fn test_mul_scalar<const N: usize>(a: [f32; N], b: f32) {
        let res = a.map(|a_elm| a_elm * b);

        let mut test = HybridSimd::<N, N>::new_from_array(a);
        test.multiply(b);

        assert_eq!(test.to_array(), res)
    }

    fn test_div_scalar<const N: usize>(a: [f32; N], b: f32) {
        let res = a.map(|a_elm| a_elm * (1. / b)); // good enough

        let mut test = HybridSimd::<N, N>::new_from_array(a);
        test.multiply(1. / b);

        assert_eq!(test.to_array(), res)
    }

    #[test]
    fn stress_test_scalar() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);

        for _ in 0..1000 {
            let cero_ratio: f32 = rng.gen();
            let a: [f32; 32] = rand::random::<[f32; 32]>().map(|x| (x - cero_ratio).max(0.));
            let b = rng.gen();
            test_mul_scalar(a, b);
            test_div_scalar(a, b);
        }
    }

    #[test]
    fn test_indexing() {
        for i in 0..10 {
            let mut test_arr = [0.; 10];
            test_arr[i] = 1.;
            let x = HybridSimd::<10, 10>::new_from_array(test_arr);

            for j in 0..10 {
                if i == j {
                    assert_eq!(1., x[j]);
                    assert_eq!(1., x.to_array()[j]);
                } else {
                    assert_eq!(0., x[j]);
                    assert_eq!(0., x.to_array()[j]);
                }
            }
        }
    }

    #[test]
    fn stress_test_to_array() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        for _ in 0..1000 {
            let cero_ratio: f32 = rng.gen();
            let a: [f32; 4] = rand::random::<[f32; 4]>().map(|x| (x - cero_ratio).max(0.));

            let mut x: HybridSimd<4, 4> = HybridSimd::zero();
            let mut order = (0..4).collect::<Vec<_>>();
            order.shuffle(&mut rng);
            for i in order {
                x[i] = a[i];
                assert_eq!(x.to_array()[i], a[i]);
            }

            for i in 0..4 {
                assert_eq!(x.to_array()[i], a[i]);
            }
        }
    }

    #[test]
    fn stress_test_to_array_overriding() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        for _ in 0..1000 {
            let cero_ratio: f32 = rng.gen();
            let a: [f32; 4] = rand::random::<[f32; 4]>().map(|x| (x - cero_ratio).max(0.));
            let b: [f32; 4] = rand::random::<[f32; 4]>().map(|x| (x - cero_ratio).max(0.));

            let mut y = HybridSimd::<4, 4>::new_from_array(b);
            let mut order = (0..4).collect::<Vec<_>>();
            order.shuffle(&mut rng);
            for i in order {
                y[i] = a[i];
                assert_eq!(y.to_array()[i], a[i]);
            }

            for i in 0..4 {
                assert_eq!(y.to_array()[i], a[i]);
            }
        }
    }

    #[test]
    fn stress_test_indexing() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);
        for _ in 0..1000 {
            let cero_ratio: f32 = rng.gen();
            let a: [f32; 32] = rand::random::<[f32; 32]>().map(|x| (x - cero_ratio).max(0.));
            let b: [f32; 32] = rand::random::<[f32; 32]>().map(|x| (x - cero_ratio).max(0.));

            let mut x: HybridSimd<32, 32> = HybridSimd::zero();
            let mut y: HybridSimd<32, 32> = HybridSimd::new_from_array(b);
            let mut order = (0..32).collect::<Vec<_>>();
            order.shuffle(&mut rng);
            for i in order {
                x[i] = a[i];
                y[i] = a[i];
            }

            for i in 0..32 {
                assert_eq!(x[i], a[i]);
                assert_eq!(y[i], a[i]);
            }
        }
    }

    #[test]
    fn test_indexing_mut() {
        for i in 0..10 {
            let mut x = HybridSimd::<10, 10>::new_from_array([0.; 10]);
            x[i] = 1.;

            for j in 0..10 {
                if i == j {
                    assert_eq!(1., x[j]);
                } else {
                    assert_eq!(0., x[j]);
                }
            }
        }
    }

    #[test]
    fn test_optimistic_allocation() {
        let test = [1., 0., 0., 2., 0., 0., 1., 3., 0., 9.];
        let x = HybridSimd::<10, 10>::new_from_array(test);

        assert_eq!(x.to_array(), test);
    }
}
