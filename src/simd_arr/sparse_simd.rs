use std::ops::{Index, IndexMut};

#[derive(Clone, Debug, PartialEq)]
pub struct VecSparseSimd<const CAPACITY: usize, const VIRTUALSIZE: usize> {
    data_index: Vec<usize>,
    data: Vec<f32>,
}

impl<const CAPACITY: usize, const VIRTUALSIZE: usize> VecSparseSimd<CAPACITY, VIRTUALSIZE> {
    pub fn non_zero_count(&self) -> usize {
        self.data.len()
    }
}

impl<const CAPACITY: usize, const S: usize> VecSparseSimd<CAPACITY, S> {
    pub fn new_from_array(arr: &[f32; S]) -> Option<Self> {
        let mut ret = Self::zero();
        for (i, &elm) in arr.iter().enumerate() {
            if elm != 0. {
                if ret.data.len() == CAPACITY {
                    return None;
                }

                ret.data_index.push(i);
                ret.data.push(elm);
            }
        }
        Some(ret)
    }

    pub fn check_nan(&self) {
        // self.data.iter().for_each(|x| assert!(x.is_finite()));
    }

    pub fn zero() -> VecSparseSimd<CAPACITY, S> {
        Self {
            data_index: Vec::new(),
            data: Vec::new(),
        }
    }

    pub fn zero_with_capacity(capacity: usize) -> VecSparseSimd<CAPACITY, S> {
        // println!("creating sparse simd with {:?} as capacity", capacity);
        Self {
            data_index: Vec::with_capacity(capacity),
            data: Vec::with_capacity(capacity),
        }
    }

    pub fn to_array(&self) -> [f32; S] {
        let mut ret = [0.; S];

        for i in 0..self.data.len() {
            ret[self.data_index[i]] = self.data[i];
        }

        ret
    }

    pub fn new_from_value_and_pos(val: f32, pos: usize) -> Self {
        let mut ret = Self::zero();
        ret.data_index.push(pos);
        ret.data.push(val);
        ret
    }

    pub fn neg(&mut self) {
        for i in 0..self.data.len() {
            self.data[i] *= -1.;
        }
    }

    pub fn acumulate(&mut self, rhs: &Self) -> Result<(), ()> {
        if rhs.data.len() == 0 {
            Ok(())
        } else if self.data.len() == 0 {
            *self = rhs.clone();
            Ok(())
        } else {
            let mut ret = Self::zero_with_capacity(self.data.len() + rhs.data.len());

            let mut rhs_iter = rhs.data_index.iter().zip(rhs.data.iter()).peekable();

            for self_index in self.data_index.iter().zip(self.data.iter()) {
                let (self_idx, self_val) = self_index;

                while let Some(&(rhs_idx, rhs_val)) = rhs_iter.peek() {
                    if rhs_idx < self_idx {
                        if ret.data.len() == CAPACITY {
                            return Err(());
                        }
                        ret.data_index.push(*rhs_idx);
                        ret.data.push(*rhs_val);
                        rhs_iter.next();
                    } else {
                        break;
                    }
                }

                if let Some(&(rhs_idx, rhs_val)) = rhs_iter.peek() {
                    if *rhs_idx == *self_idx {
                        if ret.data.len() == CAPACITY {
                            return Err(());
                        }
                        ret.data_index.push(*rhs_idx);
                        ret.data.push(*self_val + rhs_val);
                        rhs_iter.next();
                    } else {
                        if ret.data.len() == CAPACITY {
                            return Err(());
                        }
                        ret.data_index.push(*self_idx);
                        ret.data.push(*self_val);
                    }
                } else {
                    if ret.data.len() == CAPACITY {
                        return Err(());
                    }
                    ret.data_index.push(*self_idx);
                    ret.data.push(*self_val);
                }
            }

            while let Some((rhs_idx, rhs_val)) = rhs_iter.next() {
                if ret.data.len() == CAPACITY {
                    return Err(());
                }
                ret.data_index.push(*rhs_idx);
                ret.data.push(*rhs_val);
            }

            self.data = ret.data;
            self.data_index = ret.data_index;

            Ok(())
        }
    }

    pub fn multiply(&mut self, rhs: f32) {
        for i in 0..self.data.len() {
            self.data[i] *= rhs;
        }
    }
}

impl<const CAPACITY: usize, const VIRTUALSIZE: usize> Index<usize>
    for VecSparseSimd<CAPACITY, VIRTUALSIZE>
{
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        let dereference_index = self.data_index.partition_point(|i| *i < index);
        if dereference_index < self.data.len() && self.data_index[dereference_index] == index {
            &self.data[dereference_index]
        } else {
            &0.
        }
    }
}

impl<const CAPACITY: usize, const VIRTUALSIZE: usize> IndexMut<usize>
    for VecSparseSimd<CAPACITY, VIRTUALSIZE>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let dereference_index = self.data_index.partition_point(|i| *i < index);

        let prev_size = self.data.len();

        if dereference_index < self.data_index.len() && self.data_index[dereference_index] == index
        {
            &mut self.data[dereference_index]
        } else {
            self.data_index.push(index);
            self.data.push(0.);

            self.data_index[dereference_index..=prev_size].rotate_right(1);
            self.data[dereference_index..=prev_size].rotate_right(1);

            self.data_index[dereference_index] = index;
            &mut self.data[dereference_index]
        }
    }
}

#[cfg(test)]
mod vec_sparse_simd_tests {
    use rand::{self, seq::SliceRandom, Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::array::from_fn;

    use crate::simd_arr::sparse_simd::VecSparseSimd;

    #[test]
    fn test_create() {
        assert_eq!(
            VecSparseSimd::<4, 4>::new_from_array(&[1., 4., 2., 4.]).unwrap(),
            VecSparseSimd {
                data_index: vec![0, 1, 2, 3],
                data: vec![1., 4., 2., 4.],
            }
        );

        assert_eq!(
            VecSparseSimd::<7, 7>::new_from_array(&[1., 4., 0., 0., 0., 2., 4.]).unwrap(),
            VecSparseSimd {
                data_index: vec![0, 1, 5, 6],
                data: vec![1., 4., 2., 4.],
            }
        );
    }

    fn test_add<const N: usize>(a: [f32; N], b: [f32; N]) {
        let res_vec = a
            .iter()
            .zip(b)
            .map(|(a_elm, b_elm)| a_elm + b_elm)
            .collect::<Vec<_>>();
        let res: [f32; N] = from_fn(|i| res_vec[i]);

        let mut x = VecSparseSimd::<N, N>::new_from_array(&a).unwrap();
        let y = VecSparseSimd::<N, N>::new_from_array(&b).unwrap();

        x.acumulate(&y).unwrap();

        assert_eq!(x.to_array(), res)
    }

    fn test_sub<const N: usize>(a: [f32; N], b: [f32; N]) {
        let res_vec = a
            .iter()
            .zip(b)
            .map(|(a_elm, b_elm)| a_elm - b_elm)
            .collect::<Vec<_>>();
        let res: [f32; N] = from_fn(|i| res_vec[i]);

        let mut x = VecSparseSimd::<N, N>::new_from_array(&a).unwrap();
        let mut y = VecSparseSimd::<N, N>::new_from_array(&b).unwrap();
        y.neg();

        x.acumulate(&y).unwrap();

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

            assert_eq!(
                a,
                VecSparseSimd::<32, 32>::new_from_array(&a)
                    .unwrap()
                    .to_array()
            )
        }
    }

    fn test_mul_scalar<const N: usize>(a: [f32; N], b: f32) {
        let res = a.map(|a_elm| a_elm * b);

        let mut test = VecSparseSimd::<N, N>::new_from_array(&a).unwrap();
        test.multiply(b);

        assert_eq!(test.to_array(), res)
    }

    fn test_div_scalar<const N: usize>(a: [f32; N], b: f32) {
        let res = a.map(|a_elm| a_elm * (1. / b)); // good enough

        let mut test = VecSparseSimd::<N, N>::new_from_array(&a).unwrap();
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
            let x = VecSparseSimd::<10, 10>::new_from_array(&test_arr).unwrap();

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

            let mut x: VecSparseSimd<4, 4> = VecSparseSimd::zero();
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

            let mut y = VecSparseSimd::<4, 4>::new_from_array(&b).unwrap();
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

            let mut x: VecSparseSimd<32, 32> = VecSparseSimd::zero();
            let mut y: VecSparseSimd<32, 32> = VecSparseSimd::new_from_array(&b).unwrap();
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
            let mut x = VecSparseSimd::<10, 10>::new_from_array(&[0.; 10]).unwrap();
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
        let x = VecSparseSimd::<5, 10>::new_from_array(&test).unwrap();

        assert_eq!(x.to_array(), test);
    }
}
