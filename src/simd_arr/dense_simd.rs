use std::ops::{Index, IndexMut};

use super::SimdArr;

#[derive(Clone, Debug)]
pub struct DenseSimd<const S: usize>([f32; S]);

impl<const S: usize> SimdArr<S> for DenseSimd<S> {
    fn zero() -> DenseSimd<S> {
        Self([0.; S])
    }

    fn to_array(&self) -> [f32; S] {
        self.0
    }

    fn new_from_value_and_pos(val: f32, pos: usize) -> Self {
        let mut ret = Self([0.; S]);
        ret.0[pos] = val;
        ret
    }

    fn neg(&mut self) {
        for i in 0..S {
            self.0[i] *= -1.;
        }
    }

    fn acumulate(&mut self, rhs: &Self) {
        for i in 0..S {
            self.0[i] += rhs[i];
        }
    }

    fn multiply(&mut self, rhs: f32) {
        for x in &mut self.0 {
            *x *= rhs;
        }
    }

    fn new_from_array(data: [f32; S]) -> DenseSimd<S> {
        Self(data)
    }

    fn check_nan(&self) {
        // self.0.iter().for_each(|x| assert!(x.is_finite()));
    }
}
impl<const S: usize> Index<usize> for DenseSimd<S> {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const S: usize> IndexMut<usize> for DenseSimd<S> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
