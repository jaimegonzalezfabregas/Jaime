use std::ops::{Index, IndexMut};

use super::SimdArr;

/// INTERNAL an implementation of SimdArr using a dense representation (storing every value in an array). The time complexity of all operations is at most linear over the capacity.
#[derive(Clone, Debug)]
pub struct DenseSimdNaNProtected<const S: usize>([f32; S]);

fn addition_best_finite_approximation(a: f32, b: f32) -> f32 {
    let sum = a + b;

    // Check if the sum is finite
    if sum.is_finite() {
        sum
    } else {
        // If the sum is not finite, return the maximum finite value
        if sum.is_infinite() {
            if sum.is_sign_negative() {
                f32::MIN
            } else {
                f32::MAX
            }
        } else {
            // Handle NaN case
            panic!("a nan was encountered")
        }
    }
}

fn product_best_finite_approximation(a: f32, b: f32) -> f32 {
    let product = a * b;

    // Check if the product is finite
    if product.is_finite() {
        product
    } else {
        // If the product is not finite, return the maximum finite value
        if product.is_infinite() {
            if product.is_sign_negative() {
                f32::MIN
            } else {
                f32::MAX
            }
        } else {
            panic!("a nan was encountered")
        }
    }
}

impl<const S: usize> SimdArr<S> for DenseSimdNaNProtected<S> {
    fn zero() -> DenseSimdNaNProtected<S> {
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
            self.0[i] = addition_best_finite_approximation(self.0[i], rhs[i]);
        }
    }

    fn multiply(&mut self, rhs: f32) {
        for x in &mut self.0 {
            *x = product_best_finite_approximation(*x, rhs);
        }
    }

    fn new_from_array(data: [f32; S]) -> DenseSimdNaNProtected<S> {
        Self(data)
    }

    fn check_nan(&self) {
        // self.0.iter().for_each(|x| assert!(x.is_finite()));
    }
}
impl<const S: usize> Index<usize> for DenseSimdNaNProtected<S> {
    type Output = f32;
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<const S: usize> IndexMut<usize> for DenseSimdNaNProtected<S> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
