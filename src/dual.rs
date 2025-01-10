pub mod addition;
pub mod division;
pub mod extended_arithmetic;
pub mod multiply;
pub mod substraction;
mod tests;

use std::array;

use rayon::array;

use crate::simd_arr::SimdArr;

/// The internal float-oid that implements forward mode automatic diferentiation. It implements many of the traits a float number would, that way you can use it in its place on a generic function. This is an internal data structure, if you are using it on your code its likely you are doing something wrong.
/// - The generic P is the ammount of parameters the model needs.
/// - The generic S is the SimdArr implementation that will be used as the dual part
/// The dual part of the dual number is not a single float, but an array of them. This is because we need to keep track of many derivatives for each of the parameters of the model. When the parameters are inputed in the model their dual part is set to all 0.0 but a single 1.0 in the position matching their index.
///
/// Dual numbers are used throughout the full gradient computation. From the input of the model to the cost calculation. That way the cost's dual array will contain the gradient, meaning, the partial derivative of the cost function for the nth parameter in the nth position of the array.

#[derive(Clone, Debug)]
pub struct Dual<const P: usize, S: SimdArr<P>> {
    real: f32,
    sigma: S,
}

impl<const P: usize, S: SimdArr<P>> From<f32> for Dual<P, S> {
    fn from(value: f32) -> Self {
        Self::new(value)
    }
}

impl<const P: usize, S: SimdArr<P>> Dual<P, S> {
    pub fn new_param<X: Into<f32>>(real: X, i: usize) -> Dual<P, S> {
        Self {
            real: real.into(),
            sigma: S::new_from_value_and_pos(1., i),
        }
    }

    pub fn check_nan(&self) {
        assert!(self.real.is_finite());
        self.sigma.check_nan();
    }

    pub(crate) fn set_real<X: Into<f32>>(&mut self, val: X) {
        self.real = val.into()
    }

    pub fn zero() -> Self {
        Self {
            real: 0.,
            sigma: S::zero(),
        }
    }

    pub fn get_gradient(&self) -> [f32; P] {
        self.sigma.to_array()
    }

    pub fn get_real(&self) -> f32 {
        self.real
    }

    pub fn new<X: Into<f32>>(real: X) -> Self {
        let mut ret = Self::zero();
        ret.real = real.into();
        ret
    }

    pub fn new_full<X: Into<f32> + Copy>(real: X, sigma: [X; P]) -> Self {
        Self {
            real: real.into(),
            sigma: SimdArr::new_from_array(array::from_fn(|i| sigma[i].into())),
        }
    }
}

impl<const P: usize, S: SimdArr<P>> PartialEq for Dual<P, S> {
    fn eq(&self, other: &Self) -> bool {
        self.real.eq(&other.real)
    }
}

impl<const P: usize, S: SimdArr<P>> PartialOrd for Dual<P, S> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.real.partial_cmp(&other.real)
    }
}

impl<const P: usize, S: SimdArr<P>> Eq for Dual<P, S> {}

impl<const P: usize, S: SimdArr<P>> PartialEq<f32> for Dual<P, S> {
    fn eq(&self, other: &f32) -> bool {
        self.real.eq(other)
    }
}

impl<const P: usize, S: SimdArr<P>> PartialOrd<f32> for Dual<P, S> {
    fn partial_cmp(&self, other: &f32) -> Option<std::cmp::Ordering> {
        self.real.partial_cmp(other)
    }
}

impl<const P: usize, S: SimdArr<P>> From<Dual<P, S>> for f32 {
    fn from(value: Dual<P, S>) -> Self {
        value.get_real()
    }
}

fn check_nan<const P: usize, S: SimdArr<P>>(d: Dual<P, S>) -> Dual<P, S> {
    d.check_nan();

    d
}
