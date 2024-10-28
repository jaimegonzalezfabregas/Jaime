pub mod addition;
pub mod division;
pub mod extended_arithmetic;
pub mod multiply;
pub mod substraction;
mod tests;

use crate::simd_arr::SimdArr;

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
    pub fn new_param(real: f32, i: usize) -> Dual<P, S> {
        Self {
            real: real,
            sigma: S::new_from_value_and_pos(1., i),
        }
    }

    pub fn check_nan(&self) {
        assert!(self.real.is_finite());
        self.sigma.check_nan();
    }

    pub(crate) fn set_real(&mut self, val: f32) {
        self.real = val
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

    pub fn new(real: f32) -> Self {
        let mut ret = Self::zero();
        ret.real = real;
        ret
    }

    pub fn new_full(real: f32, sigma: [f32; P]) -> Self {
        Self {
            real,
            sigma: SimdArr::new_from_array(sigma),
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
