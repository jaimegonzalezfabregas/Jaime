use std::ops::Mul;

use crate::simd_arr::SimdArr;

use super::{check_nan, Dual};

impl<const P: usize, S: SimdArr<P>> Mul<Dual<P, S>> for Dual<P, S> {
    type Output = Dual<P, S>;

    fn mul(mut self, mut rhs: Dual<P, S>) -> Self::Output {
        self.sigma.multiply(rhs.real);
        rhs.sigma.multiply(self.real);

        self.real *= rhs.real;

        self.sigma.acumulate(&rhs.sigma);

        check_nan(self)
    }
}

impl<const P: usize, S: SimdArr<P>> Mul<f32> for Dual<P, S> {
    type Output = Dual<P, S>;

    fn mul(mut self, rhs: f32) -> Self::Output {
        self.real *= rhs;

        self.sigma.multiply(rhs);

        check_nan(self)
    }
}
