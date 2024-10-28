use std::ops::Div;

use crate::simd_arr::SimdArr;

use super::{check_nan, Dual};

impl<const P: usize, S: SimdArr<P>> Div<Dual<P, S>> for Dual<P, S> {
    type Output = Dual<P, S>;

    fn div(mut self, mut rhs: Dual<P, S>) -> Self::Output {
        self.sigma.multiply(rhs.real);
        rhs.sigma.multiply(self.real);

        rhs.sigma.neg();
        self.sigma.acumulate(&rhs.sigma);

        self.sigma.multiply(1. / (rhs.real * rhs.real));

        self.real /= rhs.real;

        check_nan(self)
    }
}

impl<const P: usize, S: SimdArr<P>> Div<f32> for Dual<P, S> {
    type Output = Dual<P, S>;

    fn div(mut self, rhs: f32) -> Self::Output {
        self.real /= rhs;

        self.sigma.multiply(1. / rhs);

        check_nan(self)
    }
}
