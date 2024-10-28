use std::ops::Sub;

use crate::simd_arr::SimdArr;

use super::{check_nan, Dual};

impl<const P: usize, S: SimdArr<P>> Sub<Dual<P, S>> for Dual<P, S> {
    type Output = Dual<P, S>;

    fn sub(mut self, mut rhs: Dual<P, S>) -> Self::Output {
        self.real -= &rhs.real;
        rhs.sigma.neg();
        self.sigma.acumulate(&rhs.sigma);

        check_nan(self)
    }
}

impl<const P: usize, S: SimdArr<P>> Sub<f32> for Dual<P, S> {
    type Output = Dual<P, S>;

    fn sub(mut self, rhs: f32) -> Self::Output {
        self.real -= rhs;

        check_nan(self)
    }
}
