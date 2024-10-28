#[cfg(test)]
mod dual_tests {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    use crate::{
        dual::{extended_arithmetic::ExtendedArithmetic, Dual},
        simd_arr::hybrid_simd::HybridSimd,
    };

    #[test]
    fn create() {
        let dual: Dual<4, HybridSimd<4, 4>> = Dual::zero();

        assert_eq!(dual.real, 0.)
    }

    #[test]
    fn arithmetic_stress() {
        let mut rng = ChaCha8Rng::seed_from_u64(2);

        for _ in 0..1000 {
            let scalar_a = rng.gen();
            let scalar_b = rng.gen();

            let a: Dual<4, HybridSimd<4, 4>> = Dual::new(scalar_a);
            let b: Dual<4, HybridSimd<4, 4>> = Dual::new(scalar_b);

            assert_eq!(scalar_a.sqrt(), (a.clone().sqrt()).get_real());
            assert_eq!(-scalar_a, (a.clone().neg()).get_real());
            assert_eq!(scalar_a * scalar_a, (a.clone().pow2()).get_real());
            assert_eq!(scalar_a + scalar_b, (a.clone() + b.clone()).get_real());
            assert_eq!(scalar_a - scalar_b, (a.clone() - b.clone()).get_real());
            assert_eq!(scalar_a / scalar_b, (a.clone() / b.clone()).get_real());
            assert_eq!(scalar_a * scalar_b, (a * b).get_real());
        }
    }
}
