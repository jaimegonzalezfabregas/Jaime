extern crate test;

mod polinomial_test {
    use std::ops::{Add, Mul};

    use crate::trainer::{
        adam_trainer::AdamTrainer,
        asymptotic_gradient_descent_trainer::AsymptoticGradientDescentTrainer,
        default_param_translator, genetic_trainer::GeneticTrainer, DataPoint, Trainer,
    };

    fn base_func(x: f32) -> f32 {
        1. * (x * x * x * x * x) - 4. * (x * x * x * x) - 10. * (x * x * x)
            + 40. * (x * x)
            + 9. * x
            + -11.
    }

    #[test]
    fn genetic_polinomial_test() {
        polinomial_test(
            &mut GeneticTrainer::<_, _, _, 100, 5, _, _, _>::new(
                polinomial::<6, _>,
                default_param_translator,
                (),
                0.0001,
                10,
            ),
            0.1,
        );
    }

    #[test]
    fn gradient_descent_polinomial_test() {
        polinomial_test(
            &mut AsymptoticGradientDescentTrainer::new_dense(
                polinomial::<6, _>,
                polinomial::<6, _>,
                default_param_translator,
                (),
            ),
            1.,
        );
    }

    #[test]
    fn adam_polinomial_test() {
        polinomial_test(
            &mut AdamTrainer::new_dense(
                polinomial::<6, _>,
                polinomial::<6, _>,
                default_param_translator,
                (),
                0.0001,
                10,
            ),
            0.001,
        );
    }

    fn polinomial_test<T: Trainer<6, 1, 1>>(trainer: &mut T, learning_rate: f32) {
        let mut complexity = 0.;

        while complexity < 1. {
            let dataset = dataset_service(complexity);

            while !trainer.found_local_minima() {
                trainer.train_step::<true, false, _, _>(
                    &dataset,
                    &dataset,
                    dataset.len(),
                    dataset.len(),
                    learning_rate,
                )
            }

            complexity += 0.1;
        }

        let cost = trainer.get_last_cost().unwrap();
        assert!(cost < 0.01);

        let params = trainer.get_model_params();
        let diff = params
            .iter()
            .zip([-11., 9., 40., -10., -4., 1.].iter())
            .fold(0., |acc, (a, b)| acc + a - b);
        assert!(diff < 0.01);
    }

    fn dataset_service<const P: usize>(complexity: f32) -> Vec<DataPoint<P, 1, 1>> {
        let abs_max = (complexity + 1.) * 100.;

        (-(abs_max) as isize..(abs_max) as isize)
            .map(|x| x as f32 / 100.)
            // .map(|x| x as f32 / 10.)
            .map(|x| DataPoint {
                input: [x],
                output: [base_func(x)],
            })
            .collect::<Vec<_>>()
    }

    pub fn polinomial<
        const G: usize,
        N: Clone
            + From<f32>
            + PartialOrd<f32>
            + PartialOrd<N>
            + Add<N, Output = N>
            + Mul<N, Output = N>
            + Mul<f32, Output = N>,
    >(
        params: &[N; G],
        input: &[f32; 1],
        _: &(),
    ) -> [N; 1] {
        let mut ret = N::from(0.);
        let mut x_to_the_nth = N::from(1.);

        for n in 0..G {
            ret = ret + (x_to_the_nth.clone() * params[n].clone());

            x_to_the_nth = x_to_the_nth * input[0];
        }

        [ret]
    }
}
