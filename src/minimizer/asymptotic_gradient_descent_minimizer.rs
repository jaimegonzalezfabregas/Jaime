use std::array;
use std::time::Instant;

use crate::dual::Dual;
use crate::simd_arr::dense_simd::DenseSimd;
use crate::simd_arr::hybrid_simd::{CriticalityCue, HybridSimd};
use crate::simd_arr::SimdArr;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::Minimizer;

/// This struct manages the maximizing lifecicle, it stores the trainable params and the training configuration.
/// - The generic P is the amount of parameters in the model
/// - The generic ExtraData is the type of the extra data parameter of the model. It can be used to alter manualy the model behabiour during training or to pass configuration data

#[derive(Clone)]
pub struct AsymptoticGradientDescentMinimizer<
    const P: usize,
    ExtraData: Sync + Clone,
    S: SimdArr<P>,
    FG: Fn(&[Dual<P, S>; P], &ExtraData) -> Dual<P, S> + Sync + Clone,
    F: Fn(&[f32; P], &ExtraData) -> f32 + Sync + Clone,
    ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P] + Clone,
> {
    cost_gradient: FG,
    cost: F,
    params: [Dual<P, S>; P],
    param_translator: ParamTranslate,
    extra_data: ExtraData,
    last_cost: Option<f32>,
    found_local_minima: bool,
}

impl<
        const P: usize,
        ExtraData: Sync + Clone,
        FG: Fn(&[Dual<P, DenseSimd<P>>; P], &ExtraData) -> Dual<P, DenseSimd<P>> + Sync + Clone,
        F: Fn(&[f32; P], &ExtraData) -> f32 + Sync + Clone,
        ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P] + Clone,
    > AsymptoticGradientDescentMinimizer<P, ExtraData, DenseSimd<P>, FG, F, ParamTranslate>
{
    /// Creates a Trainer instance that will use dense representation for the dual part. This can be ineficient for big parameter models because many of the elements of the dual part will be 0. This may be usefull for very small parameter numbers, but for any serious endeavor I would recomend using the new_hybrid function.
    pub fn new_dense(
        trainable: F,
        trainable_gradient: FG,
        param_translator: ParamTranslate,
        extra_data: ExtraData,
    ) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(2);

        Self {
            cost_gradient: trainable_gradient,
            cost: trainable,
            params: array::from_fn(|i| Dual::new_param(rng.gen::<f32>() - 0.5, i)),
            param_translator,
            extra_data,
            last_cost: None,
            found_local_minima: false,
        }
    }
}

impl<
        const P: usize,
        const CRITIALITY: usize,
        ExtraData: Sync + Clone,
        FG: Fn(
                &[Dual<P, HybridSimd<P, CRITIALITY>>; P],
                &ExtraData,
            ) -> Dual<P, HybridSimd<P, CRITIALITY>>
            + Sync
            + Clone,
        F: Fn(&[f32; P], &ExtraData) -> f32 + Sync + Clone,
        ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P] + Clone,
    >
    AsymptoticGradientDescentMinimizer<
        P,
        ExtraData,
        HybridSimd<P, CRITIALITY>,
        FG,
        F,
        ParamTranslate,
    >
{
    /// Creates a Trainer instance that will use sparse representation for the dual part when the amount of ceros is big. This tries to get the advantages of the dense representation when few ceros are present and the sparse representation when many ceros are present. The first parameter is the number of non cero elements that will trigger the translation from the sparse representation to dense representation. If you are experiencing slow training times try fiddleing with the CRITiCALITY value. With a CRITICALITY of 0 the trainer will behave exactly the same as a trainer created using the new_dense function with a small overhead
    pub fn new_hybrid(
        _: CriticalityCue<CRITIALITY>,
        trainable: F,
        trainable_gradient: FG,
        param_translator: ParamTranslate,
        extra_data: ExtraData,
    ) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(2);

        Self {
            cost_gradient: trainable_gradient,
            cost: trainable,
            params: array::from_fn(|i| Dual::new_param(rng.gen::<f32>() - 0.5, i)),
            param_translator,
            extra_data,
            last_cost: None,
            found_local_minima: false,
        }
    }
}

impl<
        const P: usize,
        ExtraData: Sync + Clone,
        S: SimdArr<P>,
        FG: Fn(&[Dual<P, S>; P], &ExtraData) -> Dual<P, S> + Sync + Clone,
        F: Fn(&[f32; P], &ExtraData) -> f32 + Sync + Clone,
        ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P] + Clone,
    > Minimizer<P> for AsymptoticGradientDescentMinimizer<P, ExtraData, S, FG, F, ParamTranslate>
{
    fn get_last_cost(&self) -> Option<f32> {
        self.last_cost
    }

    /// It will find the gradient and follow it. The "asintotic" means that will start trying to step in the direction of the gradient a lenght of 1. If the cost has gone up it will try half of the previous lenght untill the cost goes down or the lenght gets to a very very low number.
    /// This function will take adventage of the generalizationability of our cost function. It will calculate the gradient using Dual numbers but will calculate the cost using only floats during the internal iteration. This second calculation can be many many times faster than the first one, allowing us to test multiple step lenghts very quickly.
    /// - The VERBOSE generic will print out progress updates
    fn train_step<const VERBOSE: bool>(&mut self, learning_rate: f32) {
        let t0 = Instant::now();

        let cost = (self.cost_gradient)(&self.params, &self.extra_data);

        let f32_params = array::from_fn(|i| self.params[i].get_real());

        let fast_full_cost: f32 = (self.cost)(&f32_params, &self.extra_data);

        let mut factor = learning_rate;

        let raw_gradient = cost.get_gradient();
        let gradient_size: f32 = f32::max(
            raw_gradient.iter().fold(0., |acc, elm| acc + (elm * elm)),
            1e-30,
        );

        let unit_gradient = array::from_fn(|i| raw_gradient[i] / gradient_size.sqrt());
        let og_parameters = array::from_fn(|i| self.params[i].get_real());

        while {
            let gradient = unit_gradient.map(|e| -e * factor);

            let new_params = (self.param_translator)(&og_parameters, &gradient);

            for (i, param) in new_params.iter().enumerate() {
                self.params[i].set_real(*param);
            }

            let new_cost: f32 = (self.cost)(&new_params, &self.extra_data);
            self.last_cost = Some(new_cost);

            new_cost >= fast_full_cost
        } {
            factor *= 0.7;

            if factor < 1e-10 {
                self.found_local_minima = true;
                return;
            }
        }

        if VERBOSE {
            println!(
                "gradient length: {gradient_size:?} - fast_full_cost: {} - new cost: {} - learning factor: {} - improvement {} - time {}",
                fast_full_cost, self.last_cost.unwrap(), factor, fast_full_cost - self.last_cost.unwrap(), t0.elapsed().as_secs_f32()
            );
        }
    }

    fn get_model_params(&self) -> [f32; P] {
        self.params.clone().map(|e| e.get_real())
    }

    fn set_model_params(&mut self, parameters: [f32; P]) {
        let mut i = 0;
        self.params = parameters.map(|p| {
            i += 1;
            Dual::new_param(p, i - 1)
        });
    }

    fn found_local_minima(&self) -> bool {
        self.found_local_minima
    }
}
