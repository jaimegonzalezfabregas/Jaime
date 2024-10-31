use std::array;
use std::time::Instant;

use crate::dual::Dual;
use crate::simd_arr::dense_simd::DenseSimd;
use crate::simd_arr::hybrid_simd::{CriticalityCue, HybridSimd};
use crate::simd_arr::SimdArr;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::fmt::Debug;

use super::{dataset_cost, Trainer};

/// A data point holds the desired output for a given input. A colection of datapoints is a dataset. A dataset defines the desired behabiour of the trainable model.

#[derive(Debug, Clone, Copy)]
pub struct DataPoint<const P: usize, const I: usize, const O: usize> {
    pub input: [f32; I],
    pub output: [f32; O],
}

/// The gradient descent algorithm needs to apply the graident to the parameter vector to progress. This operation is done withing a callback so that the user can have some control over the parameter values (clamping them, adding noise or any other usecase specific requirements).
/// When that ammount of control is not required this default param translator can be used as the callback.

pub fn default_param_translator<const P: usize>(params: &[f32; P], vector: &[f32; P]) -> [f32; P] {
    array::from_fn(|i| params[i] + vector[i])
}

/// The gradient descent algorithm needs to apply the graident to the parameter vector to progress. This operation is done withing a callback so that the user can have some control over the parameter values (clamping them, adding noise or any other usecase specific requirements).
/// This is an example of the clamping usecase mentioned in the "default_param_translator" description

pub fn param_translator_with_bounds<const P: usize, const MAX: isize, const MIN: isize>(
    params: &[f32; P],
    vector: &[f32; P],
) -> [f32; P] {
    array::from_fn(|i| (params[i] + vector[i]).min(MAX as f32).max(MIN as f32))
}

/// This struct manages the training lifecicle, it stores the trainable params and the training configuration.
/// - The generic P is the amount of parameters in the model
/// - The generic I is the amount of elements in the model input
/// - The generic O is the amount of elements in the model output
/// - The generic ExtraData is the type of the extra data parameter of the model. It can be used to alter manualy the model behabiour during training or to pass configuration data

#[derive(Clone)]
pub struct GradientDescentTrainer<
    const P: usize,
    const I: usize,
    const O: usize,
    ExtraData: Sync + Clone,
    S: SimdArr<P>,
    FG: Fn(&[Dual<P, S>; P], &[f32; I], &ExtraData) -> [Dual<P, S>; O] + Sync + Clone,
    F: Fn(&[f32; P], &[f32; I], &ExtraData) -> [f32; O] + Sync + Clone,
    ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P] + Clone,
> {
    model_gradient: FG,
    model: F,
    params: [Dual<P, S>; P],
    param_translator: ParamTranslate,
    extra_data: ExtraData,
    last_cost: Option<f32>,
}

impl<
        const P: usize,
        const I: usize,
        const O: usize,
        ExtraData: Sync + Clone,
        FG: Fn(&[Dual<P, DenseSimd<P>>; P], &[f32; I], &ExtraData) -> [Dual<P, DenseSimd<P>>; O]
            + Sync
            + Clone,
        F: Fn(&[f32; P], &[f32; I], &ExtraData) -> [f32; O] + Sync + Clone,
        ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P] + Clone,
    > GradientDescentTrainer<P, I, O, ExtraData, DenseSimd<P>, FG, F, ParamTranslate>
{
    /// Creates a Trainer instance that will use dense representation for the dual part. This can be ineficient for big parameter models because many of the elements of the dual part will be 0. This may be usefull for very small parameter numbers, but for any serious endeavor I would recomend using the new_hybrid function.
    pub fn new_dense(
        trainable: F,
        trainable_gradient: FG,
        param_translator: ParamTranslate,
        extra_data: ExtraData,
    ) -> Self {
        rayon::ThreadPoolBuilder::new()
            .stack_size(1 * 1024 * 1024 * 1024)
            .build_global()
            .unwrap();

        let mut rng = ChaCha8Rng::seed_from_u64(2);

        Self {
            model_gradient: trainable_gradient,
            model: trainable,
            params: array::from_fn(|i| Dual::new_param(rng.gen::<f32>() - 0.5, i)),
            param_translator,
            extra_data,
            last_cost: None,
        }
    }
}

impl<
        const P: usize,
        const I: usize,
        const O: usize,
        const CRITIALITY: usize,
        ExtraData: Sync + Clone,
        FG: Fn(
                &[Dual<P, HybridSimd<P, CRITIALITY>>; P],
                &[f32; I],
                &ExtraData,
            ) -> [Dual<P, HybridSimd<P, CRITIALITY>>; O]
            + Sync
            + Clone,
        F: Fn(&[f32; P], &[f32; I], &ExtraData) -> [f32; O] + Sync + Clone,
        ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P] + Clone,
    > GradientDescentTrainer<P, I, O, ExtraData, HybridSimd<P, CRITIALITY>, FG, F, ParamTranslate>
{
    /// Creates a Trainer instance that will use sparse representation for the dual part when the amount of ceros is big. This tries to get the advantages of the dense representation when few ceros are present and the sparse representation when many ceros are present. The first parameter is the number of non cero elements that will trigger the translation from the sparse representation to dense representation. If you are experiencing slow training times try fiddleing with the CRITiCALITY value. With a CRITICALITY of 0 the trainer will behave exactly the same as a trainer created using the new_dense function with a small overhead
    pub fn new_hybrid(
        _: CriticalityCue<CRITIALITY>,
        trainable: F,
        trainable_gradient: FG,
        param_translator: ParamTranslate,
        extra_data: ExtraData,
    ) -> Self {
        rayon::ThreadPoolBuilder::new()
            .stack_size(1 * 1024 * 1024 * 1024)
            .build_global();

        let mut rng = ChaCha8Rng::seed_from_u64(2);

        Self {
            model_gradient: trainable_gradient,
            model: trainable,
            params: array::from_fn(|i| Dual::new_param(rng.gen::<f32>() - 0.5, i)),
            param_translator,
            extra_data,
            last_cost: None,
        }
    }
}

impl<
        const P: usize,
        const I: usize,
        const O: usize,
        ExtraData: Sync + Clone,
        S: SimdArr<P>,
        FG: Fn(&[Dual<P, S>; P], &[f32; I], &ExtraData) -> [Dual<P, S>; O] + Sync + Clone,
        F: Fn(&[f32; P], &[f32; I], &ExtraData) -> [f32; O] + Sync + Clone,
        ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P] + Sync + Clone,
    > GradientDescentTrainer<P, I, O, ExtraData, S, FG, F, ParamTranslate>
{
    
}

impl<
        const P: usize,
        const I: usize,
        const O: usize,
        const CRITIALITY: usize,
        ExtraData: Sync + Clone,
        FG: Fn(
                &[Dual<P, HybridSimd<P, CRITIALITY>>; P],
                &[f32; I],
                &ExtraData,
            ) -> [Dual<P, HybridSimd<P, CRITIALITY>>; O]
            + Sync
            + Clone,
        F: Fn(&[f32; P], &[f32; I], &ExtraData) -> [f32; O] + Sync + Clone,
        ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P] + Clone,
    > Trainer<P, I, O>
    for GradientDescentTrainer<P, I, O, ExtraData, HybridSimd<P, CRITIALITY>, FG, F, ParamTranslate>
{
    fn get_last_cost(&self) -> Option<f32> {
        self.last_cost
    }

    /// Given a dataset it will find the gradient and follow it. The "asintotic" means that will start trying to step in the direction of the gradient a lenght of 1. If the cost has gone up it will try half of the previous lenght untill the cost goes down or the lenght gets to a very very low number.
    /// This function will take adventage of the generalizationability of our model function. It will calculate the gradient using Dual numbers and the dir_dataset (dir is short for direction) but will calculate the cost using only floats and the full_dataset. This second calculation can be many many times faster than the first one, allowing us to test multiple step lenghts very quickly. This gradient-cost separation also allows us to get an aproximation of the gradient faster than calculating the real one but allows us to always step in the right direction (even if the gradient isnt perfect)
    /// - The PARALELIZE generic will switch between singlethread or paraleloperations (using rayon)  
    /// - The VERBOSE generic will print out progress updates
    /// - The dir_dataset parameter holds the dataset used to find the gradient
    /// - The full_dataset parameter holds the dataset used to find the cost (usualy bigger, as the cost calculation is many many times faster)
    fn train_step<
        'a,
        'b,
        const PARALELIZE: bool,
        const VERBOSE: bool,
        D: IntoIterator<Item = &'b DataPoint<P, I, O>>
            + IntoParallelIterator<Item = &'a DataPoint<P, I, O>>
            + Clone,
        E: IntoIterator<Item = &'b DataPoint<P, I, O>>
            + IntoParallelIterator<Item = &'a DataPoint<P, I, O>>
            + Clone,
    >(
        &mut self,
        dir_dataset: D,
        full_dataset: E,
        dir_dataset_len: usize,
        full_dataset_len: usize,
    ) -> bool {
        let t0 = Instant::now();

        let cost = dataset_cost::<VERBOSE, false, PARALELIZE, _, _, _, _, _, _, _>(
            dir_dataset,
            dir_dataset_len,
            &self.params,
            &self.model_gradient,
            &self.extra_data,
        );

        let fast_full_cost: f32 = dataset_cost::<false, false, PARALELIZE, _, _, _, _, _, _, _>(
            full_dataset.clone(),
            full_dataset_len,
            &self.params.clone().map(|x| x.get_real()),
            &self.model,
            &self.extra_data,
        );

        let mut factor = 1.;

        let raw_gradient = cost.get_gradient();
        let gradient_size: f32 = raw_gradient
            .iter()
            .fold(0., |acc, elm| acc + (elm * elm))
            .max(1e-30);

        let unit_gradient = array::from_fn(|i| raw_gradient[i] / gradient_size.sqrt());
        let og_parameters = array::from_fn(|i| self.params[i].get_real());

        while {
            let gradient = unit_gradient.map(|e| -e * factor);

            let new_params = (self.param_translator)(&og_parameters, &gradient);

            for (i, param) in new_params.iter().enumerate() {
                self.params[i].set_real(*param);
            }

            let new_cost: f32 = dataset_cost::<false, false, PARALELIZE, _, _, _, _, _, _, _>(
                full_dataset.clone(),
                full_dataset_len,
                &new_params,
                &self.model,
                &self.extra_data,
            );
            self.last_cost = Some(new_cost);

            new_cost >= fast_full_cost
        } {
            factor *= 0.7;

            if factor < 1e-10 {
                return false;
            }
        }

        if VERBOSE {
            println!(
                "gradient length: {gradient_size:?} - fast_full_cost: {} - new cost: {} - learning factor: {} - improvement {} - time {}",
                fast_full_cost, self.last_cost.unwrap(), factor, fast_full_cost - self.last_cost.unwrap(), t0.elapsed().as_secs_f32()
            );
        }

        return true;
    }

    /// Will call the model for you, as an alternative of using this function you can also take the parameters out with get_model_params and call it yourself

    fn eval(&self, input: &[f32; I]) -> [f32; O] {
        (self.model)(
            &self.params.clone().map(|e| e.get_real()),
            input,
            &self.extra_data,
        )
    }

    /// Retrieve the parameters at any point during the training process
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
}
