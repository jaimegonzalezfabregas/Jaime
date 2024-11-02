use std::array;
use std::time::Instant;

use crate::dual::Dual;
use crate::simd_arr::dense_simd::DenseSimd;
use crate::simd_arr::hybrid_simd::{CriticalityCue, HybridSimd};
use crate::simd_arr::SimdArr;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use super::{dataset_cost, DataPoint, Trainer};

/// The AdamTrainer struct manages the training lifecycle using the Adam optimization algorithm.
#[derive(Clone)]
pub struct AdamTrainer<
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
    m: [f32; P], // First moment vector
    v: [f32; P], // Second moment vector
    t: usize,    // Time step
    param_translator: ParamTranslate,
    extra_data: ExtraData,
    last_cost: Option<f32>,
    cost_stagnation_value: Option<f32>,
    cost_stagnation_time: usize,
    cost_stagnation_threshold: f32,
    max_cost_stagnation_time: usize,
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
    > AdamTrainer<P, I, O, ExtraData, DenseSimd<P>, FG, F, ParamTranslate>
{
    pub fn new_dense(
        trainable: F,
        trainable_gradient: FG,
        param_translator: ParamTranslate,
        extra_data: ExtraData,
        cost_stagnation_threshold: f32,
        max_cost_stagnation_time: usize,
    ) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(2);

        Self {
            model_gradient: trainable_gradient,
            model: trainable,
            params: array::from_fn(|i| Dual::new_param(rng.gen::<f32>() - 0.5, i)),
            m: [0.0; P],
            v: [0.0; P],
            t: 0,
            param_translator,
            extra_data,
            last_cost: None,
            cost_stagnation_value: None,
            cost_stagnation_threshold,
            cost_stagnation_time: 0,
            max_cost_stagnation_time,
        }
    }
}

impl<
        const P: usize,
        const I: usize,
        const O: usize,
        const CRITICALITY: usize,
        ExtraData: Sync + Clone,
        FG: Fn(
                &[Dual<P, HybridSimd<P, CRITICALITY>>; P],
                &[f32; I],
                &ExtraData,
            ) -> [Dual<P, HybridSimd<P, CRITICALITY>>; O]
            + Sync
            + Clone,
        F: Fn(&[f32; P], &[f32; I], &ExtraData) -> [f32; O] + Sync + Clone,
        ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P] + Clone,
    > AdamTrainer<P, I, O, ExtraData, HybridSimd<P, CRITICALITY>, FG, F, ParamTranslate>
{
    pub fn new_hybrid(
        _: CriticalityCue<CRITICALITY>,
        trainable: F,
        trainable_gradient: FG,
        param_translator: ParamTranslate,
        extra_data: ExtraData,
        cost_stagnation_threshold: f32,
        max_cost_stagnation_time: usize,
    ) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(2);

        Self {
            model_gradient: trainable_gradient,
            model: trainable,
            params: array::from_fn(|i| Dual::new_param(rng.gen::<f32>() - 0.5, i)),
            m: [0.0; P],
            v: [0.0; P],
            t: 0,
            param_translator,
            extra_data,
            last_cost: None,
            cost_stagnation_value: None,
            cost_stagnation_threshold,
            cost_stagnation_time: 0,
            max_cost_stagnation_time,
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
        ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P] + Clone,
    > Trainer<P, I, O> for AdamTrainer<P, I, O, ExtraData, S, FG, F, ParamTranslate>
{
    fn get_last_cost(&self) -> Option<f32> {
        self.last_cost
    }

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
        learning_rate: f32,
    ) {
        let t0 = Instant::now();

        // Calculate the gradient using the direction dataset
        let cost = dataset_cost::<VERBOSE, false, PARALELIZE, _, _, _, _, _, _, _>(
            dir_dataset,
            dir_dataset_len,
            &self.params,
            &self.model_gradient,
            &self.extra_data,
        );

        let raw_gradient = cost.get_gradient();
        let gradient_size = f32::max(
            raw_gradient.iter().fold(0., |acc, elm| acc + (elm * elm)),
            1e-30,
        );

        // Update the time step
        self.t += 1;

        // Update the first and second moment estimates
        for i in 0..P {
            self.m[i] = 0.9 * self.m[i] + 0.1 * raw_gradient[i]; // beta1 = 0.9, beta2 = 0.999
            self.v[i] = 0.999 * self.v[i] + 0.001 * raw_gradient[i].powi(2); // epsilon = 1e-8
        }

        // Compute bias-corrected first and second moment estimates
        let m_hat: [f32; P] = array::from_fn(|i| self.m[i] / (1.0 - 0.9_f32.powi(self.t as i32)));
        let v_hat: [f32; P] = array::from_fn(|i| self.v[i] / (1.0 - 0.999_f32.powi(self.t as i32)));

        // Update parameters
        let gradient = array::from_fn(|i| -learning_rate * m_hat[i] / (v_hat[i].sqrt() + 1e-8)); // epsilon = 1e-8

        let og_parameters = array::from_fn(|i| self.params[i].get_real());
        let new_params = (self.param_translator)(&og_parameters, &gradient);

        for (i, param) in new_params.iter().enumerate() {
            self.params[i].set_real(*param);
        }

        // Calculate the new cost
        let new_cost: f32 = dataset_cost::<false, false, PARALELIZE, _, _, _, _, _, _, _>(
            full_dataset.clone(),
            full_dataset_len,
            &new_params,
            &self.model,
            &self.extra_data,
        );

        if let Some(last_cost) = self.last_cost {
            if (last_cost - new_cost).abs() < self.cost_stagnation_threshold {
                // stagnation_is_happening

                if let Some(stagnation_value) = self.cost_stagnation_value {
                    if (new_cost - stagnation_value).abs() < self.cost_stagnation_threshold {
                        self.cost_stagnation_time += 1;
                    } else {
                        self.cost_stagnation_time = 0;
                        self.cost_stagnation_value = Some(new_cost);
                    }
                } else {
                    self.cost_stagnation_time = 0;
                    self.cost_stagnation_value = Some(new_cost);
                }
            } else {
                // reset_stagnation
                self.cost_stagnation_value = None;
                self.cost_stagnation_time = 0;
            }
        }

        self.last_cost = Some(new_cost);

        if VERBOSE {
            println!(
                "Gradient length: {gradient_size:?} - New cost: {} - Time: {}",
                self.last_cost.unwrap(),
                t0.elapsed().as_secs_f32()
            );
        }
    }

    fn eval(&self, input: &[f32; I]) -> [f32; O] {
        (self.model)(
            &self.params.clone().map(|e| e.get_real()),
            input,
            &self.extra_data,
        )
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
        self.max_cost_stagnation_time < self.cost_stagnation_time
    }
}
