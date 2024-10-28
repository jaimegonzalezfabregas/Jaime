use std::array;
use std::fs::OpenOptions;
use std::io::{self, BufRead, Write};
use std::ops::{Add, Div, Sub};
use std::time::Instant;

use crate::dual::extended_arithmetic::ExtendedArithmetic;
use crate::dual::Dual;
use crate::simd_arr::dense_simd::DenseSimd;
use crate::simd_arr::hybrid_simd::HybridSimd;
use crate::simd_arr::SimdArr;
use indicatif::ParallelProgressIterator;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::fmt::Debug;

#[derive(Debug, Clone, Copy)]
pub struct DataPoint<const P: usize, const I: usize, const O: usize> {
    pub input: [f32; I],
    pub output: [f32; O],
}

pub struct CriticalityCue<const CRITICALITY: usize>();

fn datapoint_cost<
    const P: usize,
    const I: usize,
    const O: usize,
    N: ExtendedArithmetic + Clone + Sub<f32, Output = N> + Add<N, Output = N> + Debug + From<f32>,
>(
    goal: &DataPoint<P, I, O>,
    prediction: [N; O],
) -> N {
    let mut ret = N::from(0.);

    for (pred_val, goal_val) in prediction.clone().into_iter().zip(goal.output.into_iter()) {
        let cost = pred_val.clone() - goal_val;
        // println!("    scalar cost for {pred_val:?} and {goal_val:?} is {cost:?}");
        // println!("{ret:?} + {cost:?}");

        ret = ret + cost.abs();

        // ret = ret + cost.pow2();
    }
    ret
}

fn dataset_cost<
    'a,
    'b,
    const PROGRESS: bool,
    const DEBUG: bool,
    const PARALELIZE: bool,
    const P: usize,
    const I: usize,
    const O: usize,
    ExtraData: Sync + Clone,
    N: ExtendedArithmetic
        + Clone
        + Sub<f32, Output = N>
        + Add<f32, Output = N>
        + Debug
        + From<f32>
        + Add<N, Output = N>
        + Div<f32, Output = N>
        + Send
        + Sync,
    F: Fn(&[N; P], &[f32; I], &ExtraData) -> [N; O] + Sync,
    D: IntoIterator<Item = &'b DataPoint<P, I, O>>
        + IntoParallelIterator<Item = &'a DataPoint<P, I, O>>,
>(
    dataset: D,
    dataset_len: usize,
    params: &[N; P],
    model: F,
    extra: &ExtraData,
) -> N {
    let mut accumulator = N::from(0.);
    let cost_list = if PARALELIZE {
        if PROGRESS {
            dataset
                .into_par_iter()
                .progress_count(dataset_len as u64)
                .map(|data_point| {
                    let prediction = (model)(&params, &data_point.input, &extra);

                    if DEBUG {
                        println!("goal {:?} predition {:?}", data_point.output, prediction);
                    }

                    datapoint_cost(&data_point, prediction)
                })
                .collect::<Vec<_>>()
        } else {
            dataset
                .into_par_iter()
                .map(|data_point| {
                    let prediction = (model)(&params, &data_point.input, &extra);
                    if DEBUG {
                        println!("goal {:?} predition {:?}", data_point.output, prediction);
                    }
                    datapoint_cost(&data_point, prediction)
                })
                .collect::<Vec<_>>()
        }
    } else {
        dataset
            .into_iter()
            .map(|data_point| {
                let prediction = (model)(&params, &data_point.input, &extra);
                if DEBUG {
                    println!("goal {:?} predition {:?}", data_point.output, prediction);
                }
                datapoint_cost(&data_point, prediction)
            })
            .collect::<Vec<_>>()
    };

    for cost in cost_list {
        accumulator = accumulator + cost;
    }

    accumulator = accumulator / dataset_len as f32;

    accumulator
}

pub fn default_param_translator<const P: usize>(params: &[f32; P], vector: &[f32; P]) -> [f32; P] {
    array::from_fn(|i| params[i] + vector[i])
}

pub fn param_translator_with_bounds<const P: usize, const MAX: isize, const MIN: isize>(
    params: &[f32; P],
    vector: &[f32; P],
) -> [f32; P] {
    array::from_fn(|i| (params[i] + vector[i]).min(MAX as f32).max(MIN as f32))
}

#[derive(Clone)]

pub struct Trainer<
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
    > Trainer<P, I, O, ExtraData, DenseSimd<P>, FG, F, ParamTranslate>
{
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
    > Trainer<P, I, O, ExtraData, HybridSimd<P, CRITIALITY>, FG, F, ParamTranslate>
{
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
    > Trainer<P, I, O, ExtraData, S, FG, F, ParamTranslate>
{
    pub fn get_model_params(&self) -> [f32; P] {
        self.params.clone().map(|e| e.get_real())
    }

    pub fn save(&self, file_path: &str) -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(file_path)?;

        for p in self.params.iter() {
            file.write(format!("{}\n", p.get_real()).as_bytes())?;
        }

        Ok(())
    }

    pub fn load(&mut self, file_path: &str) -> std::io::Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .read(true)
            .create(true)
            .open(file_path)?;
        let reader = io::BufReader::new(file);

        for (i, line) in reader.lines().enumerate() {
            let line = line?;
            if i < self.params.len() {
                if let Ok(param) = line.parse::<f32>() {
                    self.params[i].set_real(param);
                } else {
                    eprintln!("Failed to parse line: {}", line);
                }
            } else {
                break;
            }
        }

        Ok(())
    }

    pub fn shake(&mut self, factor: f32) {
        for i in 0..P {
            self.params[i]
                .set_real(self.params[i].get_real() + (rand::random::<f32>() - 0.5) * factor);
        }
    }

    // TODO partition the dataset

    pub fn train_stocastic_step<
        const PARALELIZE: bool,
        const VERBOSE: bool,
        CB: Fn(usize, &mut Self),
    >(
        &mut self,
        dataset: &Vec<DataPoint<P, I, O>>,
        subdataset_size: usize,
        inter_step_callback: CB,
    ) -> bool {
        let mut ret = false;
        for (i, sub_dataset) in dataset.chunks(subdataset_size).enumerate() {
            self.last_cost = None;
            ret |= self.train_step_asintotic_search::<PARALELIZE, VERBOSE, _, _>(
                sub_dataset,
                dataset,
                sub_dataset.len(),
                dataset.len(),
            );
            inter_step_callback(i, self);
        }

        ret
    }

    // TODO adam

    // TODO return a proper error when NaN apears

    pub fn train_step_asintotic_search<
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

        let cost: Dual<P, S> = dataset_cost::<VERBOSE, false, PARALELIZE, _, _, _, _, _, _, _>(
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

    // TODO

    pub fn get_last_cost(&self) -> Option<f32> {
        self.last_cost
    }

    pub fn eval(&self, input: &[f32; I]) -> [f32; O] {
        (self.model)(
            &self.params.clone().map(|e| e.get_real()),
            input,
            &self.extra_data,
        )
    }
}
