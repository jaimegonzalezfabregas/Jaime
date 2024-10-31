use std::fs::OpenOptions;
use std::io::{self, BufRead, Write};
use std::ops::{Add, Div, Sub};

use crate::dual::extended_arithmetic::ExtendedArithmetic;
use gradient_descent_trainer::DataPoint;
use indicatif::ParallelProgressIterator;
use rayon::prelude::*;
use std::fmt::Debug;

pub mod gradient_descent_trainer;

pub trait Trainer<const P: usize, const I: usize, const O: usize> {
    /// Will return the last computed cost (if any has been computed yet)
    fn get_last_cost(&self) -> Option<f32>;

    fn eval(&self, input: &[f32; I]) -> [f32; O];

    fn get_model_params(&self) -> [f32; P];

    fn set_model_params(&mut self, parameters: [f32; P]);

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
    ) -> bool;

    /// Given a dataset and a subdataset size this function will calculate the gradient per subdataset making the corresponding steps in the way. It will call train_step_asintotic_search with the subdataset as the dir_dataset and the full dataset as the full_dataset.
    /// - The PARALELIZE generic will switch between singlethread or paraleloperations (using rayon)  
    /// - The VERBOSE generic will print out progress updates
    fn train_stocastic_step<
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
            ret |= self.train_step::<PARALELIZE, VERBOSE, _, _>(
                sub_dataset,
                dataset,
                sub_dataset.len(),
                dataset.len(),
            );
            inter_step_callback(i, self);
        }

        ret
    }

    /// Introduce random variations in the parameters. Can be usefull to scape local minima.
    fn shake(&mut self, factor: f32) {
        let mut shaken_params = self.get_model_params();

        for i in 0..P {
            shaken_params[i] += (rand::random::<f32>() - 0.5) * factor;
        }

        self.set_model_params(shaken_params);
    }

    /// Store parameters into a file
    fn save(&self, file_path: &str) -> std::io::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(file_path)?;

        for p in self.get_model_params().iter() {
            file.write(format!("{}\n", p).as_bytes())?;
        }

        Ok(())
    }

    /// Load parameters from file. A lot of assumptions about the file format are made, only files saved from a similar trainer are guarantied to work.
    fn load(&mut self, file_path: &str) -> std::io::Result<()> {
        let file = OpenOptions::new()
            .write(true)
            .read(true)
            .create(true)
            .open(file_path)?;
        let reader = io::BufReader::new(file);

        let mut loading_params = [0.; P];

        for (i, line) in reader.lines().enumerate() {
            let line = line?;
            if i < P {
                if let Ok(param) = line.parse::<f32>() {
                    loading_params[i] = param;
                } else {
                    eprintln!("Failed to parse line: {}", line);
                }
            } else {
                break;
            }
        }

        self.set_model_params(loading_params);

        Ok(())
    }
}

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
