use std::array;
use std::time::Instant;

use rand::Rng;
use rayon::prelude::*;

use super::{dataset_cost, DataPoint, Trainer};

#[derive(Clone)]
struct Agent<const P: usize> {
    parameters: [f32; P],
    cost: Option<f32>,
}

impl<const P: usize> Agent<P> {
    pub fn mutate<ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P] + Clone>(
        &self,
        param_translator: ParamTranslate,
        factor: f32,
    ) -> Agent<P> {
        let mut rng = rand::thread_rng();
        let noise = array::from_fn(|_| (rng.gen::<f32>() - 0.5) * factor);

        Agent {
            parameters: param_translator(&self.parameters, &noise),
            cost: None,
        }
    }

    pub fn new() -> Agent<P> {
        let mut rng = rand::thread_rng();

        Agent {
            parameters: array::from_fn(|_| rng.gen::<f32>() - 0.5),
            cost: None,
        }
    }

    pub fn calculate_cost<
        'a,
        'b,
        const I: usize,
        const O: usize,
        const PROGRESS: bool,
        const DEBUG: bool,
        const PARALELIZE: bool,
        D: IntoIterator<Item = &'b DataPoint<P, I, O>>
            + IntoParallelIterator<Item = &'a DataPoint<P, I, O>>
            + Clone,
        ExtraData: Clone + Sync,
        F: Fn(&[f32; P], &[f32; I], &ExtraData) -> [f32; O] + Sync,
    >(
        &mut self,
        dataset: D,
        dataset_len: usize,
        model: F,
        extra_data: &ExtraData,
    ) {
        if let None = self.cost {
            self.cost = Some(dataset_cost::<
                PROGRESS,
                DEBUG,
                PARALELIZE,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            >(
                dataset, dataset_len, &self.parameters, model, extra_data
            ))
        }
    }
}

/// The GeneticTrainer struct manages the training lifecycle using a evolutionary genetic algorithm
#[derive(Clone)]
pub struct GeneticTrainer<
    const P: usize,
    const I: usize,
    const O: usize,
    const GENERATION_SURVIVORS: usize,
    const GROUTH_FACTOR: usize,
    ExtraData: Sync + Clone,
    F: Fn(&[f32; P], &[f32; I], &ExtraData) -> [f32; O] + Sync + Clone,
    ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P] + Clone,
> {
    model: F,
    population: Vec<Agent<P>>,
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
        const GENERATION_SURVIVORS: usize,
        const GROUTH_FACTOR: usize,
        ExtraData: Sync + Clone,
        F: Fn(&[f32; P], &[f32; I], &ExtraData) -> [f32; O] + Sync + Clone,
        ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P] + Clone,
    > GeneticTrainer<P, I, O, GENERATION_SURVIVORS, GROUTH_FACTOR, ExtraData, F, ParamTranslate>
{
    pub fn new(
        trainable: F,
        param_translator: ParamTranslate,
        extra_data: ExtraData,
        cost_stagnation_threshold: f32,
        max_cost_stagnation_time: usize,
    ) -> Self {
        Self {
            model: trainable,
            population: vec![Agent::new(); GENERATION_SURVIVORS],
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
        const GENERATION_SURVIVORS: usize,
        const GROUTH_FACTOR: usize,
        ExtraData: Sync + Clone,
        F: Fn(&[f32; P], &[f32; I], &ExtraData) -> [f32; O] + Sync + Clone,
        ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P] + Clone,
    > Trainer<P, I, O>
    for GeneticTrainer<P, I, O, GENERATION_SURVIVORS, GROUTH_FACTOR, ExtraData, F, ParamTranslate>
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
        _: D,
        full_dataset: E,
        _: usize,
        full_dataset_len: usize,
        learning_rate: f32,
    ) {
        let t0 = Instant::now();

        let mut full_population: Vec<Agent<P>> = vec![];

        for i in 0..GENERATION_SURVIVORS {
            full_population.push(self.population[i].clone());

            for j in 0..(GROUTH_FACTOR) {
                full_population
                    .push(self.population[i].mutate(&self.param_translator, learning_rate / j as f32))
            }
        }

        full_population.iter_mut().for_each(|e| {
            e.calculate_cost::<_, _, false, false, false, _, _, _>(
                full_dataset.clone(),
                full_dataset_len,
                &self.model,
                &self.extra_data,
            )
        });

        full_population.sort_by(|a, b| a.cost.unwrap().partial_cmp(&b.cost.unwrap()).unwrap());

        let new_cost = full_population[0].cost.unwrap();

        for i in 0..GENERATION_SURVIVORS {
            self.population[i] = full_population[i].clone();
        }

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
                "New cost: {} - Time: {}",
                self.last_cost.unwrap(),
                t0.elapsed().as_secs_f32()
            );
        }
    }

    fn eval(&self, input: &[f32; I]) -> [f32; O] {
        (self.model)(&self.get_model_params(), input, &self.extra_data)
    }

    fn get_model_params(&self) -> [f32; P] {
        self.population[0].parameters
    }

    fn set_model_params(&mut self, parameters: [f32; P]) {
        self.population[0] = Agent {
            parameters,
            cost: None,
        };

        for i in 0..(GENERATION_SURVIVORS - 1) {
            self.population[i] = self.population[0].mutate(&self.param_translator, 1.);
        }
    }

    fn found_local_minima(&self) -> bool {
        self.max_cost_stagnation_time < self.cost_stagnation_time
    }
}
