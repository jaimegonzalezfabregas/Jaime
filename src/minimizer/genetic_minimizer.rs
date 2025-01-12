use std::array;
use std::time::Instant;

use rand::Rng;

use super::Minimizer;

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

    pub fn calculate_cost<ExtraData: Clone + Sync, F: Fn(&[f32; P], &ExtraData) -> f32 + Sync>(
        &mut self,
        cost: F,
        extra_data: &ExtraData,
    ) {
        if let None = self.cost {
            self.cost = Some(cost(&self.parameters, extra_data))
        }
    }
}

/// The GeneticTrainer struct manages the training lifecycle using a evolutionary genetic algorithm
#[derive(Clone)]
pub struct GeneticMinimizer<
    const P: usize,
    const GENERATION_SURVIVORS: usize,
    const GROUTH_FACTOR: usize,
    ExtraData: Sync + Clone,
    F: Fn(&[f32; P], &ExtraData) -> f32 + Sync + Clone,
    ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P] + Clone,
> {
    cost: F,
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
        const GENERATION_SURVIVORS: usize,
        const GROUTH_FACTOR: usize,
        ExtraData: Sync + Clone,
        F: Fn(&[f32; P], &ExtraData) -> f32 + Sync + Clone,
        ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P] + Clone,
    > GeneticMinimizer<P, GENERATION_SURVIVORS, GROUTH_FACTOR, ExtraData, F, ParamTranslate>
{
    pub fn new(
        trainable: F,
        param_translator: ParamTranslate,
        extra_data: ExtraData,
        cost_stagnation_threshold: f32,
        max_cost_stagnation_time: usize,
    ) -> Self {
        Self {
            cost: trainable,
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
        const GENERATION_SURVIVORS: usize,
        const GROUTH_FACTOR: usize,
        ExtraData: Sync + Clone,
        F: Fn(&[f32; P], &ExtraData) -> f32 + Sync + Clone,
        ParamTranslate: Fn(&[f32; P], &[f32; P]) -> [f32; P] + Clone,
    > Minimizer<P>
    for GeneticMinimizer<P, GENERATION_SURVIVORS, GROUTH_FACTOR, ExtraData, F, ParamTranslate>
{
    fn get_last_cost(&self) -> Option<f32> {
        self.last_cost
    }

    fn train_step<const VERBOSE: bool>(&mut self, learning_rate: f32) {
        let t0 = Instant::now();

        let mut full_population: Vec<Agent<P>> = vec![];

        for i in 0..GENERATION_SURVIVORS {
            full_population.push(self.population[i].clone());

            for j in 0..(GROUTH_FACTOR) {
                full_population.push(
                    self.population[i].mutate(&self.param_translator, learning_rate / j as f32),
                )
            }
        }

        full_population
            .iter_mut()
            .for_each(|e| e.calculate_cost::<_, _>(&self.cost, &self.extra_data));

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

    fn get_model_params(&self) -> [f32; P] {
        self.population[0].parameters
    }

    fn set_model_params(&mut self, parameters: [f32; P]) {
        self.population[0] = Agent {
            parameters,
            cost: None,
        };

        for i in 1..(GENERATION_SURVIVORS - 1) {
            self.population[i] = self.population[0].mutate(&self.param_translator, 1.);
        }
    }

    fn found_local_minima(&self) -> bool {
        self.max_cost_stagnation_time < self.cost_stagnation_time
    }
}
