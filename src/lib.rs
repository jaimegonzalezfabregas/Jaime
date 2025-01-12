#![feature(generic_arg_infer)]
#![feature(test)]

//! # Overview
//!
//! J.a.i.m.e., pronounced as /hɑːɪmɛ/, is a all purpose ergonomic gradient descent engine. It can configure **ANY** \* and **ALL**\*\* models to find the best fit for your dataset. It will magicaly take care of the gradient computations with little effect on your coding style.
//!
//! \* not only neuronal
//!
//! \*\* derivability conditions apply
//!
//! # Basic example
//!
//!  ```
//!#![feature(generic_arg_infer)] // this will save a lot of time and make your code much more readable
//!
//!use std::ops::Mul; // this will allow us to specify the properties of our "float-oid"
//!
//!use jaime::trainer::{
//!    asymptotic_gradient_descent_trainer::AsymptoticGradientDescentTrainer,
//!    default_param_translator, DataPoint, Trainer,
//!};
//!
//!// this is the model, Y = X*P, where P is the parameter, X the input and Y the output
//!fn direct<N: Clone + Mul<N, Output = N> + From<f32>>(
//!    parameters: &[N; 1],
//!    input: &[f32; 1],
//!    _: &(),
//!) -> [N; 1] {
//!    [parameters[0].clone() * N::from(input[0])]
//!}
//!
//!fn main() {
//!    // define the desired behabiour as a dataset
//!    let dataset = vec![
//!        DataPoint {
//!            input: [1.],
//!            output: [2.],
//!        },
//!        DataPoint {
//!            input: [2.],
//!            output: [4.],
//!        },
//!        DataPoint {
//!            input: [4.],
//!            output: [8.],
//!        },
//!    ];
//!    // initialize the trainer, this struct will store the parameters and nudge them down the gradient
//!    let mut trainer =
//!        AsymptoticGradientDescentTrainer::new_dense(direct, direct, default_param_translator, ());
//!    // the function train_step_asintotic_search will step towards the local minimum. When the local minimum is found it will return false and the loop will exit.
//!    while !trainer.found_local_minima() {
//!        trainer.train_step::<false, false, _, _>(
//!            &dataset,
//!            &dataset,
//!            dataset.len(),
//!            dataset.len(),
//!            1.,
//!        );
//!
//!        println!("{:?}", trainer.get_model_params());
//!    }
//!    // At this point the param should be equal to 2, as that best fits our model.
//!    println!("{:?}", trainer.get_model_params());
//!}
//!
//!  ```
//! download and compile this example from [the github repo](https://github.com/jaimegonzalezfabregas/jaime_hello_world)
//!
//! # Further reading
//! A deeper explanation of the usage of this crate can be found in the Traier struct [documentation](https://docs.rs/jaime/latest/jaime/trainer/struct.Trainer.html)

extern crate indicatif;
extern crate rand;
extern crate rand_chacha;
extern crate rayon;
mod tests;

/// INTERNAL forward mode automatic derivation engine.
pub mod dual;
/// INTERNAL Implementation of arrays of floats that can apply the same operation to all the elements at the same time
pub mod simd_arr;
/// User facing gradient descent implementation
pub mod trainer;
pub mod minimizer;

/// for medium to big dense 
pub fn set_rayon_stack() {
    rayon::ThreadPoolBuilder::new()
        .stack_size(1 * 1024 * 1024 * 1024)
        .build_global()
        .unwrap();
}
