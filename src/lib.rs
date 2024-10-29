#![feature(generic_arg_infer)]


//! # Overview
//! 
//! J.a.i.m.e., pronounced as /hɑːɪmɛ/, is a all purpose ergonomic gradient descent engine. It can configure **ANY** \* and **ALL**\*\* models to find the best fit for your dataset. It will magicaly take care of the gradient computations with little effect on your coding style. 
//! \* not only neuronal
//! \*\* derivability conditions apply 
//! 
//! # Basic example
//! 
//!  ```
//!#![feature(generic_arg_infer)] // this will save a lot of time and make your code much more redeable
//!
//!use std::ops::Mul; // this will allow us to specify the properties of our "float-oid"
//!
//!use jaime::{
//!    simd_arr::dense_simd::DenseSimd,
//!    trainer::{default_param_translator, DataPoint, Trainer},
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
//!    let dataset = vec![
//!        // define the desired behabiour as a dataset
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
//!    let mut trainer: Trainer<_, _, _, _, DenseSimd<_>, _, _, _> =
//!        Trainer::new_dense(direct, direct, default_param_translator, ());
//!
//!    // the function train_step_asintotic_search will step towards the local minimum. When the local minimum is found it will return false and the loop will exit.
//!    while trainer.train_step_asintotic_search::<false, false, _, _>(
//!        &dataset,
//!        &dataset,
//!        dataset.len(),
//!        dataset.len(),
//!    ) {
//!        println!("{:?}", trainer.get_model_params());
//!    }
//!
//!    // At this point the param should be equal to 2, as that best fits our model.
//!    println!("{:?}", trainer.get_model_params());
//!}
//!  ```
//! download and compile this example from [the github repo](https://github.com/jaimegonzalezfabregas/jaime_hello_world)

extern crate indicatif;
extern crate rand_chacha;
extern crate rand;
extern crate rayon;

pub mod dual;
pub mod simd_arr;
pub mod trainer;
