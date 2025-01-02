# Jaime's Artificial Inteligence and Machine learning Engine

![Crates.io Version](https://img.shields.io/crates/v/jaime)
![Passing tests](https://github.com/jaimegonzalezfabregas/Jaime/actions/workflows/rust.yml/badge.svg)
![docs.rs](https://img.shields.io/docsrs/jaime)


J.a.i.m.e., pronounced as /hɑːɪmɛ/, is a all purpose ergonomic gradient descent engine. It can configure **ANY**\* and **ALL**\*\* models to find the best fit for your dataset. It will magicaly take care of the gradient computations with little effect on your coding style. 

\* not only neuronal

\** derivability conditions apply 

# Concepts and explanation

- Input: For our purposes the input of our Model will be a vector of floating point numbers
- Output: For our purposes the output of our Model will be a vector of floating point numbers
- Dataset: a set of input-output pairs. Jaime will reconfigure the model to aproximate the behabiour described in the dataset.
- Model: a function that maps from input to ouput using a set of configuration parameters that define its behabiour. For our purposes small changes in the parameters should translate to small changes in the behabiour of the function. Examples of suitable models:
    - Polinomial functions: Defined as `y = P_0 * x^0 + P_1 * x^1 + ... + P_n * x^n`. The vector `[x]` will be our input, the vector `[y]` will be our output, The vector `[P_0, P_1, ... ,P_2]` will be our parameter vector. An example of this crate for this precise case can be found [here](https://github.com/jaimegonzalezfabregas/jaime_polinomial)
    - Neuronal networks: In their most basic form they are defined as consecutive matrix multiplications with delinearization steps in between. The classical meaning of parameters, input and output for a NN matches the concepts used in this crate.  An example of this crate for this precise case can be found [here](https://github.com/jaimegonzalezfabregas/jaime_mnist_perceptron)

If you are able to define a model this crate will happily apply [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent) to find some local minumum that aproximates the behabiour defined in the dataset.

# Examples

To make sure this crate was as usable and performant as posible I've also implemented a few exercises that use its functions. 

- [Hello world](https://github.com/jaimegonzalezfabregas/jaime_hello_world)
- [Polinomial deduction](https://github.com/jaimegonzalezfabregas/jaime_polinomial)
- [Voronoi Image Aproximator](https://github.com/jaimegonzalezfabregas/jaime_voronoi_image_aproximator)
- Xor neuronal network (Comming soon)
- [MNIST neuronal network](https://github.com/jaimegonzalezfabregas/jaime_mnist_perceptron)

- **WIP** Asteroid shape deduction
- **WIP** Optical lens designer
- **WIP** Path optimization

If you belive your understanding of what my crate does is clear I encourage you to study the voronoi image aproximator.

# Geeky internal wizardry

If you are a little math savy and know how gradient descent works you may be wondering how am I able to do the partial derivatives for the parameters without knowing beforehand what operations will the model perform. The solution relies on [Forward Mode Automatic Differentiation](https://jameshfisher.com/2024/04/02/automatic-differentiation-with-dual-numbers/) using dual numbers. Jaime will require you to define a generic function that manipulates a vector of float-oids and returns a vector of float-oids. That function will later be instanciated with a custom dual number type, that will allow me hijack the mathematic operations and keep track of the necesary extra data.

Rust, specificaly rust's generics and trait system, is perfect for this task. I can unambiguosly define what a float-oid is to rust as a set of traits that overload operators and other functionality.

After that the only thing remaining is to follow the calculated gradient towards victory.

# Usage Documentation

Comming soon, for now try having a look at the examples.

# Contributing
Yes please.
Make a PR to this repo and I will happily merge it.

# A note on optimization

I heavily used [Samply](https://github.com/mstange/samply) for profiling during this project.
