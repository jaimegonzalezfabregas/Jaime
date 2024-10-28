use std::{
    fmt::Debug,
    ops::{Index, IndexMut},
};

pub mod dense_simd;
pub mod hybrid_simd;
mod sparse_simd;

pub trait SimdArr<const S: usize>:
    Debug + Sized + Send + Sync + Index<usize, Output = f32> + IndexMut<usize, Output = f32> + Clone
{
    fn new_from_array(data: [f32; S]) -> Self;

    fn new_from_value_and_pos(val: f32, pos: usize) -> Self;

    fn zero() -> Self;

    fn neg(&mut self); 

    fn to_array(&self) -> [f32; S];

    fn acumulate(&mut self, rhs: &Self);

    fn multiply(&mut self, rhs: f32);

    fn check_nan(&self);
}