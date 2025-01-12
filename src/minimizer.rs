use std::fs::OpenOptions;
use std::io::{self, BufRead, Write};

pub mod adam_minimizer;
pub mod asymptotic_gradient_descent_minimizer;
pub mod genetic_minimizer;

/// A data point holds the desired output for a given input. A colection of datapoints is a dataset. A dataset defines the desired behabiour of the trainable model.

pub trait Minimizer<const P: usize> {
    /// Will return the last computed cost (if any has been computed yet)
    fn get_last_cost(&self) -> Option<f32>;

    /// Retrieve the parameters at any point during the training process
    fn get_model_params(&self) -> [f32; P];

    /// Set the parameters at any point during the training process
    fn set_model_params(&mut self, parameters: [f32; P]);

    fn train_step<const VERBOSE: bool, >(&mut self, learning_rate: f32);

    fn found_local_minima(&self) -> bool;

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
