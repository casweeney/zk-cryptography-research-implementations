pub struct UnivariatePolynomial {
    coefficients: Vec<f64>
}

impl UnivariatePolynomial {
    pub fn new(points: Vec<f64>) -> Self {
        Self {
            coefficients: points
        }
    }

    pub fn degree(&self) -> u32 {
        self.coefficients.len() as u32 - 1
    }

    pub fn evaluate(&self, value: f64) -> f64 {
        // using index as the exponent of the coefficients
        let mut index_counter = 0;
        let mut sum = 0.0;
        
        for i in self.coefficients.iter() {
            sum = sum + (i * value.powf(index_counter as f64));

            index_counter = index_counter + 1;
        }

        sum
    }

    pub fn interpolate(x_values: Vec<f64>, y_values: Vec<f64>) -> UnivariatePolynomial {
        todo!()
    }
}

fn lagrange_basis(interpolating_set: Vec<f64>) -> UnivariatePolynomial {
    todo!()
}

fn multiply_polynomials() -> UnivariatePolynomial {
    todo!()
}

fn add_polynomials() -> UnivariatePolynomial {
    todo!()
}