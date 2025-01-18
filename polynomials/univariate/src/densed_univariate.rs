pub struct DensedUnivariatePolynomial {
    coefficients: Vec<f64>
}

impl DensedUnivariatePolynomial {
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
        let mut result = 0.0;
        
        for coeff in self.coefficients.iter() {
            result += coeff * value.powf(index_counter as f64);

            index_counter += 1;
        }

        result
    }

    pub fn evaluate_advanced(&self, value: f64) -> f64 {
        let mut result = 0.0;

        for (exp, coeff) in self.coefficients
            .iter()
            .enumerate() {
                result += coeff * value.powf(exp as f64);
            }

        result
    }

    pub fn interpolate(x_values: Vec<f64>, y_values: Vec<f64>) -> DensedUnivariatePolynomial {
        todo!()
    }
}

fn lagrange_basis(interpolating_set: Vec<f64>) -> DensedUnivariatePolynomial {
    todo!()
}

fn multiply_polynomials() -> Vec<f64> {
    todo!()
}

pub fn add_polynomials(left: Vec<f64>, right: Vec<f64>) -> Vec<f64> {
    let mut summed_polynomial: Vec<f64> = Vec::new();

    let (larger_polynomial, smaller_polynomial) = if left.len() > right.len() {
        (left, right)
    } else {
        (right, left)
    };

    for (exp, coeff) in larger_polynomial.iter().enumerate() {
        if exp < smaller_polynomial.len() {
            summed_polynomial.push(coeff + smaller_polynomial[exp]);
        } else {
            summed_polynomial.push(*coeff);
        }
    }

    summed_polynomial
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_setup() -> DensedUnivariatePolynomial {
        let set_of_points = vec![0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0];
        let polynomial = DensedUnivariatePolynomial::new(set_of_points.clone());

        polynomial
    }

    #[test]
    fn test_degree() {
        let polynomial = test_setup();
        assert_eq!(polynomial.degree(), 7);
    }

    #[test]
    fn test_evaluation() {
        let polynomial = test_setup();
        let evaluation_value = 2.0;

        assert_eq!(polynomial.evaluate(evaluation_value), 392.0);
    }

    #[test]
    fn test_polynomials_addition() {
        let p1 = vec![5.0, 2.0, 5.0];
        let p2 = vec![2.0, 1.0, 8.0, 10.0];

        assert_eq!(add_polynomials(p1, p2), vec![7.0, 3.0, 13.0, 10.0]);
    }
}