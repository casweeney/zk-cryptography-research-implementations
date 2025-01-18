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

    pub fn interpolate(y_values: Vec<f64>) -> DensedUnivariatePolynomial {
        let mut final_interpolated_polynomial = vec![0.0];
        let mut x_values: Vec<f64> = Vec::new();

        for (x, _y) in y_values.iter().enumerate() {
            x_values.push(x as f64);
        }
        
        for (index, x_value) in x_values.iter().enumerate() {
            let current_polynomial = lagrange_basis(y_values[index], *x_value, x_values.clone());
            final_interpolated_polynomial = add_polynomials(final_interpolated_polynomial, current_polynomial)
        }

        DensedUnivariatePolynomial {
            coefficients: final_interpolated_polynomial
        }
    }
}

fn lagrange_basis(y_point: f64, x_point: f64, interpolating_set: Vec<f64>) -> Vec<f64> {
    todo!()
}

pub fn multiply_polynomials(left: Vec<f64>, right: Vec<f64>) -> Vec<f64> {
    let mut polynomial_product = vec![0.0; (left.len() + right.len()) - 1];

    // We are using index as the power
    // for left_index in 0..left.len() {
    //     for right_index in 0..right.len() {
    //         polynomial_product[left_index + right_index] += left[left_index] * right[right_index];
    //     }
    // }

    for (left_index, left_coeff) in left.iter().enumerate(){
        for (right_index, right_coeff) in right.iter().enumerate() {
            polynomial_product[left_index + right_index] += left_coeff * right_coeff;
        }
    }

    polynomial_product
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
    fn test_add_polynomials() {
        let p1 = vec![5.0, 2.0, 5.0];
        let p2 = vec![2.0, 1.0, 8.0, 10.0];

        assert_eq!(add_polynomials(p1, p2), vec![7.0, 3.0, 13.0, 10.0]);
    }

    #[test]
    fn test_multiply_polynomials() {
        let p1 = vec![5.0, 0.0, 2.0];
        let p2 = vec![6.0, 2.0];

        assert_eq!(multiply_polynomials(p1, p2), vec![30.0, 10.0, 12.0, 4.0]);
    }
}