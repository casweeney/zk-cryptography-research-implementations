use ark_ff::PrimeField;

#[derive(Clone, Debug)]
pub struct DenseUnivariatePolynomial<F: PrimeField> {
    pub coefficients: Vec<F>,
}

impl<F: PrimeField> DenseUnivariatePolynomial<F> {
    pub fn new(coefficients: &Vec<F>) -> Self {
        Self {
            coefficients: coefficients.to_vec(),
        }
    }

    pub fn degree(&self) -> u32 {
        self.coefficients.len() as u32 - 1
    }

    /// In this evaluate function, we are manually tracking and updating the index: using the index_counter variable
    /// This is a dense representation of a univariate polynomial: where the polynomial is blown up
    /// Given a polynomial => 2x^2 + 3x + 1 => 1 + 3x + 2x^2 => the dense representation is => coefficient: [1, 3, 2], power: [0, 1, 2]
    /// We only take in the coefficients, then we use the loop index to represent the powers.
    pub fn evaluate_(&self, value: F) -> F {
        // using index as the exponent of the coefficients
        let mut index_counter = 0;
        let mut result = F::zero();

        for coefficient in self.coefficients.iter() {
            result += *coefficient * value.pow([index_counter]);

            index_counter += 1;
        }

        result
    }

    // Here we are getting the loop index from the enumerate() method
    pub fn evaluate_advanced(&self, value: F) -> F {
        let mut result = F::zero();

        for (exp, coeff) in self.coefficients.iter().enumerate() {
            result += *coeff * value.pow([exp as u64]);
        }

        result
    }

    /// The is the lastly implemented evaluate function, it is considered to be more optimized and faster
    /// because the two implementations above uses pow([index_counter]) method which requires creating an array for each exponentiation
    /// and potentially does more complex calculation internally.
    /// In this implementation we perform a single multiplication per term to calculate the next power, which avoids overhead creation of arrays
    /// and calling the pow() method
    /// Reduces the total number of multiplication from O(n^2) to O(n)
    /// Given a polynomial => 2x^2 + 3x + 1 => 1 + 3x + 2x^2 => the dense representation is => coefficient: [1, 3, 2], power: [0, 1, 2]
    /// If value = 2
    /// (1 * 1) + (3 * (1 * 2)) + (2 * (1 * 2 * 2)) => Notice that 2 repeats based on the power: [0, 1, 2]
    pub fn evaluate(&self, value: F) -> F {
        let mut result = F::zero();
        let mut current_power = F::one();

        for coefficient in self.coefficients.iter() {
            result += *coefficient * current_power;

            current_power = current_power * value;
        }

        result
    }

    /// The interpolate function can be implemented in two ways
    /// We can either take in two arguments: x_values and y_values,
    /// or we can generate our x_values from the given y_values
    /// Passing x_values will make our function run faster because it will avoid the first loop to get x_values
    pub fn lagrange_interpolate(
        x_values: &Vec<F>,
        y_values: &Vec<F>,
    ) -> DenseUnivariatePolynomial<F> {
        let mut final_interpolated_polynomial = vec![F::zero()]; //any value added to zero, gives you that value
                                                                 // let mut x_values: Vec<F> = Vec::new();

        // Generating x_values from the given y_values
        // for (x, _y) in y_values.iter().enumerate() {
        //     x_values.push(F::from(x as u64));
        // }

        // Using the generated x_values to derive the lagrange basis at each x_point and corresponding y_point,
        // passing in the interpolating set, which is the x_values.
        // Lastly using add_polynomials to sum up each polynomial derived from lagrange basis
        for (index, x_value) in x_values.iter().enumerate() {
            let current_polynomial = lagrange_basis(&y_values[index], x_value, &x_values);
            final_interpolated_polynomial =
                add_polynomials(final_interpolated_polynomial, current_polynomial)
        }

        DenseUnivariatePolynomial {
            coefficients: final_interpolated_polynomial,
        }
    }
}

fn lagrange_basis<F: PrimeField>(
    y_point: &F,
    focus_x_point: &F,
    interpolating_set: &Vec<F>,
) -> Vec<F> {
    // numerator
    let mut numerator = vec![F::one()];

    // (x-1)(x-2) => [1, -1][1, -2] => Reverse it based on index => [-1, 1][-2, 1]
    // Notice that in the reversed for, 1 is always constant, and the first values in the interpolating set are always negative
    for x in interpolating_set.iter() {
        if *x != *focus_x_point {
            numerator = multiply_polynomials(numerator, vec![-*x, F::one()]);
        }
    }

    // denominator
    let univariate_poly: DenseUnivariatePolynomial<F> =
        DenseUnivariatePolynomial::new(&numerator);
    let denominator = univariate_poly.evaluate(*focus_x_point);

    // numerator/denominator is the same this as (1/denominator) * numerator
    // y_point * 1 = y_point
    let interpolated_polynomial = scalar_mul(*y_point / denominator, numerator);

    interpolated_polynomial
}

fn scalar_mul<F: PrimeField>(scalar: F, polynomial: Vec<F>) -> Vec<F> {
    let mut result_polynomial: Vec<F> = Vec::new();

    for coeff in polynomial.iter() {
        result_polynomial.push(scalar * coeff);
    }

    result_polynomial
}

/// We are using index as the power/exponent
/// Then we use the sum of the powers of the two polynomials that are multiplying each other
/// as the index of resulting polynomial = polynomial_product[pow1 + pow2] = coeff1 * coeff2
pub fn multiply_polynomials<F: PrimeField>(left: Vec<F>, right: Vec<F>) -> Vec<F> {
    //  0 1 1     0 1 -> powers(exponents)
    //  | | |     | |
    // [5,0,2] * [6,2] -> Coefficients

    let mut polynomial_product = vec![F::zero(); (left.len() + right.len()) - 1];

    // for left_index in 0..left.len() {
    //     for right_index in 0..right.len() {
    //         polynomial_product[left_index + right_index] += left[left_index] * right[right_index];
    //     }
    // }

    for (left_index, left_coeff) in left.iter().enumerate() {
        for (right_index, right_coeff) in right.iter().enumerate() {
            polynomial_product[left_index + right_index] += *left_coeff * *right_coeff;
        }
    }

    polynomial_product
}

pub fn add_polynomials<F: PrimeField>(left: Vec<F>, right: Vec<F>) -> Vec<F> {
    let mut summed_polynomial: Vec<F> = Vec::new();

    let (larger_polynomial, smaller_polynomial) = if left.len() > right.len() {
        (left, right)
    } else {
        (right, left)
    };

    for (exp, coeff) in larger_polynomial.iter().enumerate() {
        if exp < smaller_polynomial.len() {
            summed_polynomial.push(*coeff + smaller_polynomial[exp]);
        } else {
            summed_polynomial.push(*coeff);
        }
    }

    summed_polynomial
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fq;

    fn test_setup() -> DenseUnivariatePolynomial<Fq> {
        let set_of_points = vec![
            Fq::from(0),
            Fq::from(0),
            Fq::from(2),
            Fq::from(0),
            Fq::from(0),
            Fq::from(0),
            Fq::from(0),
            Fq::from(3),
        ];
        let polynomial = DenseUnivariatePolynomial::new(&set_of_points);

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
        let evaluation_value = Fq::from(2);

        assert_eq!(polynomial.evaluate(evaluation_value), Fq::from(392));
    }

    #[test]
    fn test_evaluation_advanced() {
        let polynomial = test_setup();
        let evaluation_value = Fq::from(2);

        assert_eq!(
            polynomial.evaluate_advanced(evaluation_value),
            Fq::from(392)
        );
    }

    #[test]
    fn test_add_polynomials() {
        let p1 = vec![Fq::from(5), Fq::from(2), Fq::from(5)];
        let p2 = vec![Fq::from(2), Fq::from(1), Fq::from(8), Fq::from(10)];

        assert_eq!(
            add_polynomials(p1, p2),
            vec![Fq::from(7), Fq::from(3), Fq::from(13), Fq::from(10)]
        );
    }

    #[test]
    fn test_multiply_polynomials() {
        let p1 = vec![Fq::from(5), Fq::from(0), Fq::from(2)];
        let p2 = vec![Fq::from(6), Fq::from(2)];

        assert_eq!(
            multiply_polynomials(p1, p2),
            vec![Fq::from(30), Fq::from(10), Fq::from(12), Fq::from(4)]
        );
    }

    #[test]
    fn test_lagrange_interpolate() {
        let x_values = vec![Fq::from(0), Fq::from(1), Fq::from(2)];
        let y_values = vec![Fq::from(2), Fq::from(4), Fq::from(10)];

        assert_eq!(
            DenseUnivariatePolynomial::lagrange_interpolate(&x_values, &y_values).coefficients,
            vec![Fq::from(2), Fq::from(0), Fq::from(2)]
        );
    }
}
