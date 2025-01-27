use ark_ff::{PrimeField, BigInteger};

// This implementation of Multi linear interpolation uses an evaluation over the boolean hypercube
// then the values from the boolean hypercube evaluation is used as the polynomial
// which will be evaluated at a given variable values using partial evaluation
#[derive(Debug, Clone, PartialEq)]
pub struct MultilinearPolynomial<F: PrimeField> {
    evaluated_values: Vec<F>
}

impl <F: PrimeField>MultilinearPolynomial<F> {
    pub fn new(evaluated_values: Vec<F>) -> Self {
        Self {
            evaluated_values
        }
    }

    // The evaluate function calls the partial evaluate multiple times
    pub fn evaluate(&self, values: Vec<F>) -> F {
        let mut r_polynomial = self.evaluated_values.clone();
        let expected_number_of_partial_eval = values.len();

        let mut i = 0;

        while i < expected_number_of_partial_eval {
            r_polynomial = partial_evaluate(&r_polynomial, 0, values[i]);
            i += 1;
        }

        r_polynomial[0]
    }

    pub fn convert_to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        for value in &self.evaluated_values {
            bytes.extend(value.into_bigint().to_bytes_be());
        }

        bytes
    }
}

// This function will receive a polynomial in it's evaluated form
// That means the polynomial it will receive has already been evaluated over a boolean hypercube
pub fn partial_evaluate<F: PrimeField>(polynomial: &Vec<F>, evaluating_variable: usize, value: F) -> Vec<F> {
    let polynomial_size = polynomial.len();
    let expected_polynomial_size = polynomial_size / 2;
    let mut result_polynomial: Vec<F> = Vec::with_capacity(expected_polynomial_size);

    let mut i = 0;
    let mut j = 0;

    while i < expected_polynomial_size {
        let first_pair_value = polynomial[j]; // y1

        // since the number of boolean hypercube evaluations for a polynomial with n number of variable is 2^n
        // The number of variables, when given the evaluations: n = log2(polynomial length)
        let number_of_variables = polynomial.len().ilog2() as usize;

        // 0 1 2 => evaluating variable for a = 0, b = 1, c = 2
        // | | |
        // a b c
        // using evaluating_variable as variable index in boolean hypercube
        let power = number_of_variables - 1 - evaluating_variable;
        
        let second_pair_value = polynomial[j | (1 << power)]; // y2

        // using the formula: y1 + r(y2 - y1)
        // y1 => first_pair_value
        // y2 => second_pair_value
        // r => value
        result_polynomial.push(first_pair_value + (value * (second_pair_value - first_pair_value)));

        i += 1;

        // After pairing, we need to determine what our next y1 value, which will be used for pairing to get a y2
        // To get the next y1, we first add 1 to the previous y1 and check if the modulo with the 2^power of the variable we are evaluating at is zero
        // ie: (previous_y1 + 1) % 2^power
        // If it is zero we jump by (previous_y1 + 1 + 2^power)
        // If it is not zero, we jump by adding 1: (previous_y1 + 1)
        j = if (j + 1) % (1 << power) == 0 {
            j + 1 + (1 << power)
        } else {
            j + 1
        }
    }

    result_polynomial
}


#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fq;

    #[test]
    fn test_partial_evaluate() {
        let polynomial = vec![Fq::from(0), Fq::from(0), Fq::from(3), Fq::from(8)];

        assert_eq!(partial_evaluate(&polynomial, 0, Fq::from(6)), vec![Fq::from(18), Fq::from(48)]);
        assert_eq!(partial_evaluate(&polynomial, 1, Fq::from(2)), vec![Fq::from(0), Fq::from(13)]);

        let small_polynomial = vec![Fq::from(18), Fq::from(48)];
        assert_eq!(partial_evaluate(&small_polynomial, 0, Fq::from(2)), vec![Fq::from(78)]);

        let bigger_polynomial = vec![Fq::from(0), Fq::from(0), Fq::from(0), Fq::from(3), Fq::from(0), Fq::from(0), Fq::from(2), Fq::from(5)];
        assert_eq!(partial_evaluate(&bigger_polynomial, 2, Fq::from(3)), vec![Fq::from(0), Fq::from(9), Fq::from(0), Fq::from(11)]);
    }

    #[test]
    fn test_evaluate() {
        let evaluated_values = vec![Fq::from(0), Fq::from(0), Fq::from(3), Fq::from(8)];
        let polynomial = MultilinearPolynomial::new(evaluated_values);
        let values = vec![Fq::from(6), Fq::from(2)];

        assert_eq!(polynomial.evaluate(values), Fq::from(78));
    }
}