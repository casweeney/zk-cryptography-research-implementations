use ark_ff::PrimeField;

struct MultilinearPolynomial {
    evaluated_values: Vec<i32>,
    number_of_variables: usize
}

impl MultilinearPolynomial {
    pub fn new(evaluated_values: Vec<i32>) -> Self {
        let number_of_variables = evaluated_values.len().ilog2() as usize;

        Self {
            evaluated_values,
            number_of_variables
        }
    }
}

// The evaluate function calls the partial evaluate multiple times
pub fn evaluate(polynomial: Vec<i32>, values: Vec<i32>) -> i32 {
    let mut r_polynomial = polynomial.clone();
    let expected_number_of_partial_eval = values.len();

    let mut i = 0;

    while i < expected_number_of_partial_eval {
        r_polynomial = partial_evaluate(&r_polynomial, 0, values[i]);
        i += 1;
    }

    r_polynomial[0]
}

// This function will receive a polynomial in it's evaluated form
// That means the polynomial it will receive has already been evaluated over a boolean hypercube
pub fn partial_evaluate(polynomial: &Vec<i32>, evaluating_variable: usize, value: i32) -> Vec<i32> {
    let polynomial_size = polynomial.len();
    let expected_polynomial_size = polynomial_size / 2;
    let mut result_polynomial: Vec<i32> = Vec::with_capacity(expected_polynomial_size);

    let mut i = 0;
    let mut j = 0;

    while i < expected_polynomial_size {
        let first_pair_value = polynomial[j];

        // since the number of boolean hypercube evaluations for a polynomial with n number of variable is 2^n
        // The number of variables, when given the evaluations: n = log2(polynomial length)
        let number_of_variables = polynomial.len().ilog2() as usize;

        // 0 1 2 => evaluating variable for a = 0, b = 1, c = 2
        // | | |
        // a b c
        // using evaluating_variable as variable index in boolean hypercube
        let power = number_of_variables - 1 - evaluating_variable;
        let second_pair_value = polynomial[j | (1 << power)];

        // using the formula: y1 + r(y2 - y1)
        result_polynomial.push(first_pair_value + ((value * second_pair_value) - (value * first_pair_value)) );

        // A shorter way to represent the above evaluation
        // result_polynomial.push(first_pair_value * (1 - value) + second_pair_value * value);

        i += 1;

        if j + 1 % (1 << power) == 0 {
            j = j + 1 + (1 << power)
        } else {
            j = j + 1
        }
    }

    result_polynomial
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partial_evaluate() {
        let polynomial = vec![0, 0, 3, 8];

        assert_eq!(partial_evaluate(&polynomial, 0, 6), vec![18, 48]);
        assert_eq!(partial_evaluate(&polynomial, 1, 2), vec![0, 13]);

        let small_polynomial = vec![18, 48];
        assert_eq!(partial_evaluate(&small_polynomial, 0, 2), vec![78]);
    }

    #[test]
    fn test_evaluate() {
        let polynomial = vec![0, 0, 3, 8];
        let values = vec![6, 2];

        assert_eq!(evaluate(polynomial, values), 78);
    }
}