use ark_ff::PrimeField;

// This function will receive a polynomial in it's evaluated form
// That means the polynomial it will receive has already been evaluated over a boolean hypercube
pub fn partial_evaluate(polynomial: Vec<i32>, evaluating_variable: usize, value: i32) -> Vec<i32> {
    let polynomial_size = polynomial.len();
    let expected_polynomial_size = polynomial_size / 2;
    let mut result_polynomial: Vec<i32> = Vec::with_capacity(expected_polynomial_size);

    let mut i = 0;
    let mut j = 0;

    while i < expected_polynomial_size {
        // a b c
        let first_pair_value = polynomial[j];
        let number_of_variables = polynomial.len().ilog2() as usize;
        let power = number_of_variables - 1 - evaluating_variable;
        let second_pair_value = polynomial[j | (1 << power)];

        result_polynomial.push(first_pair_value * (1 - value) + second_pair_value * value);

        i += 1;

        j = if j + 1 % (1 << power) == 0 {
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

    #[test]
    fn test_partial_evaluate() {
        let polynomial = vec![0, 0, 3, 8];

        assert_eq!(partial_evaluate(polynomial.clone(), 0, 6), vec![18, 48]);
        assert_eq!(partial_evaluate(polynomial, 1, 2), vec![0, 13]);
    }
}