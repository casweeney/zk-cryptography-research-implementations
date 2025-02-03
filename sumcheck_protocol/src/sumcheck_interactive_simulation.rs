use polynomials::multilinear::evaluation_form::{partial_evaluate, MultilinearPolynomial};
use ark_ff::{PrimeField, BigInteger};

pub struct Prover<F: PrimeField> {
    pub initial_polynomial: MultilinearPolynomial<F>,
    pub initial_claimed_sum: F,
    pub current_polynomial: Vec<F>,
    pub round: usize
}

impl <F: PrimeField>Prover<F> {
    pub fn new(evaluated_values: Vec<F>) -> Self {
        Self {
            initial_polynomial: MultilinearPolynomial::new(evaluated_values.clone()),
            initial_claimed_sum: evaluated_values.iter().sum(),
            current_polynomial: evaluated_values,
            round: 0
        }
    }

    pub fn prove(&mut self, random_challenge: F) -> (F, Vec<F>) {
        let univariate = split_polynomial_and_sum_each(&self.current_polynomial);

        if self.round == 0 {
            self.round += 1;
            (self.initial_claimed_sum, univariate)
        } else {
            self.current_polynomial = partial_evaluate(&self.current_polynomial, 0, random_challenge);
            let new_claimed_sum = self.current_polynomial.iter().sum();

            self.round += 1;

            (new_claimed_sum, split_polynomial_and_sum_each(&self.current_polynomial))
        }
    }
}

pub fn split_polynomial_and_sum_each<F: PrimeField>(polynomial_evaluated_values: &Vec<F>) -> Vec<F> {
    let mut univariate_polynomial: Vec<F> = Vec::with_capacity(2);

    let mid = polynomial_evaluated_values.len() / 2;
    let (left, right) = polynomial_evaluated_values.split_at(mid);

    let left_sum: F = left.iter().sum();
    let right_sum: F = right.iter().sum();

    univariate_polynomial.push(left_sum);
    univariate_polynomial.push(right_sum);

    univariate_polynomial
}

pub struct Verifier<F: PrimeField> {
    pub initial_polynomial: MultilinearPolynomial<F>,
    pub initial_claimed_sum: F
}

impl <F: PrimeField>Verifier<F> {
    pub fn new(evaluated_values: Vec<F>) -> Self {
        Self {
            initial_polynomial: MultilinearPolynomial::new(evaluated_values.clone()),
            initial_claimed_sum: evaluated_values.iter().sum()
        }
    }

    pub fn verify(claimed_sum: F, univariate_polynomial: Vec<F>) {
        
    }

    pub fn oracle_check() {

    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_sumcheck_interactive_simulation() {

    }
}