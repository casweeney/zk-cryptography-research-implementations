use polynomials::multilinear::evaluation_form::{partial_evaluate, MultilinearPolynomial};
use ark_ff::PrimeField;


/////// Prover Implementation Starts Here ///////////
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
/////// Prover Implementation Ends Here ///////////


/////// Verifier Implementation Starts Here //////////
pub struct Verifier<F: PrimeField> {
    pub initial_polynomial: MultilinearPolynomial<F>,
    pub current_claimed_sum: F,
    pub challenges: Vec<F>,
    pub round: usize
}

impl <F: PrimeField>Verifier<F> {
    pub fn new(evaluated_values: Vec<F>) -> Self {
        Self {
            initial_polynomial: MultilinearPolynomial::new(evaluated_values.clone()),
            current_claimed_sum: F::zero(),
            challenges: Vec::new(),
            round: 0
        }
    }

    pub fn verify(&mut self, claimed_sum: F, univariate_polynomial: Vec<F>) -> bool {
        if univariate_polynomial.len() != 2 {
            return false;
        }

        let actual_univariate_polynomial = MultilinearPolynomial::new(univariate_polynomial);

        let eval_at_zero = actual_univariate_polynomial.evaluate(vec![F::zero()]);
        let eval_at_one = actual_univariate_polynomial.evaluate(vec![F::one()]);

        if eval_at_zero + eval_at_one != claimed_sum {
            return false;
        }

        self.current_claimed_sum = claimed_sum;

        return true;
    }

    pub fn generate_challenge(&mut self) -> F {
        let mut rng = rand::thread_rng();
        let challenge = F::rand(&mut rng);

        self.challenges.push(challenge);

        challenge
    }

    pub fn oracle_check(&self) -> bool {
        self.current_claimed_sum == self.initial_polynomial.evaluate(self.challenges.clone())
    }
}
/////// Verifier Implementation Ends Here //////////


/////// Interactive Sumcheck Protocol Simulation Starts Here //////////
#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;

    #[test]
    fn test_sumcheck_interactive_simulation() {
        let values = vec![
            Fr::from(0),
            Fr::from(0),
            Fr::from(2),
            Fr::from(7),
            Fr::from(3),
            Fr::from(3),
            Fr::from(6),
            Fr::from(11),
        ];
        let mut prover = Prover::new(values.clone());
        let mut verifier = Verifier::new(values.clone());

        // First round - no challenge needed for the prover
        let (claimed_sum, univariate) = prover.prove(Fr::from(0)); // Zero challenge not used in first round
        println!("Round 0 - Claimed sum: {:?}, Univariate: {:?}", claimed_sum, univariate);
        assert!(verifier.verify(claimed_sum, univariate));

        let no_of_varibles = values.len().ilog2();

        // Subsequent rounds
        for i in 0..no_of_varibles {
            let challenge = verifier.generate_challenge();
            let (claimed_sum, univariate) = prover.prove(challenge);

            println!("Round {} - Challenge: {:?}, Claimed sum: {:?}, Univariate: {:?}", i+1, challenge, claimed_sum, univariate);

            assert!(verifier.verify(claimed_sum, univariate));
        }
        
        assert!(verifier.oracle_check());
    }
}

// Summary of the test:
// The first eval is not where the round starts. 
// The first eval is just the prover sending the initial claimed sum of the original polynomial and a univariate polynomial to prove the sum.

// A round starts when the verifier generates a random challenge, 
// and the random challenge generation will be 3 which is the number of variables, anything less than that will fail.