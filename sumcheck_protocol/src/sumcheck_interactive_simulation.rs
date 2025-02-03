use polynomials::multilinear::evaluation_form::{partial_evaluate, MultilinearPolynomial};
use ark_ff::{PrimeField, BigInteger};

pub struct Prover<F: PrimeField> {
    pub initial_polynomial: MultilinearPolynomial<F>,
    pub initial_claimed_sum: F,
}

impl <F: PrimeField>Prover<F> {
    pub fn new() -> Self {
        todo!()
    }

    pub fn prove() {

    }
}

pub struct Verifier<F: PrimeField> {
    pub initial_polynomial: MultilinearPolynomial<F>,
    pub initial_claimed_sum: F,
}

impl <F: PrimeField>Verifier<F> {
    pub fn new() -> Self {
        todo!()
    }

    pub fn verify() {

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