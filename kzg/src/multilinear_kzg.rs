use ark_ff::PrimeField;
use ark_ec::{pairing::Pairing, PrimeGroup};
use polynomials::multilinear::evaluation_form::MultilinearPolynomial;
use crate::trusted_setup::TrustedSetup;

pub struct MultilinearKZGProof<F: PrimeField, P: Pairing> {
    pub evaluation: F,
    pub proofs: Vec<P::G1>
}


pub fn commit_to_polynomial<F: PrimeField, P: Pairing>(
    polynomial: MultilinearPolynomial<F>,
    trust_setup: TrustedSetup<P>
) -> P::G1 {
    assert_eq!(polynomial.evaluated_values.len(), trust_setup.g1_powers_of_tau.len(), "Polynomial evaluation must match g1 length");

    let commitment = polynomial.evaluated_values
        .iter()
        .zip(trust_setup.g1_powers_of_tau.iter())
        .map(|(coefficient, power)| power.mul_bigint(coefficient.into_bigint()))
        .sum();

    commitment
}

pub fn prove<F: PrimeField, P: Pairing>(
    polynomial: MultilinearPolynomial<F>,
    trust_setup: TrustedSetup<P>,
    opening_values: &[F]
) -> MultilinearKZGProof<F, P> {

    todo!()
}

pub fn verify<F: PrimeField, P: Pairing>(
    trust_setup: TrustedSetup<P>,
    commitment: &P::G1,
    opening_values: &[F],
    proof: MultilinearKZGProof<F, P>
) -> bool {

    todo!()
}

// This is a helper function to generate random numbers as values of tau, based on the number of variables
pub fn generate_taus(no_of_variables: usize) -> Vec<usize> {
    let mut values_of_tau: Vec<usize> = Vec::with_capacity(no_of_variables);

    todo!()
}