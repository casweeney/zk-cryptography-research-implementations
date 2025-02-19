use ark_ff::PrimeField;
use circuit::arithmetic_circuit::Circuit;
use sumcheck_protocol::gkr_sumcheck::gkr_sumcheck_protocol::GKRSumcheck;
use polynomials::{
    composed::{product_polynomial, sum_polynomial},
    multilinear::evaluation_form::MultilinearPolynomial
};
use transcripts::fiat_shamir::fiat_shamir_transcript::Transcript;

pub struct Proof<F: PrimeField> {
    claimed_sum: F
}

pub fn prove() {

}

pub fn verify() {

}