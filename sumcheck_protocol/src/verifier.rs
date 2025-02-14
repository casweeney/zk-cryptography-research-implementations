use transcripts::fiat_shamir::{fiat_shamir_transcript::Transcript, interface::FiatShamirTranscriptInterface};
use crate::prover::{SumcheckProof, field_element_to_bytes};
use ark_ff::PrimeField;
use std::marker::PhantomData;

pub struct Verifier<F: PrimeField> {
    pub transcript: Transcript,
    pub is_initialized: bool,
    _phantom: PhantomData<F>
}

impl <F: PrimeField>Verifier<F> {
    pub fn init() -> Self {
        Self {
            transcript: Transcript::new(),
            is_initialized: true,
            _phantom: PhantomData,
        }
    }

    pub fn verify(&mut self, proof: SumcheckProof<F>) -> bool {
        assert!(self.is_initialized, "Can't verify without init");

        if proof.round_univariate_polynomials.len() != proof.initial_polynomial.number_of_variables() as usize {
            return false;
        }

        let mut current_claim_sum = proof.initial_claimed_sum;

        self.transcript.append(&proof.initial_polynomial.convert_to_bytes());
        self.transcript.append(&field_element_to_bytes(proof.initial_claimed_sum));

        let mut challenges: Vec<F> = Vec::with_capacity(proof.round_univariate_polynomials.len());

        // Loop through the vector of univariate polynomials
        // Generate random challenge for each univariate polynomial
        // Add the evaluation to the current_sum
        // After the end loop, use the vector of challenges to evaluate the main initial polynomial
        // The evaluation of the first initial multilinear polynomial -
        // - should be equal to the sum of the individual univariate polynomial evaluation at the different generated challenges
        for i in 0..proof.round_univariate_polynomials.len() {
            let eval_at_zero = vec![F::zero()];
            let eval_at_one = vec![F::one()];

            if proof.round_univariate_polynomials[i].evaluate(&eval_at_zero) + proof.round_univariate_polynomials[i].evaluate(&eval_at_one) != current_claim_sum {
                return false;
            }

            self.transcript.append(&proof.round_univariate_polynomials[i].convert_to_bytes());

            let challenge: F = self.transcript.random_challenge_as_field_element();
            challenges.push(challenge);

            current_claim_sum = proof.round_univariate_polynomials[i].evaluate(&vec![challenge])
        }

        let final_evaluation = proof.initial_polynomial.evaluate(&challenges);

        // Oracle Check
        final_evaluation == current_claim_sum
    }
}