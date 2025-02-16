use polynomials::multilinear::evaluation_form::MultilinearPolynomial;
use transcripts::fiat_shamir::{fiat_shamir_transcript::Transcript, interface::FiatShamirTranscriptInterface};
use ark_ff::{PrimeField, BigInteger};

pub struct GKRSumcheck<F: PrimeField> {
    pub polynomial: MultilinearPolynomial<F>,
    pub claimed_sum: F
}

pub struct GKRSumcheckProof<F: PrimeField> {
    pub polynomial: MultilinearPolynomial<F>,
    pub round_univariate_polynomials: Vec<Vec<F>>
}

impl <F: PrimeField>GKRSumcheck<F> {
    pub fn init(polynomial: MultilinearPolynomial<F>) -> Self {
        Self {
            polynomial,
            claimed_sum: Default::default()
        }
    }

    pub fn prove() -> GKRSumcheckProof<F> {
        let transcript = Transcript::new();

        todo!()
    }

    pub fn verify(&self, proof: &MultilinearPolynomial<F>) -> bool {
        let transcript = Transcript::new();

        todo!()
    }
}