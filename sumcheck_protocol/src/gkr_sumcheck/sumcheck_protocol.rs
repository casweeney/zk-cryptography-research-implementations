use polynomials::composed::composed_multilinear::ComposedPolynomial;
use transcripts::fiat_shamir::{fiat_shamir_transcript::Transcript, interface::FiatShamirTranscriptInterface};
use ark_ff::{PrimeField, BigInteger};

pub struct GKRSumcheck<F: PrimeField> {
    pub polynomial: ComposedPolynomial<F>,
    pub claimed_sum: F
}

pub struct GKRSumcheckProof<F: PrimeField> {
    pub polynomial: ComposedPolynomial<F>,
    pub round_univariate_polynomials: Vec<Vec<F>>
}

impl <F: PrimeField>GKRSumcheck<F> {
    pub fn init(polynomial: ComposedPolynomial<F>) -> Self {
        let claimed_sum = Self::calculate_product_poly_sum(&polynomial);

        Self {
            polynomial,
            claimed_sum
        }
    }

    pub fn calculate_product_poly_sum(polynomial: &ComposedPolynomial<F>) -> F {
        polynomial.multiply_polynomials_element_wise().evaluated_values.iter().sum()
    }

    pub fn prove() -> GKRSumcheckProof<F> {
        let transcript = Transcript::new();

        todo!()
    }

    pub fn verify() -> bool {
        let transcript = Transcript::new();

        todo!()
    }
}