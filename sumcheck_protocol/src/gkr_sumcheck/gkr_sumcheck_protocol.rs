use polynomials::univariate::densed_univariate::DensedUnivariatePolynomial;
use polynomials::composed::sum_polynomial::SumPolynomial;
use transcripts::fiat_shamir::{fiat_shamir_transcript::Transcript, interface::FiatShamirTranscriptInterface};
use ark_ff::{PrimeField, BigInteger};

pub struct GKRSumcheck<F: PrimeField> {
    pub sum_polynomial: SumPolynomial<F>,
    pub number_of_variables: u32,
}

// no need init poly
// prover needs to return challenges
// Verify will use

pub struct GKRSumcheckProverProof<F: PrimeField> {
    pub claimed_sum: F,
    pub round_univariate_polynomials: Vec<Vec<F>>,
    pub random_challenges: Vec<F>,
    pub degree: usize
}

pub struct GKRSumcheckVerifierProof<F: PrimeField> {
    pub is_proof_valid: bool,
    pub random_challenges: Vec<F>,
}

impl <F: PrimeField>GKRSumcheck<F> {
    pub fn init(sum_poly: SumPolynomial<F>) -> Self {
        let num_variables = sum_poly.number_of_variables();

        Self {
            sum_polynomial: sum_poly,
            number_of_variables: num_variables
        }
    }

    pub fn prove(&mut self, claimed_sum: F) -> GKRSumcheckProverProof<F> {
        let mut transcript = Transcript::new();
        let mut round_univariate_polynomials = Vec::new();
        let mut random_challenges = Vec::with_capacity(self.number_of_variables as usize);
        let mut current_polynomial = self.sum_polynomial.clone();

        transcript.append(&field_element_to_bytes(claimed_sum));

        for _round in 0..self.number_of_variables {
            let univariate = self.generate_round_univariate();
            transcript.append(&univariate_to_bytes(&univariate));

            round_univariate_polynomials.push(univariate);

            let random_challenge: F = transcript.random_challenge_as_field_element();
            random_challenges.push(random_challenge);

            current_polynomial = current_polynomial.partial_evaluate(0, random_challenge);
        }

        GKRSumcheckProverProof {
            claimed_sum: claimed_sum,
            round_univariate_polynomials,
            random_challenges,
            degree: self.sum_polynomial.degree()
        }
    }

    pub fn verify(&self, proof: &GKRSumcheckProverProof<F>) -> GKRSumcheckVerifierProof<F> {
        let mut transcript = Transcript::new();
        transcript.append(&field_element_to_bytes(proof.claimed_sum));
        
        let mut current_sum = proof.claimed_sum;
        let mut random_challenges = Vec::new();

        let x_values: Vec<F> = (0..=proof.degree).map(|i| F::from(i as u64)).collect();

        for round_polynomial in &proof.round_univariate_polynomials {
            let univariate_poly = DensedUnivariatePolynomial::lagrange_interpolate(&x_values, &round_polynomial);

            let eval_at_zero = univariate_poly.evaluate(F::zero());
            let eval_at_one = univariate_poly.evaluate(F::one());

            if eval_at_zero + eval_at_one != current_sum {
                return GKRSumcheckVerifierProof {
                    is_proof_valid: false,
                    random_challenges: vec![],
                }
            }

            transcript.append(&univariate_to_bytes(round_polynomial));

            let random_challenge = transcript.random_challenge_as_field_element();

            current_sum = univariate_poly.evaluate(random_challenge);

            random_challenges.push(random_challenge);
        }

        GKRSumcheckVerifierProof {
            is_proof_valid: true,
            random_challenges
        }
    }

    pub fn generate_round_univariate(&self) -> Vec<F> {
        let degree = self.sum_polynomial.degree();
        let num_evaluations = degree + 1;

        let mut evaluations = Vec::with_capacity(num_evaluations);

        for i in 0..num_evaluations {
            let value = F::from(i as u64);
            let partial_eval_sum_poly = self.sum_polynomial.partial_evaluate(0, value); // holding Vec<ProductPoly> : length of 2
            let evaluation = partial_eval_sum_poly.add_polynomials_element_wise().evaluated_values.iter().sum();

            evaluations.push(evaluation)
        }

        evaluations
    }
}

pub fn univariate_to_bytes<F: PrimeField>(univariate_poly: &[F]) -> Vec<u8> {
    univariate_poly
        .iter()
        .flat_map(|x| x.into_bigint().to_bytes_le())
        .collect()
}

pub fn field_element_to_bytes<F: PrimeField>(field_element: F) -> Vec<u8> {
    field_element.into_bigint().to_bytes_be()
}

#[cfg(test)]
mod tests {
    use super::*;
    
}