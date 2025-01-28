use polynomials::multilinear::evaluation_form::{MultilinearPolynomial, partial_evaluate};
use transcripts::fiat_shamir::{fiat_shamir_transcript::Transcript, interface::FiatShamirTranscriptInterface};
use ark_ff::PrimeField;

pub struct Prover<F: PrimeField> {
    polynomial: MultilinearPolynomial<F>,
    transcript: Transcript,
    claimed_sum: F,
    round_polynomials: Vec<MultilinearPolynomial<F>>
}

struct SumcheckProof<F: PrimeField> {
    claimed_sum: F,
    round_polynomials: Vec<MultilinearPolynomial<F>>,
    sampled_challenges: Vec<F>,
}

impl <F: PrimeField>Prover<F> {
    pub fn init(polynomial_values_values: Vec<F>) -> Self {
        let polynomial = MultilinearPolynomial::new(polynomial_values_values.clone());
        let transcript = Transcript::new();

        Prover {
            polynomial,
            transcript,
            claimed_sum: compute_sum(polynomial_values_values),
            round_polynomials: Vec::new()
        }
    }

    pub fn prove() {

    }
}

pub fn compute_sum<F: PrimeField>(polynomial_values_values: Vec<F>) -> F {
    let mut sum = F::from(0);

    for i in polynomial_values_values.iter() {
        sum += i;
    }

    sum
}


#[cfg(test)]
mod test {
    use super::*;
    use ark_bn254::Fq;

    #[test]
    fn test_prover_init() {
        let evaluated_values = vec![Fq::from(0), Fq::from(0), Fq::from(3), Fq::from(8)];
        let prover = Prover::init(evaluated_values.clone());

        assert_eq!(prover.claimed_sum, Fq::from(11));
        assert_eq!(prover.polynomial.evaluated_values, evaluated_values);
    }
}