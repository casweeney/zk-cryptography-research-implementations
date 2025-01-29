use polynomials::multilinear::evaluation_form::{partial_evaluate, MultilinearPolynomial};
use transcripts::fiat_shamir::{
    fiat_shamir_transcript::Transcript,
    interface::FiatShamirTranscriptInterface
};
use ark_ff::{PrimeField, BigInteger};

pub struct Prover<F: PrimeField> {
    pub initial_polynomial: MultilinearPolynomial<F>,
    pub initial_claimed_sum: F,
    pub transcript: Transcript,
    pub round_univariate_polynomials: Vec<MultilinearPolynomial<F>>,
    pub is_initialized: bool,
}

pub struct SumcheckProof<F: PrimeField> {
    pub initial_polynomial: MultilinearPolynomial<F>,
    pub initial_claimed_sum: F,
    pub round_univariate_polynomials: Vec<MultilinearPolynomial<F>>,
}

impl <F: PrimeField>Prover<F> {
    pub fn init(polynomial_evaluated_values: Vec<F>) -> Self {
        let polynomial = MultilinearPolynomial::new(polynomial_evaluated_values.clone());
        let transcript = Transcript::new();

        Prover {
            initial_polynomial: polynomial,
            initial_claimed_sum: compute_sum(polynomial_evaluated_values),
            transcript,
            round_univariate_polynomials: Vec::new(),
            is_initialized: true
        }
    }

    pub fn prove(&mut self) -> SumcheckProof<F> {
        assert!(self.is_initialized, "Can't prove without init");

        let mut random_challenges: Vec<F> = vec![];

        self.transcript.append(&self.initial_polynomial.convert_to_bytes());
        self.transcript.append(&field_element_to_bytes(self.initial_claimed_sum));

        let mut current_polynomial = self.initial_polynomial.evaluated_values.clone();

        for _ in 0..self.initial_polynomial.number_of_variables() {
            // The split_polynomial_and_sum_each() does the main work of converting a multilinear polynomial to a univariate polynomial
            // It does this by splitting the multilinear polynomial into 2 equal halves and summing each half
            // This will return a univariate polynomial where the first variable of the multilinear polynomial is evaluated at 0 and 1
            let univariate_polynomial_values = split_polynomial_and_sum_each(&current_polynomial);
            let univariate_polynomial = MultilinearPolynomial::new(univariate_polynomial_values);
            let univariate_poly_in_bytes = univariate_polynomial.convert_to_bytes();
            self.round_univariate_polynomials.push(univariate_polynomial);
            self.transcript.append(&univariate_poly_in_bytes);

            // get random challenge
            let random_challenge: F = self.transcript.random_challenge_as_field_element();
            random_challenges.push(random_challenge);

            // Partial evaluate current polynomial using the random challenge
            current_polynomial = partial_evaluate(&current_polynomial, 0, random_challenge);
        }

        SumcheckProof {
            initial_polynomial: self.initial_polynomial.clone(),
            initial_claimed_sum: self.initial_claimed_sum,
            round_univariate_polynomials: self.round_univariate_polynomials.clone(),
        }
    }
}

pub fn compute_sum<F: PrimeField>(polynomial_evaluated_values: Vec<F>) -> F {
    let mut sum = F::zero();

    for i in polynomial_evaluated_values.iter() {
        sum += i;
    }

    sum
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

pub fn field_element_to_bytes<F: PrimeField>(field_element: F) -> Vec<u8> {
    field_element.into_bigint().to_bytes_be()
}


#[cfg(test)]
mod test {
    use super::*;
    use ark_bn254::Fq;

    #[test]
    fn test_prover_init() {
        let evaluated_values = vec![Fq::from(0), Fq::from(0), Fq::from(3), Fq::from(8)];
        let prover = Prover::init(evaluated_values.clone());

        assert_eq!(prover.initial_claimed_sum, Fq::from(11));
        assert_eq!(prover.initial_polynomial.evaluated_values, evaluated_values);
    }
}