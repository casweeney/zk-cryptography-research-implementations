use ark_ff::{PrimeField, AdditiveGroup};
use ark_ec::{pairing::{Pairing, PairingOutput}, PrimeGroup};
use polynomials::multilinear::evaluation_form::MultilinearPolynomial;
use crate::trusted_setup::TrustedSetup;

pub struct MultilinearKZGProof<F: PrimeField, P: Pairing> {
    pub evaluation: F, // this is represented as "v" in the verification formula : it is the evaluation of the polynomial at verifier's selected points
    pub proofs: Vec<P::G1> // this holds the proof, and the length is based on the number of variables in the polynomial
}

/// This function is used to commit to a polynomial
/// The commitment is sent to the verifier, this commitment is what makes verification succinct
pub fn commit_to_polynomial<F: PrimeField, P: Pairing>(
    polynomial: &MultilinearPolynomial<F>,
    trust_setup: &TrustedSetup<P>
) -> P::G1 {
    assert_eq!(polynomial.evaluated_values.len(), trust_setup.g1_powers_of_tau.len(), "Polynomial evaluation must match g1 length");

    // Commitment to a polynomial is basically a dot product operation between the values of the polynomial and the powers of tau.
    // The powers of tau is based on the lagrange basis of the variables over the boolean hypercube
    let commitment = polynomial.evaluated_values
        .iter()
        .zip(trust_setup.g1_powers_of_tau.iter())
        .map(|(evaluated_value, power)| power.mul_bigint(evaluated_value.into_bigint()))
        .sum();

    commitment
}

/// This function is used to open polynomial random points that will be sent by the verifier.
/// The length of the array of points will be equal to the number of variables in the polynomial
/// The prover will evaluate the polynomial at the points the verifier sent, then generate proofs to validate the evaluation at those points
pub fn open_and_prove<F: PrimeField, P: Pairing>(
    polynomial: &MultilinearPolynomial<F>,
    trust_setup: &TrustedSetup<P>,
    opening_values: &[F]
) -> MultilinearKZGProof<F, P> {
    assert_eq!(polynomial.number_of_variables() as usize, opening_values.len(), "number of polynomial variables must match length of opening values");
    assert_eq!(opening_values.len(), trust_setup.g2_powers_of_tau.len(), "Opening values must match number of variables from trusted setup");

    let mut proofs: Vec<_> = Vec::with_capacity(opening_values.len());

    // NOTE: Before we start generating proofs, we need to first evaluate the polynomial using the values given by the verifier to open the polynomial.
    // Evaluating Multilinear polynomial at the opening points => opening_values
    let evaluation_v = polynomial.evaluate(opening_values); // evaluation of polynomial at random opening values => v

    // We need to subtract the evaluation from the polynomial to get: f(x) - v => f(a,b,c) - v
    // Where v => evaluation of polynomial at opening values
    let polynomial_minus_v: Vec<F> = polynomial.evaluated_values.iter().map(|value| *value - evaluation_v).collect();
    let mut sub_polynomial = MultilinearPolynomial::new(&polynomial_minus_v); // f(a,b,c) - v => => This will be updated with the remainder polynomial

    // Let's generate the proofs to prove that the evaluation at the giving values is correct/valid.
    // To do this we need to perform a running division => but in this case we use partial evaluation
    // The running division will run number of variable times

    for i in 0..opening_values.len() {
        let evaluating_value = opening_values[i];

        // Get the quotient polynomial, which will be evaluated at tau, to get the proof of each iteration
        let quotient_polynomial = compute_quotient_polynomial(&sub_polynomial);
        
        // With the quotient polynomial, compute the proof and push into the proofs vector
        // But before we need to blow up to replace the removed variable with zero so that
        // the quotient polynomial will have the same length with as lagrange basis polynomial for dot product operation
        // Blow up is based on number of variables removed (i + 1)
        let blow_up_times = i + 1;
        let blown_quotient_polynomial = blow_up(quotient_polynomial, blow_up_times);

        // Generate Proof
        let proof: P::G1 = blown_quotient_polynomial
            .evaluated_values
            .iter()
            .zip(trust_setup.g1_powers_of_tau.iter())
            .map(|(evaluated_value, g1_power)| g1_power.mul_bigint(evaluated_value.into_bigint()))
            .sum();

        proofs.push(proof);


        // Compute remainder polynomial and update current polynomial with the remainder
        // This will be used in the next iteration
        let remainder_polynomial = MultilinearPolynomial::partial_evaluate(&sub_polynomial.evaluated_values, 0, evaluating_value);

        sub_polynomial = remainder_polynomial;
    }

    MultilinearKZGProof {
        evaluation: evaluation_v,
        proofs,
    }
}

/// This function is used to verify the prover's claim. The verifier calls this function
/// It uses the verifier equation/formula to compute the left-hand-side and right-hand-side
/// then it checks if both sides are equal
pub fn verify<F: PrimeField, P: Pairing>(
    trust_setup: &TrustedSetup<P>,
    commitment: &P::G1,
    opening_values: &[F],
    proof: &MultilinearKZGProof<F, P>
) -> bool {
    assert_eq!(
        opening_values.len(),
        proof.proofs.len(),
        "Number of opening values must match number of proofs"
    );

    // Compute LHS => g1_commitment - g1_evaluation * g2_1
    let commitment_minus_v = *commitment - P::G1::generator().mul_bigint(proof.evaluation.into_bigint());
    let lhs = P::pairing(commitment_minus_v, P::G2::generator());

    // Compute RHS => Running summation of: (g1_Qi * (g2_tau - g2_xi))
    let mut rhs = PairingOutput::ZERO;
    for (i, tau) in trust_setup.g2_powers_of_tau.iter().enumerate() {
        rhs += P::pairing(proof.proofs[i], *tau - P::G2::generator().mul_bigint(opening_values[i].into_bigint()))
    }

    lhs == rhs
}


////////////////////////////////////////
///         Helper Functions        ///
///////////////////////////////////////

fn compute_quotient_polynomial<F: PrimeField>(
    polynomial: &MultilinearPolynomial<F>
) -> MultilinearPolynomial<F> {
    let mid = polynomial.evaluated_values.len() / 2;
    let (evals_at_zero, evals_at_one) = polynomial.evaluated_values.split_at(mid);
    
    // Perform element-wise subtraction: subtract partial evaluation at 0 from partial evaluation at 1
    let quotient_evaluations: Vec<F> = evals_at_one.iter()
        .zip(evals_at_zero.iter())
        .map(|(eval_at_one, eval_at_zero)| *eval_at_one - *eval_at_zero)
        .collect();

    MultilinearPolynomial::new(&quotient_evaluations)
}

fn blow_up<F: PrimeField>(
    polynomial: MultilinearPolynomial<F>,
    blow_up_times: usize,
) -> MultilinearPolynomial<F> {
    assert!(
        polynomial.evaluated_values.len().is_power_of_two(),
        "length of polynomial must be power 2"
    );

    let mut blown_values = polynomial.evaluated_values;

    for _ in 0..blow_up_times {
        blown_values = expand_vec(&blown_values);
    }

    MultilinearPolynomial::new(&blown_values)
}

fn expand_vec<F: PrimeField>(values: &[F]) -> Vec<F> {
    let number_of_bits = values.len();
    let total_combinations = 2 * number_of_bits;
    let mut blown_values = Vec::with_capacity(total_combinations);

    for _ in 0..2 {
        blown_values.extend(values.iter().copied());
    }

    blown_values
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::{Bls12_381, Fr};

    #[test]
    fn test_multilinear_kzg() {
        let taus = vec![Fr::from(5), Fr::from(2), Fr::from(3)];
        let setup = TrustedSetup::<Bls12_381>::initialize_setup(&taus);
        let values = vec![
            Fr::from(0),
            Fr::from(4),
            Fr::from(0),
            Fr::from(4),
            Fr::from(0),
            Fr::from(4),
            Fr::from(3),
            Fr::from(7)
        ];
        let polynomial = MultilinearPolynomial::new(&values);

        let commitment = commit_to_polynomial(&polynomial, &setup);
        let opening_values = vec![Fr::from(6), Fr::from(4), Fr::from(0)];

        let proof = open_and_prove(&polynomial, &setup, &opening_values.to_vec());

        let verification = verify(&setup, &commitment, &opening_values, &proof);

        assert_eq!(verification, true);
    }

    #[test]
    fn test_multilinear_kzg2() {
        let taus = vec![Fr::from(2), Fr::from(3), Fr::from(4)];
        let setup = TrustedSetup::<Bls12_381>::initialize_setup(&taus);

        let values = vec![
            Fr::from(0),
            Fr::from(7),
            Fr::from(0),
            Fr::from(5),
            Fr::from(0),
            Fr::from(7),
            Fr::from(4),
            Fr::from(9),
        ];
        let polynomial = MultilinearPolynomial::new(&values);

        let commitment = commit_to_polynomial(&polynomial, &setup);
        let opening_values = vec![Fr::from(5), Fr::from(9), Fr::from(6)];

        let proof = open_and_prove(&polynomial, &setup, &opening_values.to_vec());

        let verification = verify(&setup, &commitment, &opening_values, &proof);

        assert_eq!(verification, true);
    }

    #[test]
    fn test_multilinear_kzg3() {
        let taus = vec![Fr::from(12), Fr::from(9), Fr::from(28), Fr::from(40)];
        let setup = TrustedSetup::<Bls12_381>::initialize_setup(&taus);

        let values = vec![
            Fr::from(0),
            Fr::from(0),
            Fr::from(0),
            Fr::from(2),
            Fr::from(0),
            Fr::from(0),
            Fr::from(10),
            Fr::from(12),
            Fr::from(0),
            Fr::from(-12),
            Fr::from(4),
            Fr::from(-6),
            Fr::from(0),
            Fr::from(-12),
            Fr::from(14),
            Fr::from(4),
        ];

        let polynomial = MultilinearPolynomial::new(&values);

        let commitment = commit_to_polynomial(&polynomial, &setup);
        let opening_values = vec![Fr::from(54), Fr::from(90), Fr::from(76), Fr::from(160)];

        let proof = open_and_prove(&polynomial, &setup, &opening_values.to_vec());

        let verification = verify(&setup, &commitment, &opening_values, &proof);

        assert_eq!(verification, true);
    }
}