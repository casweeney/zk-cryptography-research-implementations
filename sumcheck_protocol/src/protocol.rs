#[cfg(test)]
mod tests {
    use ark_bls12_381::Fr;
    use crate::prover::Prover;
    use crate::verifier::Verifier;
    use polynomials::multilinear::evaluation_form::MultilinearPolynomial;

    #[test]
    fn test_sumcheck_protocol_init() {
        let polynomial_evaluated_values = vec![
            Fr::from(0),
            Fr::from(0),
            Fr::from(2),
            Fr::from(7),
            Fr::from(3),
            Fr::from(3),
            Fr::from(6),
            Fr::from(11),
        ];

        let prover = Prover::init(polynomial_evaluated_values);

        assert_eq!(prover.initial_claimed_sum, Fr::from(32));
        assert_eq!(prover.is_initialized, true);
    }

    #[test]
    fn test_sumcheck_protocol_prove_and_verify() {
        let polynomial_evaluated_values = vec![
            Fr::from(0),
            Fr::from(0),
            Fr::from(2),
            Fr::from(7),
            Fr::from(3),
            Fr::from(3),
            Fr::from(6),
            Fr::from(11),
        ];

        let mut prover = Prover::init(polynomial_evaluated_values);
        let proof = prover.prove();

        let mut verifier: Verifier<Fr> = Verifier::init();
        assert_eq!(verifier.is_initialized, true);

        assert_eq!(verifier.verify(proof), true);
    }
}