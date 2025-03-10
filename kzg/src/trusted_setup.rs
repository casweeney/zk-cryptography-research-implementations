use ark_ff::PrimeField;
use ark_ec::{pairing::Pairing, PrimeGroup};


pub struct TrustedSetup<P: Pairing> {
    pub g1_powers_of_tau: Vec<P::G1>, // used for evaluation of multilinear polynomial by performing dot product
    pub g2_powers_of_tau: Vec<P::G2> // used for verification where (tau - a) will be used
}

impl <P: Pairing>TrustedSetup<P> {
    pub fn initialize_setup<F: PrimeField>(taus: &[F]) -> Self {
        let lagrange_basis = compute_lagrange_basis(taus);

        let g1_powers_of_tau = compute_g1_powers_of_tau::<P, F>(&lagrange_basis);
        let g2_powers_of_tau = compute_g2_powers_of_tau::<P, F>(taus);

        Self {
            g1_powers_of_tau,
            g2_powers_of_tau
        }
    }
}

fn compute_lagrange_basis<F: PrimeField>(taus: &[F]) -> Vec<F> {
    let num_of_variables = taus.len();
    assert!(num_of_variables > 0, "requires at least one variable");

    let num_of_evaluations = 1 << num_of_variables; // 2^number_of_variable = number boolean hypercube evaluation

    let mut lagrange_basis = Vec::with_capacity(num_of_evaluations);

    for index in 0..num_of_evaluations {
        let mut exponent = F::one();

        for i in 0..num_of_variables {
            let bit = (index >> (num_of_variables - 1 - i)) & 1;

            if bit == 1 {
                exponent *= taus[i];
            } else {
                exponent *= F::one() - taus[i]
            }
        }

        lagrange_basis.push(exponent);
    }

    lagrange_basis
}

fn compute_g1_powers_of_tau<P: Pairing, F: PrimeField>(lagrange_basis: &[F]) -> Vec<P::G1> {
    let g1 = P::G1::generator();
    let mut g1_powers_of_tau = Vec::with_capacity(lagrange_basis.len());

    for exponent in lagrange_basis {
        g1_powers_of_tau.push(g1.mul_bigint(exponent.into_bigint()));
    }

    g1_powers_of_tau
}

pub fn compute_g2_powers_of_tau<P: Pairing, F: PrimeField>(taus: &[F]) -> Vec<P::G2> {
    let num_of_variables = taus.len();
    assert!(num_of_variables > 0, "requires at least one variable");

    let g2 = P::G2::generator();
    let mut g2_powers_of_tau = Vec::with_capacity(num_of_variables);

    for i in 0..num_of_variables {
        g2_powers_of_tau.push(g2.mul_bigint(taus[i].into_bigint()));
    }

    g2_powers_of_tau
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;

    #[test]
    fn test_compute_lagrange_basis() {
        let taus = vec![Fr::from(5), Fr::from(2), Fr::from(3)];
        let lagrange_basis = compute_lagrange_basis(&taus);
        
        let expected_lagrange_basis = vec![
            Fr::from(-8),
            Fr::from(12),
            Fr::from(16),
            Fr::from(-24),
            Fr::from(10),
            Fr::from(-15),
            Fr::from(-20),
            Fr::from(30),
        ];

        assert_eq!(lagrange_basis, expected_lagrange_basis);
    }
}