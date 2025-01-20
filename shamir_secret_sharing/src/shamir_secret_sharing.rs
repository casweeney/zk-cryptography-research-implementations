// if f(0) = 17 => (0,17) Our secret is 17
// Let's assume we got other random points for x,y as (24), (35), (40) => degree = 3
//                                       0   1   2   3
// Now we have the following y_values = 17, 24, 35, 40

// Using the polynomial, we can generate 10 extra x_values
// From the newly generated 10 x_values, evaluate the polynomial at those x_values to get the y_values
// Interpolate them to get a polynomial
// Then we evaluate the polynomial at f(0), we are expected to get back the secret (17)

//share(secret, threshold, number_shares)

use univariate::densed_univariate::DensedUnivariatePolynomial;
use ark_ff::PrimeField;

pub fn shares<F: PrimeField>(secret: F, threshold: u64, number_shares: u64) -> Vec<(F, F)> {
    let mut y_values = vec![secret];
    let mut rng = rand::thread_rng();
    let mut shares: Vec<(F, F)> = Vec::new();

    for _ in 1..threshold {
        y_values.push(F::rand(&mut rng));
    }

    let polynomial = DensedUnivariatePolynomial::new(y_values);

    for i in 1..number_shares {
        shares.push((F::from(i), polynomial.evaluate(F::from(i))))
    } 

    shares
}

pub fn recover_secret<F: PrimeField>(shares: Vec<(F, F)>) -> F {
    let mut x_values: Vec<F> = Vec::new();
    let mut y_values: Vec<F> = Vec::new();

    for i in shares.iter() {
        x_values.push(i.0);
        y_values.push(i.1);
    }


    let polynomial = DensedUnivariatePolynomial::lagrange_interpolate(x_values, y_values);

    let secret = polynomial.evaluate(F::from(0));

    secret
}


#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fq;

    #[test]
    fn test_recover_secret() {
        let secret = Fq::from(17);
        let threshold = 4;
        let number_of_shares = 10;

        let shares = shares(secret, threshold, number_of_shares);

        let recovered_secret = recover_secret(shares);

        assert_eq!(recovered_secret, secret);

    }

    #[test]
    fn test_recover_wrong_secret_fails() {
        let secret = Fq::from(17);
        let threshold = 4;
        let number_of_shares = 10;

        let shares = shares(secret, threshold, number_of_shares);

        let recovered_secret = recover_secret(shares);

        assert_ne!(recovered_secret, Fq::from(10));

    }
}