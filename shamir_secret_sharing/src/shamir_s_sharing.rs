use polynomials::univariate::densed_univariate::DensedUnivariatePolynomial;
use ark_ff::PrimeField;

// In this implementation, given (0, 17) as the secret, evaluated at 0
// we generated random x_values and y_values using the threshold
// we interpolated with the (x,y) values to get a polynomial
// then we evaluated the polynomial from 1..number_of_shares to get more shares (x,y)
// using the number of shares, we separated the x and y values
// Then we interpolated again to get a polynomial which when evaluated at 0, we get back the secret

/// Added password to serve as an extra level of security.
/// The password will be the point at which x will be evaluated to get the secret.
pub fn s_shares<F: PrimeField>(secret: F, password: u64, threshold: u64, number_shares: u64) -> Vec<(F, F)> {
    let mut x_values = vec![F::from(password)];
    let mut y_values = vec![secret];

    let mut rng = rand::thread_rng();
    let mut shares: Vec<(F, F)> = Vec::new();

    for i in password + 1..threshold {
        x_values.push(F::from(i));
        y_values.push(F::rand(&mut rng));
    }

    let polynomial = DensedUnivariatePolynomial::lagrange_interpolate(x_values, y_values);

    // todo!() : We need a while loop to regenerate the random numbers if the condition below fails 
    if polynomial.degree() as u64 == threshold - 1 {
        for i in password + 1..number_shares {
            shares.push((F::from(i), polynomial.evaluate(F::from(i))))
        }
    }

    shares
}

pub fn s_recover_secret<F: PrimeField>(shares: Vec<(F, F)>) -> F {
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
    fn test_s_recover_secret() {
        let password = 0;
        let secret = Fq::from(17);
        let threshold = 4;
        let number_of_shares = 10;

        let shares = s_shares(secret, password, threshold, number_of_shares);

        let recovered_secret = s_recover_secret(shares);

        assert_eq!(recovered_secret, secret);

    }

    #[test]
    fn test_s_recover_wrong_secret_fails() {
        let password = 0;
        let secret = Fq::from(17);
        let threshold = 4;
        let number_of_shares = 10;

        let shares = s_shares(secret, password, threshold, number_of_shares);

        let recovered_secret = s_recover_secret(shares);

        assert_ne!(recovered_secret, Fq::from(10));

    }
}