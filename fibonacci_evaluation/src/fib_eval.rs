use polynomials::univariate::densed_univariate::DensedUnivariatePolynomial;
use ark_ff::PrimeField;

pub fn evaluation<F: PrimeField>(evaluation_value: F) -> F {
    let x_values = vec![F::from(1), F::from(2), F::from(3), F::from(4), F::from(5), F::from(6), F::from(7)];
    let y_values = vec![F::from(1), F::from(2), F::from(3), F::from(5), F::from(8), F::from(13), F::from(21)];

    let polynomial = DensedUnivariatePolynomial::lagrange_interpolate(x_values, y_values);

    polynomial.evaluate(evaluation_value)
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_bn254::Fq;

    #[test]
    fn test_evaluation() {
        // f(x) => f(4) => 5
        let value = Fq::from(4);

        // f(x) => f(x-1) + f(x-2) => 5
        // f(x-1) => f(3)
        // f(x-2) => f(2)

        let value_sub_one = value - Fq::from(1);
        let value_sub_two = value - Fq::from(2);

        assert_eq!(evaluation(value), Fq::from(5));
        assert_eq!(evaluation(value_sub_one) + evaluation(value_sub_two), Fq::from(5));
    }
}