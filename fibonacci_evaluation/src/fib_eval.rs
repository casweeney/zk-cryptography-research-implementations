use polynomials::univariate::densed_univariate::DensedUnivariatePolynomial;
use ark_ff::PrimeField;

pub fn evaluation<F: PrimeField>(evaluation_value: F) -> F {
    let x_values = vec![F::from(1), F::from(2), F::from(3), F::from(4), F::from(5), F::from(6), F::from(7)];
    let y_values = vec![F::from(1), F::from(2), F::from(3), F::from(5), F::from(8), F::from(13), F::from(21)];

    let polynomial = DensedUnivariatePolynomial::lagrange_interpolate(&x_values, &y_values);

    polynomial.evaluate(evaluation_value)
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_bn254::Fq;

    #[test]
    fn test_evaluation_points_within() {
        // Given that:
        // f(x) => f(4) => 5

        // From the fibonacci sequence, we concluded that:
        // f(x) => f(x-1) + f(x-2) => 5
        // f(x-1) => f(3)
        // f(x-2) => f(2)

        // Let's test this to be sure

        for x in 1..=7 {
            let value = Fq::from(x);
            
            if x > 2 {
                let value_sub_one = value - Fq::from(1);
                let value_sub_two = value - Fq::from(2);

                assert_eq!(evaluation(value), evaluation(value_sub_one) + evaluation(value_sub_two));
            }
        }
    }

    #[test]
    fn test_evaluation_at_7() {
        // We want to check to be sure that f(7) => 21

        assert_eq!(evaluation(Fq::from(7)), Fq::from(21));
    }

    // #[test]
    // fn test_evaluation_points_outside() {
    //     let eval_at_8 = evaluation(Fq::from(8));

    //     println!("Evaluation at 8 = {}", eval_at_8);

    //     for x in 8..=10 {
    //         let value = Fq::from(x);
            
    //         if x > 2 {
    //             let value_sub_one = value - Fq::from(1);
    //             let value_sub_two = value - Fq::from(2);

    //             assert_eq!(evaluation(value), evaluation(value_sub_one) + evaluation(value_sub_two));
    //         }
    //     }
    // }

    // The above test is failing:
    // This implies that the polynomial interpolation is not maintaining the Fibonacci property outside the interpolated range.
}