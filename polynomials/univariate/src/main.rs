use univariate::densed_univariate::{DensedUnivariatePolynomial, multiply_polynomials};

fn main() {
    let set_of_points = vec![0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0];
    let polynomial = DensedUnivariatePolynomial::new(set_of_points.clone());
    let evaluation_value = 2.0;

    let p2 = vec![5.0, 0.0, 2.0];
    let p1 = vec![6.0, 2.0];

    println!("Degree of polynomial = {}", polynomial.degree());
    println!("The polynomial {:?} evaluated at {} = {}", set_of_points, evaluation_value, polynomial.evaluate_advanced(evaluation_value));
    println!("Product poly {:?}", multiply_polynomials(p1, p2));
}
