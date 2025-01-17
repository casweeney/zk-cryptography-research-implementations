use univariate_interpolation::UnivariatePolynomial;

fn main() {
    let set_of_points = vec![0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0];
    let polynomial = UnivariatePolynomial::new(set_of_points.clone());
    let evaluation_value = 2.0;

    println!("Degree of polynomial = {}", polynomial.degree());
    println!("The polynomial {:?} evaluated at {} = {}", set_of_points, evaluation_value, polynomial.evaluate(evaluation_value));
}
