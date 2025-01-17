struct UnivariatePolynomialSparsed {
    points: Vec<(f64, u32)>
}

impl UnivariatePolynomialSparsed {
    fn new(points: Vec<(f64, u32)>) -> Self  {
        Self {
            points
        }
    }

    fn degree(&self) -> u32 {
        let mut highest_power = 0;
    
        for i in self.points.iter() {
            if i.1 > highest_power {
                highest_power = i.1;
            }
        }
    
        highest_power
    }

    fn evaluate(&self, evaluation_value: f64) -> f64 {
        let mut result = 0.0;
    
        for i in self.points.iter() {
            result = result + i.0 * evaluation_value.powi(i.1 as i32)
        }
    
        result
    }

    fn interpolate(&self) -> Vec<(f64, u32)> {
        let n = self.points.len();
        
        // First, let's evaluate our polynomial at n different x points
        // For simplicity, let's use x = 0, 1, 2, ..., n-1
        let mut points = Vec::new();
        for i in 0..n {
            let x = i as f64;
            let y = self.evaluate(x);
            points.push((x, y));
        }
        
        // Now we can interpolate using these points
        let mut result = vec![(0.0, 0)];
        
        for k in 0..n {
            let mut basis = vec![(1.0, 0)];
            let (x_k, y_k) = points[k];
            
            for j in 0..n {
                if j != k {
                    let (x_j, _) = points[j];
                    
                    if (x_k - x_j).abs() < f64::EPSILON {
                        continue;
                    }
                    
                    let numerator = vec![(1.0, 1), (-x_j, 0)];
                    basis = multiply_polynomials(&basis, &numerator);
                    basis = multiply_by_scalar(&basis, 1.0/(x_k - x_j));
                }
            }
            
            basis = multiply_by_scalar(&basis, y_k);
            result = add_polynomials(&result, &basis);
        }
        
        result
    }
}

fn add_polynomials(polynomial1: &Vec<(f64, u32)>, polynomial2: &Vec<(f64, u32)>) -> Vec<(f64, u32)> {
    let mut resolved_polynomial = polynomial1.clone();

    // merge both polynomials into one vector
    resolved_polynomial.extend(polynomial2.clone());

    // resolve polynomials by adding the coefficients with same exponent(power)
    simplify_terms(&mut resolved_polynomial);

    resolved_polynomial
}

fn multiply_polynomials(polynomial1: &Vec<(f64, u32)>, polynomial2: &Vec<(f64, u32)>) -> Vec<(f64, u32)> {
    let mut resolved_polynomial = Vec::new();
    
    for (coefficient1, exponent1) in polynomial1 {
        for (coefficient2, exponent2) in polynomial2 {
            // multiply coefficients, add exponents
            let coef = coefficient1 * coefficient2;
            let exp = exponent1 + exponent2;
            resolved_polynomial.push((coef, exp));
        }
    }
    
    // combine like terms
    simplify_terms(&mut resolved_polynomial);
    resolved_polynomial
}

fn simplify_terms(terms: &mut Vec<(f64, u32)>) {
    terms.sort_by_key(|points| points.1);

    let mut i = 0;

    while i < terms.len() - 1 {
        if terms[i].1 == terms[i + 1].1 {
            // combine coefficients with the same exponents
            terms[i].0 = terms[i].0 + terms[i + 1].0;
            terms.remove(i + 1);
        } else {
            i += 1;
        }
    }
    
    // remove terms with zero coefficients
    terms.retain(|&(coef, _)| coef.abs() > f64::EPSILON);
}

fn multiply_by_scalar(p: &Vec<(f64, u32)>, scalar: f64) -> Vec<(f64, u32)> {
    let mut result = Vec::new();
    for (coef, exp) in p {
        result.push((coef * scalar, *exp));
    }
    result
}

fn main() {
    let set_of_points = vec!((3.0,2),(2.0,1), (5.0,0));
    let evaluated_at = 2.0;
    let polynomial = UnivariatePolynomialSparsed::new(set_of_points.clone());

    let p1 = vec![(2.0,1), (5.0,0)];
    let p2 = vec![(1.0,1), (2.0,0)];

    
    println!("Degree of polynomial = {}", polynomial.degree());
    println!("The polynomial of points {:?} evaluated at value {} = {}", set_of_points, evaluated_at, polynomial.evaluate(evaluated_at));
    println!("Polynomials added: {:?}", add_polynomials(&p1, &p2));
    println!("Polynomials multiplied: {:?}", multiply_polynomials(&p1, &p2));
    println!("Polynomial Interpolation: {:?}", polynomial.interpolate())
}
