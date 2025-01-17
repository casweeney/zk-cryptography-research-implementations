struct UnivariatePolynomialSparsed {
    points: Vec<(u32, u32)>
}

impl UnivariatePolynomialSparsed {
    fn new(points: Vec<(u32, u32)>) -> Self  {
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

    fn evaluate(&self, evaluation_value: u32) -> u32 {
        let mut result = 0;
    
        for i in self.points.iter() {
            result = result + i.0 * evaluation_value.pow(i.1.into()) 
        }
    
        result
    }
}

fn add_polynomials(polynomial1: &Vec<(u32, u32)>, polynomial2: &Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    let mut resolved_polynomial = polynomial1.clone();

    // merge both polynomials into one vector
    resolved_polynomial.extend(polynomial2.clone());

    // resolve polynomials by adding the coefficients with same exponent(power)
    simplify_terms(&mut resolved_polynomial);

    resolved_polynomial
}

fn simplify_terms(terms: &mut Vec<(u32, u32)>) {
    terms.sort_by_key(|points| points.1);

    let mut i = 0;

    while i < terms.len() - 1 {
        if terms[i].1 == terms[i + 1].1 {
            // combine coefficients with the same exponents
            terms[i].0 = terms[i].0.wrapping_add(terms[i + 1].0);
            terms.remove(i + 1);
        } else {
            i += 1;
        }
    }
    
    // remove terms with zero coefficients
    terms.retain(|&(c, _)| c != 0);
}

fn main() {
    let set_of_points = vec!((2,1), (5,0));
    let evaluated_at = 2;
    let polynomial = UnivariatePolynomialSparsed::new(set_of_points.clone());

    let p1 = vec![(2,1), (5,0)];
    let p2 = vec![(1,1), (2,0)];

    
    println!("Degree of polynomial = {}", polynomial.degree());
    println!("The polynomial of points {:?} evaluated at value {} = {}", set_of_points, evaluated_at, polynomial.evaluate(evaluated_at));
    println!("Polynomial added: {:?}", add_polynomials(&p1, &p2));
}
