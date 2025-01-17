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

    fn interpolate(&self) {
    
    }
}

fn main() {
    let set_of_points = vec!((3,2), (2,1), (5,0));
    let evaluated_at = 2;
    let polynomial = UnivariatePolynomialSparsed::new(set_of_points.clone());

    
    println!("Degree of polynomial = {}", polynomial.degree());
    
    println!("The polynomial of points {:?} evaluated at value {} = {}", set_of_points, evaluated_at, polynomial.evaluate(evaluated_at));
}
