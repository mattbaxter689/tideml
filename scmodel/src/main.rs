pub mod schema;
use crate::schema::extract_feature_target;

mod sc_model;

pub mod fileops;
use crate::fileops::read_csv;

use std::path::PathBuf;
use polars::datatypes::Float64Type;


fn main() {
    let path = PathBuf::from("./scmodel/src/data/housing.csv");
    let df = read_csv(path).unwrap();
    let (x, y) = extract_feature_target(&df);
    let xs = x.as_ref().unwrap();
    let xdense = sc_model::create_x_dense(&xs).unwrap();

    //set up y to be array
    let ydense = y.as_ref().unwrap().to_ndarray::<Float64Type>().unwrap();
    let mut target: Vec<f64> = Vec::new();
    for val in ydense.iter() {
        target.push(*val);
    }

    sc_model::fit_smartcore(xdense, target);
    
}
