#[derive(serde::Deserialize)]
pub struct SCInput {
    pub data: Vec<f64>
}

#[derive(serde::Serialize)]
pub struct SCPred {
    pub data: Vec<f64>
}
