use tide::Request;
// use serde::Serialize;
use scmodel::sc_model::{load_model, to_dense_mat};


mod input;
use crate::input::{SCInput, SCPred};



#[async_std::main]
async fn main() -> tide::Result<()> {
    femme::start();
    let mut app = tide::new();
    app.with(tide::log::LogMiddleware::new());
    app.at("/predict").post(get_prediction);
    app.listen("0.0.0.0:8080").await?;
    Ok(())
} 


async fn get_prediction(mut req: Request<()>) -> tide::Result<String> {
    let model = load_model("./scmodel/model/sc_reg.model".to_string());
    let SCInput { data } = req.body_json().await?;
    let input_dmat = to_dense_mat(data);
    let pred = model.predict(&input_dmat).unwrap();
    let out = SCPred {
        data: pred
    };
    Ok(format!("{}", serde_json::to_string(&out)?).into())
}
