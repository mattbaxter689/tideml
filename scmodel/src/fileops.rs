use polars::prelude::*;
use std::path::{PathBuf, Path};

//Generic reading in csv function
pub fn read_csv(path: PathBuf) -> PolarsResult<DataFrame> {
    CsvReader::from_path(path)?.has_header(true).finish()
}

pub fn check_model_dir() -> bool {
    if Path::new("./scmodel/model/").is_dir() {
        return true;
    }

    return false;
}

