use polars::prelude::*;

// Get the feature and target variables separate
pub fn extract_feature_target(
    df: &DataFrame,
) -> (PolarsResult<DataFrame>, PolarsResult<DataFrame>) {
    let features = df.select(vec![
        "crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "black",
        "lstat",
    ]);

    let target = df.select(["medv"]);

    return (features, target);
}
