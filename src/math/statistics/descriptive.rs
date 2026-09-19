use crate::math::vector::Vector;

pub fn mean<T>(values: &Vector<T>) -> f64
where
    T: Into<f64> + Copy,
{
    values.mean()
}

pub fn median<T>(values: &Vector<T>) -> f64
where
    T: Into<f64> + Copy,
{
    assert!(!values.is_empty(), "Cannot calculate median of empty vector");

    let mut data = values.to_vec();

    data.sort_by(|a, b| {
        let a: f64 = (*a).into();
        let b: f64 = (*b).into();

        a.partial_cmp(&b).unwrap()
    });

    let middle = data.len() / 2;

    if data.len() % 2 == 0 {
        let left: f64 = data[middle - 1].into();
        let right: f64 = data[middle].into();

        (left + right) / 2.0
    } else {
        data[middle].into()
    }
}


pub fn min<T>(values: &Vector<T>) -> Option<&T>
where
    T: PartialOrd,
{
    values.min()
}

pub fn max<T>(values: &Vector<T>) -> Option<&T>
where
    T: PartialOrd,
{
    values.max()
}

pub fn range<T>(values: &Vector<T>) -> Option<f64>
where
    T: Into<f64> + Copy + PartialOrd,
{
    let min = values.min()?;
    let max = values.max()?;

    Some((*max).into() - (*min).into())
}

fn sum_squared_deviations<T>(values: &Vector<T>, mean: f64) -> f64
where
    T: Into<f64> + Copy,
{
    values
        .as_slice()
        .iter()
        .map(|&x| {
            let x: f64 = x.into();
            let difference = x - mean;

            difference * difference
        })
        .sum()
}

pub fn population_variance<T>(values: &Vector<T>) -> f64
where
    T: Into<f64> + Copy,
{
    assert!(
        !values.is_empty(),
        "Cannot calculate population variance of empty vector"
    );

    let mean = values.mean();

    let sum_squared_diff = sum_squared_deviations(values, mean);

    sum_squared_diff / values.len() as f64
}

pub fn sample_variance<T>(values: &Vector<T>) -> f64
where
    T: Into<f64> + Copy,
{
    assert!(
        values.len() >= 2,
        "Sample variance requires at least two values"
    );

    let mean = values.mean();

    let sum_squared_diff = sum_squared_deviations(values, mean);

    sum_squared_diff / (values.len() - 1) as f64
}

pub fn population_std_dev<T>(values: &Vector<T>) -> f64
where
    T: Into<f64> + Copy,
{
    population_variance(values).sqrt()
}

pub fn sample_std_dev<T>(values: &Vector<T>) -> f64
where
    T: Into<f64> + Copy,
{
    sample_variance(values).sqrt()
}




