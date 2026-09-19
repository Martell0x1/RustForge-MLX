use rustforge_mlx::math::statistics::descriptive::*;
use rustforge_mlx::math::vector::Vector;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-10,
        "expected {expected}, got {actual}"
    );
}

#[test]
#[should_panic(expected = "Cannot calculate mean of empty vector")]
fn mean_empty() {
    let values = Vector::new(Vec::<f64>::new());

    mean(&values);
}

#[test]
fn mean_one_element() {
    let values = Vector::new(vec![5.0]);

    assert_close(mean(&values), 5.0);
}

#[test]
fn mean_negative_values() {
    let values = Vector::new(vec![-2.0, -4.0, -6.0]);

    assert_close(mean(&values), -4.0);
}

#[test]
fn mean_floating_point() {
    let values = Vector::new(vec![1.5, 2.5, 3.5]);

    assert_close(mean(&values), 2.5);
}

#[test]
fn mean_constant() {
    let values = Vector::new(vec![7.0, 7.0, 7.0]);

    assert_close(mean(&values), 7.0);
}

#[test]
#[should_panic(expected = "Cannot calculate median of empty vector")]
fn median_empty() {
    let values = Vector::new(Vec::<f64>::new());

    median(&values);
}

#[test]
fn median_one_element() {
    let values = Vector::new(vec![5.0]);

    assert_close(median(&values), 5.0);
}

#[test]
fn median_odd() {
    let values = Vector::new(vec![5.0, 1.0, 3.0]);

    assert_close(median(&values), 3.0);
}

#[test]
fn median_even() {
    let values = Vector::new(vec![4.0, 1.0, 3.0, 2.0]);

    assert_close(median(&values), 2.5);
}

#[test]
fn median_negative_values() {
    let values = Vector::new(vec![-1.0, -5.0, -3.0]);

    assert_close(median(&values), -3.0);
}

#[test]
fn min_and_max() {
    let values = Vector::new(vec![4.0, 1.0, 7.0, 2.0]);

    assert_eq!(min(&values), Some(&1.0));
    assert_eq!(max(&values), Some(&7.0));
}

#[test]
fn min_and_max_empty() {
    let values = Vector::new(Vec::<f64>::new());

    assert_eq!(min(&values), None);
    assert_eq!(max(&values), None);
}

#[test]
fn population_variance_known_value() {
    let values = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    assert_close(population_variance(&values), 2.0);
}

#[test]
fn population_variance_constant() {
    let values = Vector::new(vec![5.0, 5.0, 5.0]);

    assert_close(population_variance(&values), 0.0);
}

#[test]
#[should_panic(expected = "Cannot calculate population variance of empty vector")]
fn population_variance_empty() {
    let values = Vector::new(Vec::<f64>::new());

    population_variance(&values);
}

#[test]
fn sample_variance_known_value() {
    let values = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    assert_close(sample_variance(&values), 2.5);
}

#[test]
fn sample_variance_constant() {
    let values = Vector::new(vec![5.0, 5.0, 5.0]);

    assert_close(sample_variance(&values), 0.0);
}

#[test]
#[should_panic(expected = "Sample variance requires at least two values")]
fn sample_variance_one_element() {
    let values = Vector::new(vec![5.0]);

    sample_variance(&values);
}

#[test]
fn population_std_dev_known_value() {
    let values = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    assert_close(population_std_dev(&values), 2.0_f64.sqrt());
}

#[test]
fn sample_std_dev_known_value() {
    let values = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    assert_close(sample_std_dev(&values), 2.5_f64.sqrt());
}

#[test]
fn standard_deviation_constant() {
    let values = Vector::new(vec![5.0, 5.0, 5.0]);

    assert_close(population_std_dev(&values), 0.0);
    assert_close(sample_std_dev(&values), 0.0);
}
