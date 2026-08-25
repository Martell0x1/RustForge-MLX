use rustforge_mlx::math::vector::Vector;

#[test]
fn vector_has_correct_dimension() {
    let vector = Vector::new(vec![1.0, 2.0, 3.0]);

    assert_eq!(vector.len(), 3);
}

#[test]
fn vector_returns_four_zeros() {
    let vector = Vector::<f64>::zeros(4);
    let expected = vec![0.0; 4];

    assert_eq!(vector.len(), 4);
    assert_eq!(vector.to_vec(), expected);
}

#[test]
fn vector_returns_four_ones() {
    let vector = Vector::<f64>::ones(4);
    let expected = vec![1.0; 4];

    assert_eq!(vector.len(), 4);
    assert_eq!(vector.to_vec(), expected);
}

#[test]
fn vector_can_be_created_from_slice() {
    let arr = [1.0, 2.0, 3.0];
    let vector = Vector::from_slice(&arr);

    assert_eq!(vector.len(), 3);
    assert_eq!(vector.to_vec(), arr.to_vec());
}

#[test]
fn vector_is_not_empty() {
    let vector = Vector::new([1.0, 2.0]);

    assert!(!vector.is_empty());
}

#[test]
fn vector_can_be_created_from_array() {
    let vector = Vector::new([1, 2, 3, 4]);

    assert_eq!(vector.len(), 4);
    assert_eq!(vector.to_vec(), vec![1, 2, 3, 4]);
}

#[test]
fn vector_can_be_created_from_range() {
    let vector = Vector::new(0..5);

    assert_eq!(vector.len(), 5);
    assert_eq!(vector.to_vec(), vec![0, 1, 2, 3, 4]);
}

#[test]
fn vector_can_be_empty() {
    let vector: Vector<f64> = Vector::new([]);

    assert!(vector.is_empty());
    assert_eq!(vector.len(), 0);
}

#[test]
fn vector_get_element() {
    let vector = Vector::<i32>::new(1..3); // 1 2 3
    let second = vector.get(1);
    assert_eq!(second, Some(2));
}

#[test]
fn vector_can_set_items() {
    let mut vector = Vector::<i64>::new(vec![1, 2, 4]);
    vector.set(2, 3);
    dbg!(&vector);

    assert_eq!(vector.get(2), Some(3));
}

#[test]
fn vector_arithmetic_operations() {
    let vec1 = Vector::<i32>::new(1..4);
    let vec2 = Vector::<i32>::new(1..4);

    // Vector + Vector
    let vec3 = vec1.clone() + vec2.clone();
    let res3 = Vector::<i32>::from_slice(&[2, 4, 6]);

    // Vector - Vector
    let vec4 = vec1.clone() - vec2.clone();
    let res4 = Vector::<i32>::zeros(3);

    // Vector * Vector
    let vec5 = vec1.clone() * vec2.clone();
    let res5 = Vector::<i32>::from_slice(&[1, 4, 9]);

    // Vector / Vector
    let vec6 = vec1.clone() / vec2.clone();
    let res6 = Vector::<i32>::ones(3);

    // Vector * Scalar
    let vec7 = vec1.clone() * 5;
    let res7 = Vector::<i32>::from_slice(&[5, 10, 15]);

    assert_eq!(vec3, res3);
    assert_eq!(vec4, res4);
    assert_eq!(vec5, res5);
    assert_eq!(vec6, res6);
    assert_eq!(vec7, res7);
}

#[test]
fn vector_dot_product() {
    let vec1 = Vector::new(vec![1, 2, 3]);
    let vec2 = Vector::new(vec![4, 5, 6]);

    assert_eq!(vec1.dot(&vec2), 32);
}

#[test]
fn vector_norm() {
    let vector = Vector::new(vec![3.0, 4.0]);

    assert_eq!(vector.norm(), 5.0);
}

#[test]
fn vector_normalize() {
    let vector = Vector::new(vec![3.0, 4.0]);

    let result = vector.normalize();
    let expected = Vector::new(vec![0.6, 0.8]);

    assert_eq!(result, expected);
}

#[test]
fn vector_distance() {
    let vec1 = Vector::new(vec![1.0, 2.0]);
    let vec2 = Vector::new(vec![4.0, 6.0]);

    assert_eq!(vec1.distance(&vec2), 5.0);
}

#[test]
fn vector_sum() {
    let vector = Vector::new(vec![1, 2, 3, 4]);

    assert_eq!(vector.sum(), 10);
}

#[test]
fn vector_mean() {
    let vector = Vector::new(vec![1.0, 2.0, 3.0, 4.0]);

    assert_eq!(vector.mean(), 2.5);
}

#[test]
fn vector_min() {
    let vector = Vector::new(vec![4, 2, 8, 1]);

    assert_eq!(vector.min(), Some(&1));
}

#[test]
fn vector_max() {
    let vector = Vector::new(vec![4, 2, 8, 1]);

    assert_eq!(vector.max(), Some(&8));
}

#[test]
fn vector_argmin() {
    let vector = Vector::new(vec![4, 2, 8, 1]);

    assert_eq!(vector.argmin(), Some(3));
}

#[test]
fn vector_argmax() {
    let vector = Vector::new(vec![4, 2, 8, 1]);

    assert_eq!(vector.argmax(), Some(2));
}

#[test]
fn vector_map() {
    let vector = Vector::new(vec![1, 2, 3]);

    let result = vector.map(|x| x * x);
    let expected = Vector::new(vec![1, 4, 9]);

    assert_eq!(result, expected);
}

#[test]
fn vector_zip() {
    let vec1 = Vector::new(vec![1, 2, 3]);
    let vec2 = Vector::new(vec![4, 5, 6]);

    let result = vec1.zip(&vec2);
    let expected = Vector::new(vec![(1, 4), (2, 5), (3, 6)]);

    assert_eq!(result, expected);
}
