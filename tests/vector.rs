use rustforge_mlx::math::vector::Vector;

#[test]

fn vector_has_correct_dimention() {
    let vector = Vector::new(vec![1.0,2.0,3.0]);

    assert_eq!(vector.len(),3);
}