pub struct Vector {
    data: Vec<f64>,
}

impl Vector {
    pub fn new(data: Vec<f64>) ->Self {
        Self {data}
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

}