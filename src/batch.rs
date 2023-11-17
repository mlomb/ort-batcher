use ndarray::{ArrayD, Axis, Slice};
use ort::{Session, Value};

/// A batch of samples
pub struct Batch {
    /// Current number of samples in the batch
    size: usize,

    /// Maximum number of samples that can fit in the batch
    capacity: usize,

    /// Tensor buffers
    batch_inputs: Vec<ArrayD<f32>>,
}

impl Batch {
    pub fn new(session: &Session, capacity: usize) -> Self {
        // create buffers for all the inputs
        // so samples can be added without allocating
        let batch_tensors: Vec<ArrayD<f32>> = session
            .inputs
            .iter()
            .map(|input| {
                let shape = input
                    .input_type
                    .tensor_dimensions()
                    .expect("input must be a tensor, other types are unsupported")
                    .clone();

                // TODO: an assert for the data type (ort does not expose this info currently...)
                assert!(shape[0] == -1, "model must support dynamic batch size");
                assert!(shape[1..].iter().all(|&d| d != -1), "only one dynamic"); // is this even possible?

                let mut shape: Vec<usize> = shape
                    .into_iter()
                    .map(|d| d as usize) // i64 -> usize
                    .collect();

                // reserve for the biggest batch
                shape[0] = capacity;

                // allocate
                ArrayD::zeros(shape)
            })
            .collect();

        Batch {
            size: 0,
            capacity,
            batch_inputs: batch_tensors,
        }
    }

    /// Add a sample to the batch
    pub fn add_sample(&mut self, inputs: Vec<ArrayD<f32>>) {
        assert!(self.has_room(), "batch is full");
        assert_eq!(self.batch_inputs.len(), inputs.len(), "inputs mismatch");

        for (batch_inputs, input) in self.batch_inputs.iter_mut().zip(inputs) {
            batch_inputs
                .index_axis_mut(Axis(0), self.size)
                // this will panic if the input is not the right shape
                .assign(&input)
        }

        self.size += 1;
    }

    /// Whether the batch has room for more samples
    pub fn has_room(&self) -> bool {
        self.size < self.capacity
    }

    /// Convert the inputs into ort::Value-s
    fn get_input_values(&self) -> ort::Result<Vec<Value>> {
        let mut values = Vec::with_capacity(self.batch_inputs.len());

        for input in &self.batch_inputs {
            // only keep the samples that were added to the batch
            let samples_slice = input.slice_axis(Axis(0), Slice::from(0..self.size));

            // Note: from_array makes a copy of the data.
            // I'm not sure how to avoid it, since we have to mutate the data
            // and adjust the final tensor size to match the batch size
            values.push(Value::from_array(samples_slice)?);
        }

        Ok(values)
    }

    /// Runs inference in all the samples currently stored in the batch
    ///
    /// Returns a vector of tensors for each sample, in the same order they were added to the batch
    pub fn run(&self, session: &Session) -> ort::Result<Vec<Vec<ArrayD<f32>>>> {
        let batch_outputs = session.run(self.get_input_values()?.as_slice())?;
        let mut outputs = Vec::with_capacity(self.size);

        for sample_index in 0..self.size {
            let mut sample_outputs = Vec::with_capacity(batch_outputs.len());

            for (_, value) in batch_outputs.iter() {
                sample_outputs.push(
                    value
                        .extract_tensor::<f32>()?
                        .view()
                        // extract the result from that sample
                        .index_axis(Axis(0), sample_index)
                        // **copy**
                        .to_owned(),
                )
            }

            outputs.push(sample_outputs);
        }

        Ok(outputs)
    }

    /// Reset the batch
    pub fn clear(&mut self) {
        self.size = 0;
    }
}
