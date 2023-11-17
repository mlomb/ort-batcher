pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    /// An ort error occurred
    OrtError(String),

    /// The batcher thread was not available
    DeadBatcher,
}

// TODO: errors for back pressure (Busy), invalid shapes, invalid number of inputs, etc

impl From<&ort::Error> for Error {
    fn from(error: &ort::Error) -> Self {
        // since ort::Error cant be copied, all we can do is convert it to a string
        Self::OrtError(error.to_string())
    }
}
