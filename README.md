# ort-batcher

<a href="https://crates.io/crates/ort_batcher" target="_blank">
    <img alt="Crates.io" src="https://img.shields.io/crates/v/ort_batcher?style=for-the-badge&logo=rust">
</a>


Small crate to batch inferences of ONNX models using [ort](https://github.com/pykeio/ort). Inspired by [batched_fn](https://docs.rs/batched-fn/latest/batched_fn/).

Note that it only works with models that:
* Have their first dimension dynamic (-1), so they can be batched.
* Inputs and outputs are tensors of type `float32`.

# Usage

```rust
let max_batch_size = 32;
let max_wait_time = Duration::from_millis(80);
let batcher = Batcher::spawn(session, max_batch_size, max_wait_time);

// in some thread
let inputs = vec![ArrayD::<f32>::zeros(vec![7, 8, 9])];
let outputs = batcher.run(inputs).unwrap();
```

# Example

Check [example.rs](examples/example.rs):

```rust
use ndarray::{ArrayD, Axis};
use ort::{CUDAExecutionProvider, Environment, SessionBuilder, Value};
use ort_batcher::batcher::Batcher;
use std::time::Duration;

fn main() -> ort::Result<()> {
    tracing_subscriber::fmt::init();

    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;

    let session = Session::builder()?
        .with_intra_threads(1)?
        .with_model_from_memory(include_bytes!("../tests/model.onnx"))?;

    {
        let start = std::time::Instant::now();

        // 128 threads
        // 256 inferences each
        // sequential
        std::thread::scope(|s| {
            for _ in 0..128 {
                let session = &session;
                let input = ArrayD::<f32>::zeros(vec![7, 8, 9]);

                s.spawn(move || {
                    for _ in 0..256 {
                        let value = Value::from_array(input.clone().insert_axis(Axis(0))).unwrap();
                        let _output = session.run([value]).unwrap()[0]
                            .extract_tensor::<f32>()
                            .unwrap()
                            .view()
                            .index_axis(Axis(0), 0)
                            .to_owned();
                    }
                });
            }
        });

        println!("sequential: {:?}", start.elapsed());
    }

    let max_batch_size = 32;
    let max_wait_time = Duration::from_millis(10);
    let batcher = Batcher::spawn(session, max_batch_size, max_wait_time);

    {
        let start = std::time::Instant::now();

        // 128 threads
        // 256 inferences each
        // batched
        std::thread::scope(|s| {
            for _ in 0..128 {
                let batcher = &batcher;
                let input = ArrayD::<f32>::zeros(vec![7, 8, 9]);

                s.spawn(move || {
                    for _ in 0..256 {
                        let _output = batcher.run(vec![input.clone()]).unwrap();
                    }
                });
            }
        });

        println!("batched: {:?}", start.elapsed());
    }

    Ok(())
}
```

Note that to have good results you have to use heavy model in a GPU, otherwise you may not see any difference.
