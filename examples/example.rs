use ndarray::{ArrayD, Axis};
use ort::{CUDAExecutionProvider, Environment, SessionBuilder, Value};
use ort_batcher::batcher::Batcher;
use std::time::Duration;

fn main() -> ort::Result<()> {
    tracing_subscriber::fmt::init();

    let environment = Environment::builder()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .build()?
        .into_arc();
    let session = SessionBuilder::new(&environment)?
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
