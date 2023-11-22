use divan::{black_box, Bencher};
use ndarray::{ArrayD, Axis};
use ort::{Session, Value};
use ort_batcher::batcher::Batcher;
use std::time::Duration;

fn main() {
    divan::main();
}

fn load_test_model() -> ort::Result<Session> {
    let session = Session::builder()?
        .with_intra_threads(1)?
        .with_model_from_memory(include_bytes!("../tests/model.onnx"))?;

    Ok(session)
}

#[divan::bench]
fn sequential_without_batcher_16k() {
    let session = load_test_model().unwrap();
    let input = ArrayD::<f32>::zeros(vec![7, 8, 9]).insert_axis(Axis(0));

    for _ in 0..1024 * 16 {
        let value = black_box(Value::from_array(black_box(input.clone())).unwrap());
        black_box(
            session.run([value]).unwrap()[0]
                .extract_tensor::<f32>()
                .unwrap()
                .view()
                .index_axis(Axis(0), 0)
                .to_owned(),
        );
    }
}

#[allow(non_snake_case)]
#[divan::bench(
    // number of producers
    threads = [1024],
    // batch size
    consts = [1, 32, 64, 128, 256, 512, 1024]
)]
fn batch_sizes_1024t_16i_16k<const MAX_BATCH_SIZE: usize>(bencher: Bencher) {
    let session = load_test_model().unwrap();
    let batcher = Batcher::spawn(session, MAX_BATCH_SIZE, Duration::from_millis(10));
    let inputs = vec![ArrayD::zeros(vec![7, 8, 9])];

    bencher.bench(move || {
        for _ in 0..16 {
            black_box(batcher.run(black_box(inputs.clone()))).unwrap();
        }
    });
}
