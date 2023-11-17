use ndarray::{ArrayD, Axis, IxDyn};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use ort::{Environment, Session, SessionBuilder, Value};
use ort_batcher::batcher::Batcher;
use std::time::Duration;

fn load_test_model() -> ort::Result<Session> {
    let environment = Environment::builder().build()?.into_arc();
    let session = SessionBuilder::new(&environment)?
        .with_intra_threads(1)?
        .with_model_from_memory(include_bytes!("../tests/model.onnx"))?;

    Ok(session)
}

fn generate_samples(session: &Session, n: usize) -> Vec<(ArrayD<f32>, ArrayD<f32>)> {
    let mut samples = Vec::new();

    for _ in 0..n {
        let input = ArrayD::<f32>::random(IxDyn(&[7, 6, 4]), Uniform::new(0., 1.));
        let input_1 = input.clone().insert_axis(Axis(0));
        let value = Value::from_array(input_1).unwrap();
        let expected_output = session.run([value]).unwrap()[0]
            .extract_tensor()
            .unwrap()
            .view()
            .index_axis(Axis(0), 0)
            .to_owned();

        samples.push((input, expected_output));
    }

    samples
}

fn check_batch(samples: usize, max_batch_size: usize, max_wait_time: Duration) {
    let session = load_test_model().unwrap();
    let samples = generate_samples(&session, samples);
    let batcher = Batcher::spawn(session, max_batch_size, max_wait_time);

    for (input, expected_output) in samples {
        let output = &batcher.run(vec![input]).unwrap()[0];
        assert!(expected_output.abs_diff_eq(output, 1e5));
    }
}

#[test]
fn one_per_batch() {
    check_batch(20, 1, Duration::ZERO);
}

#[test]
fn odd_per_batch() {
    check_batch(20, 7, Duration::from_millis(10));
}

#[test]
fn chaos() {
    let session = load_test_model().unwrap();
    let samples = generate_samples(&session, 10);
    let batcher = Batcher::spawn(session, 4, Duration::from_millis(50));

    std::thread::scope(|s| {
        for _ in 0..4 {
            let batcher = &batcher;
            let samples = samples.clone();

            s.spawn(move || {
                let mut rng = rand::thread_rng();

                for (input, expected_output) in samples {
                    std::thread::yield_now();
                    std::thread::sleep(Duration::from_millis(rand::Rng::gen_range(
                        &mut rng,
                        0..150,
                    )));
                    let output = &batcher.run(vec![input]).unwrap()[0];
                    assert!(expected_output.abs_diff_eq(output, 1e5));
                }
            });
        }
    });

    batcher.stop();
}
