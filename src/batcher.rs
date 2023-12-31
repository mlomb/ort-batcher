use crate::{batch::Batch, error::*};
use flume::{unbounded, Sender};
use ndarray::ArrayD;
use ort::Session;
use std::{thread::JoinHandle, time::Duration};

enum Message {
    /// Indicates the batcher to stop processing and exit
    Terminate,

    /// A sample to be processed
    Sample(Vec<ArrayD<f32>>, Sender<Result<Vec<ArrayD<f32>>>>),
}

/// A sample batcher.
///
/// It implements Send and Sync so it can be shared between threads.
pub struct Batcher {
    /// Channel to send samples to the batch thread
    tx: Sender<Message>,

    /// Batcher's thread handle to be able to join it after stopping
    handle: Option<JoinHandle<()>>,
}

impl Batcher {
    /// Swpans a new batcher thread
    pub fn spawn(session: Session, max_batch_size: usize, max_wait_time: Duration) -> Self {
        let (tx, rx) = unbounded();

        let handle = std::thread::spawn(move || {
            let mut batch_txs = Vec::with_capacity(max_batch_size);
            let mut batch = Batch::new(&session, max_batch_size);

            // wait for the first input in the batch
            while let Ok(Message::Sample(inputs, result_tx)) = rx.recv() {
                let deadline = std::time::Instant::now() + max_wait_time;

                batch.add_sample(inputs);
                batch_txs.push(result_tx);

                // wait for more inputs to come in
                while batch.has_room() {
                    if let Ok(message) = rx.recv_deadline(deadline) {
                        match message {
                            Message::Terminate => return,
                            Message::Sample(inputs, result_tx) => {
                                batch.add_sample(inputs);
                                batch_txs.push(result_tx);
                            }
                        }
                    } else {
                        // timed out, no new inputs
                        break;
                    }
                }

                match batch.run(&session) {
                    Ok(outputs) => {
                        for (tx, outputs) in batch_txs.iter().zip(outputs.into_iter()) {
                            tx.send(Ok(outputs)).unwrap();
                        }
                    }
                    Err(ref error) => {
                        for tx in &batch_txs {
                            tx.send(Err(Error::from(error))).unwrap();
                        }
                    }
                }

                batch_txs.clear();
                batch.clear();
            }
        });

        Self {
            tx,
            handle: Some(handle),
        }
    }

    pub fn run(&self, inputs: Vec<ArrayD<f32>>) -> Result<Vec<ArrayD<f32>>> {
        // create a channel to get the data back
        let (result_tx, result_rx) = unbounded();

        // send the input to the batch thread
        self.tx
            .send(Message::Sample(inputs, result_tx))
            .map_err(|_| Error::DeadBatcher)?;

        // wait for the result
        result_rx.recv().map_err(|_| Error::DeadBatcher)?
    }
}

impl Drop for Batcher {
    fn drop(&mut self) {
        // notify the batch thread to stop
        _ = self.tx.send(Message::Terminate);

        // wait for it
        self.handle.take().unwrap().join().unwrap();
    }
}
