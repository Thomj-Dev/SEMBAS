use std::{
    io::{Read, Write},
    net::TcpStream,
    thread::{self, JoinHandle},
};

use nalgebra::SVector;
use sembas::{
    api::RemoteClassifier,
    sps::Sphere,
    structs::{Classifier, Domain},
};

/// The number of dimensions that the UAS simulation has.
const D: usize = 7;
const SAMPLE_SIZE: usize = D * 8;

/// This example is to illustrate how the RemoteClassifier works. Since this requires
/// two independent processes, we will be using a thread for both the SEMBAS program
/// and the Function Under Test (FUT).
/// Normally, you would *not* use the RemoteClassifier unless you cannot import
/// sembas crate. However, to keep things in one program we will have both the FUT
/// and the SEMBAS test solution in the same program.
/// If you have an application in a Python script, or other such programming
/// language, this can act as a versatile bridge between applications.
fn main() {
    // First start your SEMBAS program:
    let local_program_handle = setup_local();

    // Then begin you FUT (Pretend this is a different program)
    let other_program_handle = setup_remote::<D>();

    // Once they connect we can simply begin sending requests by using classifier as

    // We wait for the SEMBAS program to terminate first, since it is handling the
    // setup and teardown process.
    local_program_handle.join().unwrap();
    // The other program will then be ready to terminate.
    other_program_handle.join().unwrap();
}

fn setup_local() -> JoinHandle<()> {
    thread::spawn(|| {
        // Begin RemoteClassifier server
        let mut classifier = setup_classifier();

        let p1 = SVector::from_fn(|_, _| 0.5);
        println!("[Local] Sending {p1:?} request");
        let r = classifier.classify(p1).unwrap();
        println!("[Local] Received result of {r}");

        let p2 = SVector::zeros();
        println!("[Local] Sending {p1:?} request");
        let r = classifier.classify(p2).unwrap();
        println!("[Local] Received result of {r}");
    })
}

fn setup_remote<const N: usize>() -> JoinHandle<()> {
    thread::spawn(|| {
        println!("[Remote] Beginning connection...");
        let mut socket = TcpStream::connect("127.0.0.1:2000").unwrap();
        println!("[Remote] Connection established!");

        println!("[Remote] Sending config (dimension size)...");
        let config = N.to_be_bytes();
        socket.write_all(&config).unwrap();

        println!("[Remote] Waiting for OK...");
        let mut buffer = [0u8; 1024];
        let n = socket.read(&mut buffer).unwrap();

        let ack = std::str::from_utf8(&buffer[..n])
            .expect("Got bad message?")
            .to_string();

        println!("Got acknowledgement of {:?} back.", ack.trim());

        if ack.trim() != "OK" {
            println!("[Remote] Invalid configuration, exiting.");
            return;
        }
        println!("[Remote] Configuration accepted. Waiting for requests.");

        // This would be some function that you're testing that has some well-defined
        // performance mode. Example: an AV test scenario that runs, from which a
        // summary is returned, which you parse and return true if a collision
        // occurred otherwise false.
        let mut classifier = Sphere::new(
            SVector::<f64, N>::from_fn(|_, _| 0.5),
            0.25,
            Some(Domain::normalized()),
        );

        loop {
            let mut buffer = [0u8; SAMPLE_SIZE];
            let n = socket.read(&mut buffer).unwrap();

            // Expecting a N*8 byte sized message, which is the SVector<f64, N>
            if n < SAMPLE_SIZE {
                // In case it is not, then it's a termination signal or error
                let msg = std::str::from_utf8(&buffer[..n])
                    .expect("Received unexpected response from server")
                    .to_string();

                if msg.trim() == "end" {
                    println!("[Remote] SEMBAS Completed, ending session.");
                    break;
                } else {
                    panic!("Got unexpected message: {:?}.", msg.trim());
                }
            }

            let p: SVector<f64, N> = SVector::from_row_slice(
                bytemuck::try_cast_slice::<u8, f64>(&buffer)
                    .expect("Incorrect byte size or alignment"),
            );

            let cls = classifier
                .classify(p)
                .unwrap_or(sembas::structs::Sample::OutOfMode(p.into()))
                .class() as u8;

            socket.write_all(&[cls]).unwrap();
        }
    })
}

fn setup_classifier() -> Box<dyn Classifier<D>> {
    Box::new(RemoteClassifier::bind("127.0.0.1:2000".to_string()).expect("Failed to connect"))
}
