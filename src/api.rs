use crate::prelude::Sample;
use crate::structs::SamplingError;
use nalgebra::SVector;
use std::io::Write;
use std::io::{self, Read};
use std::net;

use crate::structs::error;
use crate::structs::Classifier;
use crate::structs::Domain;

const BUFFER_CONFIG_SIZE: usize = 8;

/// Allows an external function under test to connect to SEMBAS and request
/// where to sample next. The classifier can then be called just like any other
/// classifier.
pub struct RemoteClassifier<const N: usize> {
    stream: net::TcpStream,
    domain: Domain<N>,
}

impl<const N: usize> RemoteClassifier<N> {
    /// Opens a socket to be connected to by a remote function under test (FUT).  
    /// Once a connection is established, the RemoteClassifier will send the points
    /// to the FUT to be classified, and the FUT will return the resulting class
    /// (bool).
    /// ## Connection Sequence
    /// 1. RemoteClassifier binds to TcpListener.
    /// 2. FUT connects to socket.
    /// 3. RemoteClassifier accepts connection.
    /// 4. FUT sends config containing number of params info
    /// 5. RemoteClassifier accepts configuration, throwing error if N != num params
    /// 6. RemoteClassifier sends back 'OK\n'
    /// 7. RemoteClassifier setup complete, ready to classify.
    pub fn bind(addr: String) -> io::Result<Self> {
        let listener = net::TcpListener::bind(addr)?;
        println!("Listening for client connection...");
        let (mut stream, _) = listener.accept()?;
        println!("Connection established.");

        println!("Waiting for sim config...");
        let mut buffer = [0u8; BUFFER_CONFIG_SIZE];
        stream.read_exact(&mut buffer)?;
        let num_params = usize::from_be_bytes(buffer);

        if num_params != N {
            stream
                .write_all(format!("{N}\n").as_bytes())
                .expect("Invalid N write to stream?");

            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("Invalid number of param names! Expected {N}, Got {num_params}"),
            ));
        }

        stream
            .write_all("OK\n".as_bytes())
            .expect("Invalid 'OK' write to stream?");

        println!("Got valid config. Ready.");

        let domain = Domain::<N>::normalized();
        Ok(RemoteClassifier { stream, domain })
    }
}

impl<const N: usize> Drop for RemoteClassifier<N> {
    fn drop(&mut self) {
        let buffer = "end\n".as_bytes();
        self.stream
            .write_all(buffer)
            .expect("Invalid 'end' write to stream?")
    }
}

impl From<io::Error> for SamplingError {
    fn from(value: io::Error) -> Self {
        SamplingError::InvalidClassifierResponse(format!(
            "Invalid client response message. IO Error: {value}"
        ))
    }
}

impl<const N: usize> Classifier<N> for RemoteClassifier<N> {
    fn classify(&mut self, p: SVector<f64, N>) -> error::Result<Sample<N>> {
        if !self.domain.contains(&p) {
            return Err(SamplingError::OutOfBounds);
        }

        // Send request
        let bytes: &[u8] = bytemuck::cast_slice(p.as_slice());
        self.stream.write_all(bytes)?;

        let mut buffer = [0; 1];
        self.stream.read_exact(&mut buffer)?;
        if buffer[0] > 1 {
            Err(SamplingError::InvalidClassifierResponse(
                "Remote Classifier received non-bool response?".to_string(),
            ))
        } else {
            Ok(Sample::from_class(p, buffer[0] == 1))
        }
    }
}
