# API
The API is designed to allow for external services, simulators, functions, and
systems to interface with SEMBAS. This circumvents needing the function under
test to be defined within a Rust project alongside SEMBAS.

## Scenarios
Server is SEMBAS, Client is the Remote Classifier.

### Running SEMBAS
1. Server: Starts listening.
2. Client: Opens connection.
3. Server: Begins Global Search
4. Server: Sends sample to Client.
5. Client: Receives sample, classifies, and returns class.
6. Server: Receives class [ALT1]
7. Server: Begins surfacing algorithm.
8. Server: Sends sample to Client.
9. Client: Receives sample, classifies, and returns class.
10. Server: Receives class [ALT2] 
11. Server: Builds initial halfspace and begins boundary exploration.
12. Server: Sends sample to Client.
13. Client: Receives sample, classifies, and returns class.
14. Server: Receives class [ALT3]
15. End

### Out of Bounds Sample
Precondition: Some sampling process is in progress.
1. Server: Sends sample to Client
2. Client: Receives sample
3. Client: Sample out of bounds, sends OOB flag to Server.
4. Server: Continues as normal.
