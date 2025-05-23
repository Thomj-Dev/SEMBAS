import socket
import struct
from time import sleep

import torch


def wait_until_open(
    client: socket.socket, max_attempts: int | None = 10, delay: float = 0.1
):
    "Provides time for the server to begin before giving up."
    success = False
    i = 0
    while not success and (max_attempts is None or i < max_attempts):
        try:
            client.connect(("127.0.0.1", 2000))
            success = True
        except:
            sleep(delay)

        i += 1


def setup_socket(ndim: int, max_attempts: int = None, fail_on_refuse=False):
    "Create the FUT's connection to SEMBAS"
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    if fail_on_refuse:
        client.connect(("127.0.0.1", 2000))
    else:
        wait_until_open(client, max_attempts)

    # send config to remote classifier
    try:
        ndim_packed = struct.pack("!q", ndim)
        client.sendall(ndim_packed)

        msg = client.recv(1024).decode("utf-8")
        if msg != "OK\n":
            raise Exception("Invalid number of dimensions?")
    except Exception as e:
        client.close()
        raise e

    return client


def receive_request(client: socket.socket, ndim: int) -> torch.Tensor:
    "Receives a request from SEMBAS, i.e. an input to classify."
    data_size = ndim * 8  # ndim * size(f64)
    data = client.recv(data_size)
    return torch.tensor(struct.unpack(f"{ndim}d", data))


def send_response(client: socket.socket, cls: bool):
    "Sends a response to SEMBAS, i.e. the class of the input it requested."
    bool_byte = int(cls).to_bytes(1, byteorder="big")
    client.sendall(bool_byte)


def send_message(client: socket.socket, msg: str):
    f = client.makefile("wb")
    f.writelines([msg])


def receive_message(client: socket.socket) -> str:
    f = client.makefile("rb")
    return f.readline()
