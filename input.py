# from tensorflow.keras.datasets import mnist
import math
import socket
import json
import sys

FLOAT_PRECISION = 13
SCALE = 1 << FLOAT_PRECISION

INT32_MAX = (1 << 31) - 1
UINT32_MOD = 1 << 31


def to_signed_32bit(x):
    x = x % UINT32_MOD
    return x if x <= INT32_MAX else x - UINT32_MOD


def float_to_mytype_signed(a):
    raw = int(math.floor(a * SCALE))
    signed = to_signed_32bit(raw)
    return signed


def mytype_signed_to_float(x):
    return (x % UINT32_MOD) / SCALE


def send_share_only(client_id, request_id, inputs, port, host="127.0.0.1"):
    payload = {"client_id": client_id, "request_id": request_id, "inputs": inputs}
    message = json.dumps(payload).encode("utf-8")

    sock = socket.create_connection((host, port))
    sock.sendall(message)
    # print(f"Sent request to port {port}")
    return sock


def receive_response(sock, port):
    response_data = sock.recv(16384)
    if response_data:
        # print(f"Received response from port {port}")
        response = json.loads(response_data.decode("utf-8"))["output"]
    else:
        response = None
    sock.close()
    return response


if __name__ == "__main__":
    id = int(sys.argv[1])

    # (train_images, train_labels), (_, _) = mnist.load_data()
    # image = (train_images[id] / 255.0).flatten()
    image = [float(x) for x in open("Meteor/files/preload/SecureML/input_0").readlines()[id].split()]

    shares = [[], [], []]
    for pixel in image:
        shares[0].append(pixel)
        shares[1].append(0)
        shares[2].append(0)

    sockets = []
    sockets.append(send_share_only("alice", id, [shares[0], shares[1]], port=5000))
    sockets.append(send_share_only("alice", id, [shares[1], shares[2]], port=5001))
    sockets.append(send_share_only("alice", id, [shares[2], shares[0]], port=5002))

    out1 = receive_response(sockets[0], 5000)
    out2 = receive_response(sockets[1], 5001)
    out3 = receive_response(sockets[2], 5002)

    output = []

    for x in range(len(out1)):
        output.append(mytype_signed_to_float(out1[x][0] + (out1[x][1][0] + out1[x][1][1] + out2[x][1][1])))

    print("Output:", output)
    # print("Image ID:", id, " Label:", train_labels[id])
