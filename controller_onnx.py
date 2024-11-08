import argparse
from typing import Literal, Optional

import msgpack
import numpy as np
import onnxruntime
import zmq

from game.communication import parse_readings_message


def parse_sensor(
    sensors: list[Optional[tuple[Literal["w", "d", "e"], int]]], letters, max_val=501.0
):
    ans = []
    for s in sensors:
        if not s:
            ans.append(1.0)
            continue
        if s[0] not in letters:
            ans.append(1.0)
            continue
        ans.append(s[1] / max_val)
    return np.array(ans, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--car", default="red", choices=["red", "blue"])
    parser.add_argument("--onnx", required=True)
    args = parser.parse_args()
    topic: bytes = args.car.encode() + b"_car"

    context = zmq.Context.instance()
    subscriber: zmq.Socket = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.CONFLATE, 1)
    subscriber.connect("tcp://localhost:6000")
    subscriber.setsockopt(zmq.SUBSCRIBE, topic)

    publisher: zmq.Socket = context.socket(zmq.PUB)
    publisher.bind(f"tcp://localhost:{6001 if args.car == 'red' else 6002}")

    onnx_session = onnxruntime.InferenceSession(args.onnx)

    while True:
        readings = parse_readings_message(subscriber.recv()[len(topic) :])

        velocity = np.array([readings["velocity"]], dtype=np.float32)
        sensors = readings["sensors"]
        walls = parse_sensor(sensors, ("w", "e"))
        diamonds = parse_sensor(sensors, ("d"))
        onnx_input = np.array([np.concatenate((velocity, walls, diamonds))])

        onnx_output = onnx_session.run(["output_0"], {"input": onnx_input})[0][0]
        # print(onnx_output)

        # hack to not stop and wait until the end of the time if we have zero velocity
        # and the network decides to not move at all
        if abs(velocity[0]) < 0.1 and onnx_output[0] < 0.5 and onnx_output[1] < 0.5:
            onnx_output[0] = 1.0

        keys = {
            "u": bool(onnx_output[0] > 0.5),
            "d": bool(onnx_output[1] > 0.5),
            "l": bool(onnx_output[2] > 0.5),
            "r": bool(onnx_output[3] > 0.5),
        }
        publisher.send(msgpack.packb(keys))


if __name__ == "__main__":
    main()
