import argparse
from dataclasses import dataclass
from typing import Literal, Optional

import msgpack
import numpy as np
import zmq


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


@dataclass
class NumpyModel:
    weights: list[np.ndarray]

    def __call__(self, input) -> np.ndarray:
        hidden = np.tanh(np.matmul(input, self.weights[0]) + self.weights[1])
        output = np.matmul(hidden, self.weights[2]) + self.weights[3]
        return output

    @classmethod
    def load(cls, path) -> "NumpyModel":
        with open(path, "rb") as f:
            unpacker = msgpack.Unpacker(f)
            weights = []
            for shape in unpacker:
                weights.append(np.frombuffer(next(unpacker), dtype=np.float32).reshape(shape))
        return NumpyModel(weights=weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--car", default="red", choices=["red", "blue"])
    parser.add_argument("--model", required=True, nargs="+")
    args = parser.parse_args()
    topic: bytes = args.car.encode() + b"_car"

    context = zmq.Context.instance()
    subscriber: zmq.Socket = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.CONFLATE, 1)
    subscriber.connect("tcp://localhost:6000")
    subscriber.setsockopt(zmq.SUBSCRIBE, topic)

    publisher: zmq.Socket = context.socket(zmq.PUB)
    publisher.bind(f"tcp://localhost:{6001 if args.car == 'red' else 6002}")

    models = [NumpyModel.load(p) for p in args.model]

    while True:
        readings = msgpack.loads(subscriber.recv()[len(topic) :])

        velocity = np.array([readings["velocity"]], dtype=np.float32)
        sensors = readings["sensors"]
        walls = parse_sensor(sensors, ("w", "e"))
        diamants = parse_sensor(sensors, ("d"))
        model_input = np.array([np.concatenate((velocity, walls, diamants))])
        models_output = np.stack([model(model_input)[0] for model in models])
        model_output = np.sum(np.sign(models_output), axis=0)

        keys = {
            "u": bool(model_output[0] > 0),
            "d": bool(model_output[1] > 0),
            "l": bool(model_output[2] > 0),
            "r": bool(model_output[3] > 0),
        }
        publisher.send(msgpack.packb(keys))


if __name__ == "__main__":
    main()
