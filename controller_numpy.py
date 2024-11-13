import argparse
from dataclasses import dataclass

import msgpack
import numpy as np
import zmq

from game.common_controller import create_input, output_to_keys
from game.communication import parse_readings_message


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
                weights.append(
                    np.frombuffer(next(unpacker), dtype=np.float32).reshape(shape)
                )
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
    input_size = models[0].weights[0].shape[0]

    while True:
        readings = parse_readings_message(subscriber.recv()[len(topic) :])
        model_input = create_input(readings, input_size)
        models_output = np.stack([model(model_input)[0] for model in models])
        keys = output_to_keys(models_output, logits=True)
        publisher.send(msgpack.packb(keys))


if __name__ == "__main__":
    main()
