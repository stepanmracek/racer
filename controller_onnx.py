import argparse

import msgpack
import onnxruntime
import zmq

from game.common_controller import create_input, output_to_keys
from game.communication import parse_readings_message


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
    input_size = onnx_session.get_inputs()[0].shape[1]
    output_names = [onnx_session.get_outputs()[0].name]

    while True:
        readings = parse_readings_message(subscriber.recv()[len(topic) :])
        onnx_input = create_input(readings, input_size)
        onnx_output = onnx_session.run(output_names, {"input": onnx_input})[0]
        keys = output_to_keys(onnx_output, logits=False)
        publisher.send(msgpack.packb(keys))


if __name__ == "__main__":
    main()
