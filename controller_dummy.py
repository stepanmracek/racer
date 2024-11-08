import argparse

import msgpack
import zmq

from game.communication import parse_control_message


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--car", default="red", choices=["red", "blue"])
    args = parser.parse_args()
    topic: bytes = args.car.encode() + b"_car"

    context = zmq.Context.instance()
    subscriber: zmq.Socket = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.CONFLATE, 1)
    subscriber.connect("tcp://localhost:6000")
    subscriber.setsockopt(zmq.SUBSCRIBE, topic)

    publisher: zmq.Socket = context.socket(zmq.PUB)
    publisher.bind(f"tcp://localhost:{6001 if args.car == 'red' else 6002}")

    keys =  msgpack.packb({"u": False, "d": False, "l": False, "r": False})
    while True:
        sensor_readings = parse_control_message(subscriber.recv()[len(topic) :])
        # Do something with sensor_readings
        # ...
        # ...
        publisher.send(keys)


if __name__ == "__main__":
    main()
