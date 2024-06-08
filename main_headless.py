from argparse import ArgumentParser

import msgpack
import pygame as pg
import zmq
from tqdm import tqdm

from world import World


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--level", default="park", choices=["park", "nyan"])
    arg_parser.add_argument("--timelimit", default=60, type=int)
    arg_parser.add_argument("--scorelimit", default=10, type=int)
    args = arg_parser.parse_args()

    context = zmq.Context.instance()
    publisher: zmq.Socket = context.socket(zmq.PUB)
    publisher.bind("tcp://localhost:6000")

    red_subscriber: zmq.Socket = context.socket(zmq.SUB)
    red_subscriber.setsockopt(zmq.CONFLATE, 1)
    red_subscriber.connect("tcp://localhost:6001")
    red_subscriber.setsockopt(zmq.SUBSCRIBE, b"")

    blue_subscriber: zmq.Socket = context.socket(zmq.SUB)
    blue_subscriber.setsockopt(zmq.CONFLATE, 1)
    blue_subscriber.connect("tcp://localhost:6001")
    blue_subscriber.setsockopt(zmq.SUBSCRIBE, b"")

    car_keys = {
        "red": {"u": False, "d": False, "l": False, "r": False},
        "blue": {"u": False, "d": False, "l": False, "r": False},
    }

    pg.display.set_caption("Racer")
    world = World.create(args.level, headless=True)

    for frame in tqdm(range(args.timelimit * 30), desc="Game progress"):
        red_car_readings, blue_car_readings = world.step(
            red_up=car_keys["red"]["u"],
            red_down=car_keys["red"]["d"],
            red_left=car_keys["red"]["l"],
            red_right=car_keys["red"]["r"],
            blue_up=car_keys["blue"]["u"],
            blue_down=car_keys["blue"]["d"],
            blue_left=car_keys["blue"]["l"],
            blue_right=car_keys["blue"]["r"],
        )

        if world.red_car.score >= args.scorelimit or world.blue_car.score >= args.scorelimit:
            break

        publisher.send(
            b"red_car"
            + msgpack.packb({"sensors": red_car_readings, "velocity": world.red_car.velocity})
        )

        publisher.send(
            b"blue_car"
            + msgpack.packb({"sensors": blue_car_readings, "velocity": world.blue_car.velocity})
        )

        frame += 1

        car_keys["red"] = msgpack.loads(red_subscriber.recv())
        car_keys["blue"] = msgpack.loads(blue_subscriber.recv())


if __name__ == "__main__":
    main()
    pg.quit()
