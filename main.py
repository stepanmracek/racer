from argparse import ArgumentParser
from threading import Thread
from typing import Literal

import msgpack
import pygame as pg
import zmq

from world import World

car_keys = {
    "red": {"u": False, "d": False, "l": False, "r": False},
    "blue": {"u": False, "d": False, "l": False, "r": False},
}


def key_subscriber(car_color: Literal["red", "blue"]):
    context = zmq.Context.instance()
    subscriber: zmq.Socket = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.CONFLATE, 1)
    subscriber.connect(f"tcp://localhost:{6001 if car_color == 'red' else 6002}")
    subscriber.setsockopt(zmq.SUBSCRIBE, b'')
    global car_keys
    while True:
        try:
            car_keys[car_color] = msgpack.loads(subscriber.recv())
        except Exception as ex:
            print(f"Exception during receiving of {car_color} keys:", ex.__class__.__name__, ex)


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--level", default="park", choices=["park", "nyan"])
    arg_parser.add_argument("--fps", default=30, type=int)
    args = arg_parser.parse_args()

    context = zmq.Context.instance()
    publisher: zmq.Socket = context.socket(zmq.PUB)
    publisher.bind("tcp://localhost:6000")

    red_car_keys_thread = Thread(target=key_subscriber, args=("red",), daemon=True)
    red_car_keys_thread.start()

    blue_car_keys_thread = Thread(target=key_subscriber, args=("blue",), daemon=True)
    blue_car_keys_thread.start()

    pg.init()
    win = pg.display.set_mode((1280, 768))
    pg.display.set_caption("Racer")
    world = World.create(args.level)
    clock = pg.time.Clock()

    show_sensor_readings = 0

    while True:
        clock.tick(args.fps)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                return

        pressed_keys = pg.key.get_pressed()
        just_pressed_keys = pg.key.get_just_pressed()

        if just_pressed_keys[pg.K_r]:
            world.reset()
        if just_pressed_keys[pg.K_t]:
            show_sensor_readings = (show_sensor_readings + 1) % 4

        red_car_readings, blue_car_readings = world.step(
            red_up=car_keys["red"]["u"] or pressed_keys[pg.K_UP],
            red_down=car_keys["red"]["d"] or pressed_keys[pg.K_DOWN],
            red_left=car_keys["red"]["l"] or pressed_keys[pg.K_LEFT],
            red_right=car_keys["red"]["r"] or pressed_keys[pg.K_RIGHT],
            blue_up=car_keys["blue"]["u"] or pressed_keys[pg.K_w],
            blue_down=car_keys["blue"]["d"] or pressed_keys[pg.K_s],
            blue_left=car_keys["blue"]["l"] or pressed_keys[pg.K_a],
            blue_right=car_keys["blue"]["r"] or pressed_keys[pg.K_d],
        )

        publisher.send(
            b"red_car"
            + msgpack.packb({"sensors": red_car_readings, "velocity": world.red_car.velocity})
        )

        publisher.send(
            b"blue_car"
            + msgpack.packb({"sensors": blue_car_readings, "velocity": world.blue_car.velocity})
        )

        world.draw(
            win=win,
            red_car_readings=red_car_readings if show_sensor_readings & 1 else None,
            blue_car_readings=blue_car_readings if show_sensor_readings & 2 else None,
        )
        pg.display.update()


if __name__ == "__main__":
    main()
    pg.quit()
