from argparse import ArgumentParser

import msgpack
import pygame as pg
import zmq

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

    pg.init()
    win = pg.display.set_mode((1280, 768))
    pg.display.set_caption("Racer")
    world = World.create(args.level)

    frame_limit = args.timelimit * 30
    frame = 0
    while frame < frame_limit:

        for event in pg.event.get():
            if event.type == pg.QUIT:
                return

        just_pressed_keys = pg.key.get_just_pressed()
        if just_pressed_keys[pg.K_r]:
            world.reset()
            frame = 0

        outcome = world.step(
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
            + msgpack.packb({"sensors": outcome.red_car[0], "velocity": world.red_car.velocity})
        )

        publisher.send(
            b"blue_car"
            + msgpack.packb({"sensors": outcome.blue_car[0], "velocity": world.blue_car.velocity})
        )

        world.draw(win=win)
        time_left = frame / frame_limit * 1280
        pg.draw.rect(win, (255, 255, 0), (0, 760, time_left, 768))
        pg.display.update()
        frame += 1

        car_keys["red"] = msgpack.loads(red_subscriber.recv())
        car_keys["blue"] = msgpack.loads(blue_subscriber.recv())


if __name__ == "__main__":
    main()
    pg.quit()
