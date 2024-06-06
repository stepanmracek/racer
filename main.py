from argparse import ArgumentParser
from threading import Thread
from typing import Literal

import msgpack
import pygame as pg
import zmq

from car import Car, Sensors
from utils import init_diamonds, scale_image
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
    red_car_img = scale_image(pg.image.load("assets/cars/red.png").convert_alpha(), 0.75)
    blue_car_img = scale_image(pg.image.load("assets/cars/blue.png").convert_alpha(), 0.75)
    background_img = pg.image.load(f"assets/maps/{args.level}/bg.png").convert()
    collision_img = pg.image.load(f"assets/maps/{args.level}/map.png").convert_alpha()
    background_img.blit(collision_img, (0, 0))
    spawn_mask = pg.mask.from_surface(
        pg.image.load(f"assets/maps/{args.level}/spawn-mask.png").convert_alpha()
    )
    collision_mask = pg.mask.from_surface(collision_img)
    diamond_img = pg.image.load("assets/diamond.png").convert_alpha()
    diamond_mask = pg.mask.from_surface(diamond_img)
    diamond_sfx = pg.mixer.Sound("assets/sound/money.mp3")
    crash_sfx = pg.mixer.Sound("assets/sound/crash.mp3")
    crash_sfx.set_volume(0.5)
    font = pg.font.Font(None, 42)

    sensors = Sensors.precompute()
    world = World(
        background=background_img,
        blue_car=Car(img=blue_car_img, x=640, y=200, angle=180, sensors=sensors),
        red_car=Car(img=red_car_img, x=640, y=600, angle=0, sensors=sensors),
        diamond_image=diamond_img,
        diamond_coords=init_diamonds(spawn_mask),
        spawn_mask=spawn_mask,
        diamond_sfx=diamond_sfx,
        crash_sfx=crash_sfx,
    )
    pg.display.set_caption("Racer")
    clock = pg.time.Clock()

    show_sensor_readings = 0

    while True:
        clock.tick(30)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                return

        pressed_keys = pg.key.get_pressed()
        just_pressed_keys = pg.key.get_just_pressed()

        if just_pressed_keys[pg.K_r]:
            world.reset()
        if just_pressed_keys[pg.K_t]:
            show_sensor_readings = (show_sensor_readings + 1) % 4

        world.process_step_outcome(
            world.red_car.step(
                collision_mask=collision_mask,
                up_key=car_keys["red"]["u"] or pressed_keys[pg.K_UP],
                down_key=car_keys["red"]["d"] or pressed_keys[pg.K_DOWN],
                left_key=car_keys["red"]["l"] or pressed_keys[pg.K_LEFT],
                right_key=car_keys["red"]["r"] or pressed_keys[pg.K_RIGHT],
                other_car=world.blue_car,
                diamond_coords=world.diamond_coords,
                diamond_mask=diamond_mask,
            )
        )

        world.process_step_outcome(
            world.blue_car.step(
                collision_mask=collision_mask,
                up_key=car_keys["blue"]["u"] or pressed_keys[pg.K_w],
                down_key=car_keys["blue"]["d"] or pressed_keys[pg.K_s],
                left_key=car_keys["blue"]["l"] or pressed_keys[pg.K_a],
                right_key=car_keys["blue"]["r"] or pressed_keys[pg.K_d],
                other_car=world.red_car,
                diamond_coords=world.diamond_coords,
                diamond_mask=diamond_mask,
            )
        )

        red_car_readings = world.red_car.sensor_readings(
            collision_mask, world.blue_car, world.diamond_coords
        )
        blue_car_readings = world.blue_car.sensor_readings(
            collision_mask, world.red_car, world.diamond_coords
        )

        publisher.send(
            b"red_car"
            + msgpack.packb({"sensors": red_car_readings, "velocity": world.red_car.velocity})
        )

        publisher.send(
            b"blue_car"
            + msgpack.packb({"sensors": blue_car_readings, "velocity": world.blue_car.velocity})
        )

        world.draw(win)
        if show_sensor_readings & 1:
            world.draw_readings(win, world.red_car, red_car_readings)
        if show_sensor_readings & 2:
            world.draw_readings(win, world.blue_car, blue_car_readings)

        win.blit(
            font.render(f"{world.red_car.score}", True, (192, 32, 32), (0, 0, 0)),
            (win.get_width() / 2 - 50, win.get_height() - 36),
        )
        win.blit(
            font.render(f"{world.blue_car.score}", True, (32, 32, 192), (0, 0, 0)),
            (win.get_width() / 2 + 50, win.get_height() - 36),
        )

        pg.display.update()


if __name__ == "__main__":
    main()
    pg.quit()
