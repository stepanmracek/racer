import argparse
import csv
import math

import msgpack
import pygame as pg
import zmq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--car", default="red", choices=["red", "blue"])
    parser.add_argument("--output")
    parser.add_argument("--fps", default=30, type=int)
    args = parser.parse_args()
    topic: bytes = args.car.encode() + b"_car"

    context = zmq.Context.instance()
    subscriber: zmq.Socket = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.CONFLATE, 1)
    subscriber.connect("tcp://localhost:6000")
    subscriber.setsockopt(zmq.SUBSCRIBE, topic)

    publisher: zmq.Socket = context.socket(zmq.PUB)
    publisher.bind(f"tcp://localhost:{6001 if args.car == 'red' else 6002}")

    pg.init()
    win = pg.display.set_mode((500 + 50, 750))
    pg.display.set_caption(f"Controller for {args.car} car")
    clock = pg.time.Clock()

    if args.output:
        output_file = open(args.output, "at")
        csv_writer = csv.writer(output_file)
    else:
        output_file = None
        csv_writer = None

    last_velocity_zero = True
    while True:
        clock.tick(args.fps)
        quit_event = next((event for event in pg.event.get() if event.type == pg.QUIT), None)
        if quit_event:
            break

        readings = msgpack.loads(subscriber.recv()[len(topic) :])
        velocity = readings["velocity"]
        pressed_keys = pg.key.get_pressed()

        up, down, left, right = False, False, False, False
        if pressed_keys[pg.K_LEFT]:
            left = True
        if pressed_keys[pg.K_RIGHT]:
            right = True
        if pressed_keys[pg.K_UP]:
            up = True
        if pressed_keys[pg.K_DOWN]:
            down = True

        keys = {"u": up, "d": down, "l": left, "r": right}
        publisher.send(msgpack.packb(keys))

        if csv_writer:
            if abs(velocity) < 1e-3 and last_velocity_zero:
                # do not dump data if car is not moving and was also not moving in last frame
                pass
            else:
                sensors = [f"{s[0]}-{s[1]}" if s else None for s in readings["sensors"]]
                csv_writer.writerow((velocity, *sensors, int(up), int(down), int(left), int(right)))

        win.fill((0, 0, 0))
        what_color = {"w": (255, 255, 0), "e": (255, 0, 0), "d": (0, 0, 255)}
        for i, r in enumerate(readings["sensors"]):
            if not r:
                continue
            what, distance = r
            radians = math.radians(i * 15)
            dy = -math.cos(radians)
            dx = -math.sin(radians)

            pg.draw.line(
                win, (192, 192, 192), (250, 500), (250 + dx * distance, 500 + dy * distance)
            )
            pg.draw.circle(
                win,
                what_color[what],
                (250 + dx * distance, 500 + dy * distance),
                3 if what == "w" else 5,
            )

        pg.draw.rect(win, (64, 64, 64), (500, 0, 50, 750))
        if velocity >= 0:
            pg.draw.rect(win, (64, 64, 192), (500, 500 - velocity * 40, 50, velocity * 40))
        else:
            pg.draw.rect(win, (192, 64, 64), (500, 500, 50, abs(velocity) * 40))
        pg.display.update()

        last_velocity_zero = abs(velocity) < 1e-3

    if output_file:
        output_file.close()


if __name__ == "__main__":
    main()
