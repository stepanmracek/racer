import argparse

import math
import msgpack
import pygame as pg
import zmq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="red_car")
    topic: bytes = parser.parse_args().topic.encode()

    context = zmq.Context.instance()
    subscriber: zmq.Socket = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.CONFLATE, 1)
    subscriber.connect("tcp://localhost:6000")
    subscriber.setsockopt(zmq.SUBSCRIBE, topic)

    publisher: zmq.Socket = context.socket(zmq.PUB)
    publisher.bind("tcp://localhost:6001")

    pg.init()
    win = pg.display.set_mode((500 + 50, 750))
    pg.display.set_caption(f"Controller for '{topic.decode()}'")
    clock = pg.time.Clock()

    while True:
        clock.tick(1000)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return

        readings = msgpack.loads(subscriber.recv()[len(topic) :])
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
        publisher.send(topic + msgpack.packb(keys))

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
        velocity = readings["velocity"] * 40
        if velocity >= 0:
            pg.draw.rect(win, (64, 64, 192), (500, 500-velocity, 50, velocity))
        else:
            pg.draw.rect(win, (192, 64, 64), (500, 500, 50, abs(velocity)))
        pg.display.update()


if __name__ == "__main__":
    main()
