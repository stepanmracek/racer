import math
import random
from dataclasses import dataclass, field
from threading import Thread
from typing import Literal, Optional

import msgpack
import pygame as pg
import zmq
from tqdm import tqdm


def scale_image(img, factor):
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return pg.transform.smoothscale(img, size)


@dataclass
class Car:
    img: pg.Surface
    x: float
    y: float
    angle: int
    velocity: float = 0.0
    max_velocity: float = 10.0
    rotation_velocity: int = 2
    acceleration: float = 0.1
    score: int = 0

    images: dict[int, pg.Surface] = field(init=False)
    masks: dict[int, pg.Mask] = field(init=False)
    crash_sound: pg.mixer.Sound = field(init=False)
    sensors_mask: dict[int, pg.Mask] = field(init=False)

    def __post_init__(self):
        self.images = {
            angle: pg.transform.rotate(self.img, angle)
            for angle in range(0, 360, self.rotation_velocity)
        }
        self.masks = {angle: pg.mask.from_surface(img) for angle, img in self.images.items()}

        self.sensors_mask = {}
        sensors_img = pg.Surface((1000, 1000), pg.SRCALPHA)
        sensors_img.fill((0, 0, 0, 0))
        pg.draw.ellipse(sensors_img, (0, 0, 0, 255), (250, 0, 500, 750))
        for angle in tqdm(range(0, 360, self.rotation_velocity), desc="Pre-computing sensors mask"):
            rotated_sensors_img = pg.transform.rotate(sensors_img, angle)
            rotated_sensors_img = rotated_sensors_img.subsurface(
                (
                    rotated_sensors_img.get_width() / 2 - 500,
                    rotated_sensors_img.get_height() / 2 - 500,
                    1000,
                    1000,
                )
            )
            self.sensors_mask[angle] = pg.mask.from_surface(rotated_sensors_img)

        self.crash_sound = pg.mixer.Sound("assets/sound/crash.mp3")
        self.crash_sound.set_volume(0.5)
        self.init_vals = {"x": self.x, "y": self.y, "velocity": self.velocity, "angle": self.angle}

    def reset(self):
        self.x = self.init_vals["x"]
        self.y = self.init_vals["y"]
        self.velocity = self.init_vals["velocity"]
        self.angle = self.init_vals["angle"]
        self.score = 0

    def draw(self, win: pg.Surface):
        img = self.images[self.angle]
        r = img.get_rect()
        win.blit(img, (self.x - r.centerx, self.y - r.centery))

    def left(self):
        if self.velocity > 0.05:
            self.angle += self.rotation_velocity
        elif self.velocity < -0.05:
            self.angle -= self.rotation_velocity

        self.angle = self.angle % 360

    def right(self):
        if self.velocity > 0.01:
            self.angle -= self.rotation_velocity
        elif self.velocity < -0.01:
            self.angle += self.rotation_velocity

        self.angle = self.angle % 360

    def forward(self):
        if self.velocity >= 0:
            # accelerating forward
            self.velocity = min(self.velocity + self.acceleration, self.max_velocity)
        else:
            # breaking when reversing
            self.velocity = min(self.velocity + self.acceleration * 2.0, 0.0)

    def backward(self):
        if self.velocity > 0:
            # breaking when moving forward
            self.velocity = max(self.velocity - self.acceleration * 2.0, 0.0)
        else:
            # accelerating when reversing
            max_reverse_speed = -self.max_velocity / 2.0
            self.velocity = max(self.velocity - self.acceleration, max_reverse_speed)

    def step(
        self,
        collision_mask: pg.Mask,
        up_key: bool,
        down_key: bool,
        left_key: bool,
        right_key: bool,
        other_car: "Car",
        diamond_coords: set[tuple[int, int]],
        diamond_mask: pg.Mask,
        diamond_sfx: pg.mixer.Sound,
        spawn_mask: pg.Mask,
    ):
        self.control(up_key, down_key, left_key, right_key)
        new_pos = self.get_next_pos()

        if self.collision(new_pos, collision_mask) or self.collision_other_car(new_pos, other_car):
            self.bounce()
            return

        dsize = diamond_mask.get_size()
        dhw, dhh = dsize[0] / 2, dsize[1] / 2
        for diamond in diamond_coords:
            if self.collision(new_pos, diamond_mask, diamond[0] - dhw, diamond[1] - dhh):
                diamond_coords.remove(diamond)
                diamond_coords.add(random_pos(spawn_mask))
                diamond_sfx.play()
                self.score += 1
                break

        self.x = new_pos[0]
        self.y = new_pos[1]

    def get_next_pos(self) -> tuple[float, float]:
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.velocity
        horizontal = math.sin(radians) * self.velocity
        return self.x - horizontal, self.y - vertical

    def idle(self):
        if self.velocity > 0:
            self.velocity = max(self.velocity - self.acceleration * 0.25, 0.0)
        elif self.velocity < 0:
            self.velocity = min(self.velocity + self.acceleration * 0.25, 0.0)

    def collision(self, candidate_pos: tuple[float, float], mask: pg.Mask, x=0, y=0):
        img = self.images[self.angle]
        car_mask = self.masks[self.angle]
        r = img.get_rect()
        offset = (candidate_pos[0] - r.centerx - x, candidate_pos[1] - r.centery - y)
        return mask.overlap(car_mask, offset)

    def collision_other_car(self, candidate_pos: tuple[float, float], other_car: "Car"):
        this_img = self.images[self.angle]
        this_mask = self.masks[self.angle]
        this_r = this_img.get_rect()

        other_img = other_car.images[other_car.angle]
        other_mask = other_car.masks[other_car.angle]
        other_r = other_img.get_rect()

        offset = (
            (candidate_pos[0] - this_r.centerx) - (other_car.x - other_r.centerx),
            (candidate_pos[1] - this_r.centery) - (other_car.y - other_r.centery),
        )
        return other_mask.overlap(this_mask, offset)

    def bounce(self):
        if abs(self.velocity) > 1.0:
            self.crash_sound.play()

        self.velocity = -0.5 * self.velocity

    def control(
        self,
        up_key: bool,
        down_key: bool,
        left_key: bool,
        right_key: bool,
    ):
        moved = False
        if left_key:
            self.left()
        if right_key:
            self.right()
        if up_key:
            moved = True
            self.forward()
        if down_key:
            moved = True
            self.backward()
        if not moved:
            self.idle()

    def is_object_in_range(
        self, object_x, object_y, sensors_mask: pg.Mask, angle_step
    ) -> Optional[tuple[int, int]]:
        x = int(object_x - self.x) + 500
        y = int(object_y - self.y) + 500
        if x >= 0 and y >= 0 and x < 1000 and y < 1000 and sensors_mask.get_at((x, y)):
            x -= 500
            y -= 500
            distance = int(math.hypot(x, y))
            angle = (
                (-(self.angle - (90 - math.degrees(math.atan2(-y, -x))))) + angle_step / 2
            ) % 360
            return int(angle), distance

    def sensor_readings(
        self,
        collision_mask: pg.Mask,
        other_car: "Car",
        diamond_coords: set[tuple[int, int]],
    ) -> list[Optional[tuple[Literal["w", "d", "e"], int]]]:
        ANGLE_STEP = 15
        m: pg.Mask = self.sensors_mask[self.angle]

        # wall collisions
        readings = [None for _ in range(0, 360, ANGLE_STEP)]
        collisions = m.overlap_mask(collision_mask, (-self.x + 500, -self.y + 500))
        for i, a in enumerate(range(0, 360, ANGLE_STEP)):
            radians = math.radians(self.angle + a)
            dy = -math.cos(radians)
            dx = -math.sin(radians)
            dh = math.hypot(dx, dy)
            distance = 0.0
            x = 500.0
            y = 500.0
            while distance < 1000:
                x = x + dx
                y = y + dy
                distance += dh
                if x >= 0 and y >= 0 and x < 1000 and y < 1000 and collisions.get_at((x, y)):
                    readings[i] = ("w", int(distance))
                    break

        # other car in range?
        angle_distance = self.is_object_in_range(other_car.x, other_car.y, m, ANGLE_STEP)
        if angle_distance:
            slot = angle_distance[0] // ANGLE_STEP
            if not readings[slot] or angle_distance[1] < readings[slot][1]:
                readings[slot] = ("e", angle_distance[1])

        # diamonds in range?
        for diamond_pos in diamond_coords:
            angle_distance = self.is_object_in_range(diamond_pos[0], diamond_pos[1], m, ANGLE_STEP)
            if angle_distance:
                slot = angle_distance[0] // ANGLE_STEP
                if not readings[slot] or angle_distance[1] < readings[slot][1]:
                    readings[slot] = ("d", angle_distance[1])

        return readings


def random_pos(spawn_mask: pg.Mask) -> tuple[int, int]:
    w, h = spawn_mask.get_size()
    while True:
        pos = random.randint(0, w - 1), random.randint(0, h - 1)
        if spawn_mask.get_at(pos):
            return pos


def init_diamonds(spawn_mask: pg.Mask):
    return {random_pos(spawn_mask) for _ in range(3)}


@dataclass
class World:
    background: pg.Surface
    collision: pg.Surface
    blue_car: Car
    red_car: Car
    diamond_image: pg.Surface
    diamond_coords: set[tuple[int, int]]
    spawn_mask: pg.Mask

    def reset(self):
        self.red_car.reset()
        self.blue_car.reset()
        self.diamond_coords = init_diamonds(self.spawn_mask)

    def draw_readings(self, win: pg.Surface, car: Car, readings: list):
        win.blit(
            car.sensors_mask[car.angle].to_surface(setcolor=(0, 0, 0, 32), unsetcolor=(0, 0, 0, 0)),
            (car.x - 500, car.y - 500),
        )
        what_color = {"w": (255, 255, 0), "e": (255, 0, 0), "d": (0, 0, 255)}
        for i, r in enumerate(readings):
            if not r:
                continue
            what, distance = r
            radians = math.radians(car.angle + i * 15)
            dy = -math.cos(radians)
            dx = -math.sin(radians)

            pg.draw.line(
                win,
                (192, 192, 192),
                (car.x, car.y),
                (car.x + dx * distance, car.y + dy * distance),
            )
            pg.draw.circle(
                win,
                what_color[what],
                (car.x + dx * distance, car.y + dy * distance),
                3 if what == "w" else 5,
            )

    def draw(self, win: pg.Surface):
        win.blit(self.background, (0, 0))
        win.blit(self.collision, (0, 0))

        dhw = self.diamond_image.get_width() / 2
        dhh = self.diamond_image.get_height() / 2
        for diamond_pos in self.diamond_coords:
            win.blit(self.diamond_image, (diamond_pos[0] - dhw, diamond_pos[1] - dhh))

        self.red_car.draw(win)
        self.blue_car.draw(win)


red_car_keys = {"u": False, "d": False, "l": False, "r": False}


def key_subscriber(car_color: Literal[b"red_car", b"blue_car"]):
    context = zmq.Context.instance()
    subscriber: zmq.Socket = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.CONFLATE, 1)
    subscriber.connect("tcp://localhost:6001")
    topic = car_color
    subscriber.setsockopt(zmq.SUBSCRIBE, topic)
    global red_car_keys
    while True:
        red_car_keys = msgpack.loads(subscriber.recv()[len(topic) :])


def main():
    context = zmq.Context.instance()
    publisher: zmq.Socket = context.socket(zmq.PUB)
    publisher.bind("tcp://localhost:6000")

    red_car_keys_subscriber_thread = Thread(target=key_subscriber, args=(b"red_car",), daemon=True)
    red_car_keys_subscriber_thread.start()

    pg.init()
    RED_CAR = scale_image(pg.image.load("assets/cars/red.png"), 0.75)
    BLUE_CAR = scale_image(pg.image.load("assets/cars/blue.png"), 0.75)
    BAKGROUND = pg.image.load("assets/maps/park/bg.png")
    COLLISION = pg.image.load("assets/maps/park/map.png")
    SPAWN_MASK = pg.mask.from_surface(pg.image.load("assets/maps/park/spawn-mask.png"))
    COLLISION_MASK = pg.mask.from_surface(COLLISION)
    DIAMOND = pg.image.load("assets/diamond.png")
    DIAMOND_MASK = pg.mask.from_surface(DIAMOND)
    DIAMOND_SFX = pg.mixer.Sound("assets/sound/money.mp3")
    FONT = pg.font.Font(None, 42)

    world = World(
        background=BAKGROUND,
        collision=COLLISION,
        blue_car=Car(BLUE_CAR, 640, 200, 180),
        red_car=Car(RED_CAR, 640, 600, 0),
        diamond_image=DIAMOND,
        diamond_coords=init_diamonds(SPAWN_MASK),
        spawn_mask=SPAWN_MASK,
    )
    win = pg.display.set_mode((1280, 768))
    pg.display.set_caption("Racer")
    clock = pg.time.Clock()

    while True:
        clock.tick(30)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                return

        pressed_keys = pg.key.get_pressed()

        if pressed_keys[pg.K_r]:
            world.reset()

        world.red_car.step(
            collision_mask=COLLISION_MASK,
            up_key=red_car_keys["u"],  # pressed_keys[pg.K_UP],
            down_key=red_car_keys["d"],  # pressed_keys[pg.K_DOWN],
            left_key=red_car_keys["l"],  # pressed_keys[pg.K_LEFT],
            right_key=red_car_keys["r"],  # pressed_keys[pg.K_RIGHT],
            other_car=world.blue_car,
            diamond_coords=world.diamond_coords,
            diamond_mask=DIAMOND_MASK,
            diamond_sfx=DIAMOND_SFX,
            spawn_mask=SPAWN_MASK,
        )
        world.blue_car.step(
            collision_mask=COLLISION_MASK,
            up_key=pressed_keys[pg.K_w],
            down_key=pressed_keys[pg.K_s],
            left_key=pressed_keys[pg.K_a],
            right_key=pressed_keys[pg.K_d],
            other_car=world.red_car,
            diamond_coords=world.diamond_coords,
            diamond_mask=DIAMOND_MASK,
            diamond_sfx=DIAMOND_SFX,
            spawn_mask=SPAWN_MASK,
        )

        red_car_readings = world.red_car.sensor_readings(
            COLLISION_MASK, world.blue_car, world.diamond_coords
        )
        blue_car_readings = world.blue_car.sensor_readings(
            COLLISION_MASK, world.red_car, world.diamond_coords
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
        world.draw_readings(win, world.red_car, red_car_readings)
        # world.draw_readings(win, world.blue_car, blue_car_readings)

        win.blit(
            FONT.render(f"{world.red_car.score}", True, (192, 32, 32), (0, 0, 0)),
            (win.get_width() / 2 - 50, win.get_height() - 36),
        )
        win.blit(
            FONT.render(f"{world.blue_car.score}", True, (32, 32, 192), (0, 0, 0)),
            (win.get_width() / 2 + 50, win.get_height() - 36),
        )

        pg.display.update()


if __name__ == "__main__":
    main()
    pg.quit()
