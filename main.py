import math
from dataclasses import dataclass, field

import pygame as pg


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

    images: dict[int, pg.Surface] = field(init=False)
    masks: dict[int, pg.mask.Mask] = field(init=False)
    crash_sound: pg.mixer.Sound = field(init=False)

    def __post_init__(self):
        self.images = {
            angle: pg.transform.rotate(self.img, angle)
            for angle in range(0, 360, self.rotation_velocity)
        }
        self.masks = {
            angle: pg.mask.from_surface(img) for angle, img in self.images.items()
        }
        self.crash_sound = pg.mixer.Sound("assets/sound/crash.mp3")
        self.crash_sound.set_volume(0.5)
        self.init_vals = {
            "x": self.x,
            "y": self.y,
            "velocity": self.velocity,
            "angle": self.angle,
        }

    def reset(self):
        self.x = self.init_vals["x"]
        self.y = self.init_vals["y"]
        self.velocity = self.init_vals["velocity"]
        self.angle = self.init_vals["angle"]

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
        pressed_keys: pg.key.ScancodeWrapper,
        collision_mask: pg.Mask,
        up_key: int,
        down_key: int,
        left_key: int,
        right_key: int,
        other_car: "Car",
        diamond_coords: set[tuple[int, int]],
        diamond_mask: pg.mask.Mask,
        diamond_sfx: pg.mixer.Sound,
    ):
        self.control(pressed_keys, up_key, down_key, left_key, right_key)
        new_pos = self.get_next_pos()

        if self.collision(new_pos, collision_mask) or self.collision_other_car(
            new_pos, other_car
        ):
            self.bounce()
            return

        for diamond in diamond_coords:
            if self.collision(new_pos, diamond_mask, *diamond):
                diamond_coords.remove(diamond)
                diamond_sfx.play()
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
        pressed_keys: pg.key.ScancodeWrapper,
        up_key: int,
        down_key: int,
        left_key: int,
        right_key: int,
    ):
        moved = False
        if pressed_keys[left_key]:
            self.left()
        if pressed_keys[right_key]:
            self.right()
        if pressed_keys[up_key]:
            moved = True
            self.forward()
        if pressed_keys[down_key]:
            moved = True
            self.backward()
        if not moved:
            self.idle()


def init_diamonds():
    return {
        (250, 250),
        (500, 500),
        (450, 300),
        (750, 400),
        (950, 150),
        (1000, 600),
        (150, 450),
    }


@dataclass
class World:
    background: pg.Surface
    map: pg.Surface
    blue_car: Car
    red_car: Car
    diamond_image: pg.Surface
    diamond_coords: set[tuple[int, int]]

    def reset(self):
        self.red_car.reset()
        self.blue_car.reset()
        self.diamond_coords = init_diamonds()

    def draw(self, win: pg.Surface):
        win.blit(self.background, (0, 0))
        win.blit(self.map, (0, 0))
        for diamond_pos in self.diamond_coords:
            win.blit(self.diamond_image, diamond_pos)
        self.red_car.draw(win)
        self.blue_car.draw(win)
        pg.display.update()


def main():
    pg.init()
    RED_CAR = scale_image(pg.image.load("assets/cars/red.png"), 0.75)
    BLUE_CAR = scale_image(pg.image.load("assets/cars/blue.png"), 0.75)
    GRASS = pg.image.load("assets/grass-fullhd.png")
    COLLISION = pg.image.load("assets/collision-test.png")
    COLLISION_MASK = pg.mask.from_surface(COLLISION)
    DIAMOND = pg.image.load("assets/diamond.png")
    DIAMOND_MASK = pg.mask.from_surface(DIAMOND)
    DIAMOND_SFX = pg.mixer.Sound("assets/sound/money.mp3")

    world = World(
        background=GRASS,
        map=COLLISION,
        blue_car=Car(BLUE_CAR, 640, 200, 180),
        red_car=Car(RED_CAR, 640, 600, 0),
        diamond_image=DIAMOND,
        diamond_coords=init_diamonds(),
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
            pressed_keys=pressed_keys,
            collision_mask=COLLISION_MASK,
            up_key=pg.K_UP,
            down_key=pg.K_DOWN,
            left_key=pg.K_LEFT,
            right_key=pg.K_RIGHT,
            other_car=world.blue_car,
            diamond_coords=world.diamond_coords,
            diamond_mask=DIAMOND_MASK,
            diamond_sfx=DIAMOND_SFX,
        )
        world.blue_car.step(
            pressed_keys=pressed_keys,
            collision_mask=COLLISION_MASK,
            up_key=pg.K_w,
            down_key=pg.K_s,
            left_key=pg.K_a,
            right_key=pg.K_d,
            other_car=world.red_car,
            diamond_coords=world.diamond_coords,
            diamond_mask=DIAMOND_MASK,
            diamond_sfx=DIAMOND_SFX,
        )

        world.draw(win)


if __name__ == "__main__":
    main()
    pg.quit()
