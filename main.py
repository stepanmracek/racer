import math
import time
from dataclasses import dataclass, field

import pygame


def scale_image(img, factor):
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return pygame.transform.smoothscale(img, size)


def background_tile(win: pygame.Surface, background: pygame.Surface):
    w, h = win.get_width(), win.get_height()
    bgnd_w, bgnd_h = background.get_width(), background.get_height()
    rows = math.ceil(w / bgnd_w)
    cols = math.ceil(h / bgnd_h)

    for r in range(rows):
        for c in range(cols):
            win.blit(background, (r * bgnd_w, c * bgnd_h))


@dataclass
class Car:
    img: pygame.Surface
    x: float
    y: float
    angle: int
    velocity: float = 0.0
    max_velocity: float = 10.0
    rotation_velocity: int = 2
    acceleration: float = 0.1
    last_bounce = -1.0

    images: dict[int, pygame.Surface] = field(init=False)
    crash_sound: pygame.mixer.Sound = field(init=False)

    def __post_init__(self):
        self.images = {
            angle: pygame.transform.rotate(self.img, angle)
            for angle in range(0, 360, self.rotation_velocity)
        }
        self.crash_sound = pygame.mixer.Sound("assets/sound/crash.mp3")
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

    def draw(self, win: pygame.Surface):
        img = self.images[self.angle]
        r = img.get_rect()
        win.blit(img, (self.x - r.centerx, self.y - r.centery))

    def bounced_recently(self):
        return (time.perf_counter() - self.last_bounce) < 0.5 and abs(
            self.velocity
        ) > 0.1

    def left(self):
        if self.bounced_recently():
            return

        if self.velocity > 0.05:
            self.angle += self.rotation_velocity
        elif self.velocity < -0.05:
            self.angle -= self.rotation_velocity

        self.angle = (self.angle + 360) % 360

    def right(self):
        if self.bounced_recently():
            return

        if self.velocity > 0.01:
            self.angle -= self.rotation_velocity
        elif self.velocity < -0.01:
            self.angle += self.rotation_velocity

        self.angle = (self.angle + 360) % 360

    def forward(self):
        if self.bounced_recently():
            return

        if self.velocity >= 0:
            # accelerating forward
            self.velocity = min(self.velocity + self.acceleration, self.max_velocity)
        else:
            # breaking when reversing
            self.velocity = max(self.velocity + self.acceleration * 2, 0)

    def backward(self):
        if self.bounced_recently():
            return

        if self.velocity > 0:
            # breaking when moving forward
            self.velocity = max(self.velocity - self.acceleration * 2, 0)
        else:
            # accelerating when reversing
            self.velocity = min(
                self.velocity - self.acceleration, self.max_velocity / 2
            )

    def step(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.velocity
        horizontal = math.sin(radians) * self.velocity
        self.y -= vertical
        self.x -= horizontal

    def idle(self):
        if self.velocity > 0:
            self.velocity = max(self.velocity - self.acceleration * 0.25, 0.0)
        elif self.velocity < 0:
            self.velocity = min(self.velocity + self.acceleration * 0.25, 0.0)

    def collision(self, mask: pygame.Mask, x=0, y=0):
        img = self.images[self.angle]
        car_mask = pygame.mask.from_surface(img)
        r = img.get_rect()
        offset = (self.x - r.centerx - x, self.y - r.centery - y)
        return mask.overlap(car_mask, offset)

    def collision_other_car(self, other_car: "Car"):
        this_img = self.images[self.angle]
        this_mask = pygame.mask.from_surface(this_img)
        this_r = this_img.get_rect()

        other_img = other_car.images[other_car.angle]
        other_mask = pygame.mask.from_surface(other_img)
        other_r = other_img.get_rect()

        offset = (
            (self.x - this_r.centerx) - (other_car.x - other_r.centerx),
            (self.y - this_r.centery) - (other_car.y - other_r.centery),
        )
        return other_mask.overlap(this_mask, offset)

    def bounce(self):
        if self.bounced_recently():
            return

        if abs(self.velocity) > 1.0:
            self.crash_sound.play()

        self.last_bounce = time.perf_counter()
        self.velocity = -0.25 * self.velocity

        if abs(self.velocity) > 0.1 and abs(self.velocity) < 1.0:
            self.velocity = 1.0 if self.velocity > 0 else -1.0


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


def car_control(
    pressed_keys: pygame.key.ScancodeWrapper,
    collision_mask: pygame.Mask,
    car: Car,
    other_car: Car,
    up_key: int,
    down_key: int,
    left_key: int,
    right_key: int,
):
    moved = False
    if pressed_keys[left_key]:
        car.left()
    if pressed_keys[right_key]:
        car.right()
    if pressed_keys[up_key]:
        moved = True
        car.forward()
    if pressed_keys[down_key]:
        moved = True
        car.backward()
    if not moved:
        car.idle()

    if car.collision(collision_mask) or car.collision_other_car(other_car):
        car.bounce()


def main():
    pygame.init()
    RED_CAR = scale_image(pygame.image.load("assets/cars/red.png"), 0.75)
    BLUE_CAR = scale_image(pygame.image.load("assets/cars/blue.png"), 0.75)
    GRASS = pygame.image.load("assets/grass-fullhd.png")
    COLLISION = pygame.image.load("assets/collision-test.png")
    COLLISION_MASK = pygame.mask.from_surface(COLLISION)
    DIAMOND = pygame.image.load("assets/diamond.png")
    DIAMOND_MASK = pygame.mask.from_surface(DIAMOND)
    DIAMOND_SFX = pygame.mixer.Sound("assets/sound/money.mp3")

    win = pygame.display.set_mode((1280, 768))
    pygame.display.set_caption("Racer")
    clock = pygame.time.Clock()
    red_car = Car(RED_CAR, 640, 600, 0)
    blue_car = Car(BLUE_CAR, 640, 200, 180)
    diamonds = init_diamonds()

    while True:
        clock.tick(30)

        win.blit(GRASS, (0, 0))
        win.blit(COLLISION, (0, 0))
        for diamond in diamonds:
            win.blit(DIAMOND, diamond)
        red_car.draw(win)
        blue_car.draw(win)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        pressed_keys = pygame.key.get_pressed()

        if pressed_keys[pygame.K_r]:
            red_car.reset()
            blue_car.reset()
            diamonds = init_diamonds()

        car_control(
            pressed_keys,
            COLLISION_MASK,
            red_car,
            blue_car,
            pygame.K_UP,
            pygame.K_DOWN,
            pygame.K_LEFT,
            pygame.K_RIGHT,
        )
        car_control(
            pressed_keys,
            COLLISION_MASK,
            blue_car,
            red_car,
            pygame.K_w,
            pygame.K_s,
            pygame.K_a,
            pygame.K_d,
        )

        for diamond in diamonds:
            if red_car.collision(DIAMOND_MASK, *diamond) or blue_car.collision(
                DIAMOND_MASK, *diamond
            ):
                diamonds.remove(diamond)
                DIAMOND_SFX.play()
                break

        red_car.step()
        blue_car.step()


if __name__ == "__main__":
    main()
    pygame.quit()
