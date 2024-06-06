import math
from dataclasses import dataclass, field
from typing import Optional, Literal

import pygame as pg
from tqdm import tqdm

SENSORS_ANGLE_STEP = 15


@dataclass
class StepOutcome:
    crash_velocity: Optional[float] = None
    collected_diamond: Optional[tuple[int, int]] = False


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
    sensors_mask: dict[int, pg.Mask] = field(init=False)
    sensors_rays: dict[int, list[list[tuple[int, int]]]] = field(init=False)

    def __post_init__(self):
        self.images = {
            angle: pg.transform.rotate(self.img, angle)
            for angle in range(0, 360, self.rotation_velocity)
        }
        self.masks = {angle: pg.mask.from_surface(img) for angle, img in self.images.items()}

        self.sensors_mask = {}
        self.sensors_rays = {}
        sensors_img = pg.Surface((1000, 1000), pg.SRCALPHA)
        sensors_img.fill((0, 0, 0, 0))
        pg.draw.ellipse(sensors_img, (0, 0, 0, 255), (250, 0, 500, 750))
        for angle in tqdm(
            range(0, 360, self.rotation_velocity), desc="Pre-computing sensors mask and rays"
        ):
            rotated_sensors_img = pg.transform.rotate(sensors_img, angle)
            rotated_sensors_img = rotated_sensors_img.subsurface(
                (
                    rotated_sensors_img.get_width() / 2 - 500,
                    rotated_sensors_img.get_height() / 2 - 500,
                    1000,
                    1000,
                )
            )
            mask = pg.mask.from_surface(rotated_sensors_img)
            self.sensors_mask[angle] = mask
            self.sensors_rays[angle] = [[] for _ in range(0, 360, SENSORS_ANGLE_STEP)]
            for i, ray_orientation in enumerate(range(0, 360, SENSORS_ANGLE_STEP)):
                radians = math.radians(angle + ray_orientation)
                dy = -math.cos(radians)
                dx = -math.sin(radians)
                distance = 0
                x = 500.0
                y = 500.0
                while distance < 500:
                    if x < 0.0 or y < 0.0 or x >= 1000.0 or y >= 1000.0:
                        break
                    if not mask.get_at((x, y)):
                        break

                    self.sensors_rays[angle][i].append((int(x), int(y)))
                    x = x + dx
                    y = y + dy
                    distance += 1

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

    def left(self, angle: int) -> int:
        if self.velocity > 0.05:
            angle += self.rotation_velocity
        elif self.velocity < -0.05:
            angle -= self.rotation_velocity

        angle = angle % 360
        return angle

    def right(self, angle: int) -> int:
        if self.velocity > 0.01:
            angle -= self.rotation_velocity
        elif self.velocity < -0.01:
            angle += self.rotation_velocity

        angle = angle % 360
        return angle

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
    ) -> StepOutcome:
        new_angle = self.control(up_key, down_key, left_key, right_key)
        new_pos = self.get_next_pos()

        if self.collision(new_angle, new_pos, collision_mask) or self.collision_other_car(
            new_angle, new_pos, other_car
        ):
            crash_velocity = abs(self.velocity)
            self.bounce()
            return StepOutcome(crash_velocity=crash_velocity)

        dsize = diamond_mask.get_size()
        dhw, dhh = dsize[0] / 2, dsize[1] / 2
        collected_diamond = None
        for diamond in diamond_coords:
            if self.collision(new_angle, new_pos, diamond_mask, diamond[0] - dhw, diamond[1] - dhh):
                collected_diamond = diamond
                self.score += 1
                break

        self.x = new_pos[0]
        self.y = new_pos[1]
        self.angle = new_angle
        return StepOutcome(collected_diamond=collected_diamond)

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

    def collision(
        self, candidate_angle: int, candidate_pos: tuple[float, float], mask: pg.Mask, x=0, y=0
    ):
        img = self.images[candidate_angle]
        car_mask = self.masks[candidate_angle]
        r = img.get_rect()
        offset = (candidate_pos[0] - r.centerx - x, candidate_pos[1] - r.centery - y)
        return mask.overlap(car_mask, offset)

    def collision_other_car(
        self, candidate_angle: int, candidate_pos: tuple[float, float], other_car: "Car"
    ):
        this_img = self.images[candidate_angle]
        this_mask = self.masks[candidate_angle]
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
        self.velocity = -0.5 * self.velocity

    def control(
        self,
        up_key: bool,
        down_key: bool,
        left_key: bool,
        right_key: bool,
    ) -> int:
        moved = False
        new_angle = self.angle
        if left_key:
            new_angle = self.left(new_angle)
        if right_key:
            new_angle = self.right(new_angle)
        if up_key:
            moved = True
            self.forward()
        if down_key:
            moved = True
            self.backward()
        if not moved:
            self.idle()
        return new_angle

    def is_object_in_range(
        self, object_x, object_y, sensors_mask: pg.Mask, angle_step
    ) -> Optional[tuple[int, int]]:
        x = int(object_x - self.x) + 500
        y = int(object_y - self.y) + 500
        if 0 <= x < 1000 and 0 <= y < 1000 and sensors_mask.get_at((x, y)):
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
        m: pg.Mask = self.sensors_mask[self.angle]

        # wall collisions
        readings = [None for _ in range(0, 360, SENSORS_ANGLE_STEP)]
        collisions = m.overlap_mask(collision_mask, (-self.x + 500, -self.y + 500))

        for i, ray in enumerate(self.sensors_rays[self.angle]):
            for distance, (x, y) in enumerate(ray):
                if collisions.get_at((x, y)):
                    readings[i] = ("w", distance)
                    break

        # other car in range?
        angle_distance = self.is_object_in_range(other_car.x, other_car.y, m, SENSORS_ANGLE_STEP)
        if angle_distance:
            slot = angle_distance[0] // SENSORS_ANGLE_STEP
            if not readings[slot] or angle_distance[1] < readings[slot][1]:
                readings[slot] = ("e", angle_distance[1])

        # diamonds in range?
        for diamond_pos in diamond_coords:
            angle_distance = self.is_object_in_range(
                diamond_pos[0], diamond_pos[1], m, SENSORS_ANGLE_STEP
            )
            if angle_distance:
                slot = angle_distance[0] // SENSORS_ANGLE_STEP
                if not readings[slot] or angle_distance[1] < readings[slot][1]:
                    readings[slot] = ("d", angle_distance[1])

        return readings
