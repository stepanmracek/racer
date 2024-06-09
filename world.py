import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pygame as pg

import const
from car import Car, Sensors, SensorReadings, StepOutcome
from utils import init_diamonds, random_pos, scale_image


@dataclass
class WorldStepOutcome:
    red_car: tuple[SensorReadings, StepOutcome]
    blue_car: tuple[SensorReadings, StepOutcome]


@dataclass
class World:
    background: pg.Surface
    collision_mask: pg.Mask
    blue_car: Car
    red_car: Car
    diamond_image: pg.Surface
    diamond_mask: pg.Mask
    diamond_sfx: Optional[pg.mixer.Sound]
    diamond_coords: set[tuple[int, int]]
    crash_sfx: Optional[pg.mixer.Sound]
    spawn_mask: pg.Mask
    font: Optional[pg.Font]

    collision_matrix: np.ndarray = field(init=False)

    def __post_init__(self):
        collision_surface = self.collision_mask.to_surface()
        self.collision_matrix = pg.surfarray.array_red(collision_surface) > 127

    def reset(self):
        self.red_car.reset()
        self.blue_car.reset()
        self.diamond_coords = init_diamonds(self.spawn_mask)

    def draw_readings(self, win: pg.Surface, car: Car, readings: SensorReadings):
        win.blit(
            car.sensors.masks[car.angle].to_surface(
                setcolor=(0, 0, 0, 32), unsetcolor=(0, 0, 0, 0)
            ),
            (car.x - const.SENSORS_SIZE / 2, car.y - const.SENSORS_SIZE / 2),
        )
        what_color = {"w": (255, 255, 0), "e": (255, 0, 0), "d": (0, 0, 255)}
        for i, r in enumerate(readings):
            if not r:
                continue
            what, distance = r
            radians = math.radians(car.angle + i * const.SENSORS_ANGLE_STEP)
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

    def draw(
        self,
        win: pg.Surface,
        red_car_readings: Optional[SensorReadings] = None,
        blue_car_readings: Optional[SensorReadings] = None,
    ):
        win.blit(self.background, (0, 0))

        dhw = self.diamond_image.get_width() / 2
        dhh = self.diamond_image.get_height() / 2
        for diamond_pos in self.diamond_coords:
            win.blit(self.diamond_image, (diamond_pos[0] - dhw, diamond_pos[1] - dhh))

        self.red_car.draw(win)
        self.blue_car.draw(win)

        if red_car_readings:
            self.draw_readings(win, self.red_car, red_car_readings)
        if blue_car_readings:
            self.draw_readings(win, self.blue_car, blue_car_readings)

        if self.font:
            win.blit(
                self.font.render(f"{self.red_car.score}", True, (192, 32, 32), (0, 0, 0)),
                (win.get_width() / 2 - 50, win.get_height() - 36),
            )
            win.blit(
                self.font.render(f"{self.blue_car.score}", True, (32, 32, 192), (0, 0, 0)),
                (win.get_width() / 2 + 50, win.get_height() - 36),
            )

    def process_step_outcome(self, step_outcome: StepOutcome):
        if step_outcome.collected_diamond:
            if self.diamond_sfx:
                self.diamond_sfx.play()
            self.diamond_coords.remove(step_outcome.collected_diamond)
            self.diamond_coords.add(random_pos(self.spawn_mask))

        if (
            step_outcome.crash_velocity
            and abs(step_outcome.crash_velocity) > 1.0
            and self.crash_sfx
        ):
            self.crash_sfx.play()

    def step(
        self,
        red_up: bool,
        red_down: bool,
        red_left: bool,
        red_right: bool,
        blue_up: bool,
        blue_down: bool,
        blue_left: bool,
        blue_right: bool,
    ) -> WorldStepOutcome:
        red_car_step_outcome = self.red_car.step(
            collision_mask=self.collision_mask,
            up_key=red_up,
            down_key=red_down,
            left_key=red_left,
            right_key=red_right,
            other_car=self.blue_car,
            diamond_coords=self.diamond_coords,
            diamond_mask=self.diamond_mask,
        )
        self.process_step_outcome(red_car_step_outcome)

        blue_car_step_outcome = self.blue_car.step(
            collision_mask=self.collision_mask,
            up_key=blue_up,
            down_key=blue_down,
            left_key=blue_left,
            right_key=blue_right,
            other_car=self.red_car,
            diamond_coords=self.diamond_coords,
            diamond_mask=self.diamond_mask,
        )
        self.process_step_outcome(blue_car_step_outcome)

        red_car_readings = self.red_car.sensor_readings(
            self.collision_matrix, self.blue_car, self.diamond_coords
        )
        blue_car_readings = self.blue_car.sensor_readings(
            self.collision_matrix, self.red_car, self.diamond_coords
        )
        return WorldStepOutcome(
            red_car=[red_car_readings, red_car_step_outcome],
            blue_car=[blue_car_readings, blue_car_step_outcome],
        )

    @classmethod
    def create(cls, level: str, headless: bool = False):
        red_car_img = scale_image(pg.image.load("assets/cars/red.png"), 0.75)
        blue_car_img = scale_image(pg.image.load("assets/cars/blue.png"), 0.75)
        background_img = pg.image.load(f"assets/maps/{level}/bg.png")
        collision_img = pg.image.load(f"assets/maps/{level}/map.png")
        spawn_image = pg.image.load(f"assets/maps/{level}/spawn-mask.png")
        diamond_img = pg.image.load("assets/diamond.png")
        diamond_sfx = None if headless else pg.mixer.Sound("assets/sound/money.mp3")
        crash_sfx = None if headless else pg.mixer.Sound("assets/sound/crash.mp3")
        if not headless:
            red_car_img = red_car_img.convert_alpha()
            blue_car_img = blue_car_img.convert_alpha()
            background_img = background_img.convert()
            collision_img = collision_img.convert_alpha()
            spawn_image = spawn_image.convert_alpha()
            diamond_img = diamond_img.convert_alpha()
            crash_sfx.set_volume(0.5)

        background_img.blit(collision_img, (0, 0))
        collision_mask = pg.mask.from_surface(collision_img)
        spawn_mask = pg.mask.from_surface(spawn_image)
        diamond_mask = pg.mask.from_surface(diamond_img)

        sensors = Sensors.precompute()
        return World(
            background=background_img,
            collision_mask=collision_mask,
            blue_car=Car(img=blue_car_img, x=640, y=200, angle=180, sensors=sensors),
            red_car=Car(img=red_car_img, x=640, y=600, angle=0, sensors=sensors),
            diamond_image=diamond_img,
            diamond_mask=diamond_mask,
            diamond_coords=init_diamonds(spawn_mask),
            spawn_mask=spawn_mask,
            diamond_sfx=diamond_sfx,
            crash_sfx=crash_sfx,
            font=None if headless else pg.font.Font(None, 42),
        )
