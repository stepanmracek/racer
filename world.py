import math
from dataclasses import dataclass

import pygame as pg

from car import Car, StepOutcome
from utils import init_diamonds, random_pos


@dataclass
class World:
    background: pg.Surface
    blue_car: Car
    red_car: Car
    diamond_image: pg.Surface
    diamond_sfx: pg.mixer.Sound
    diamond_coords: set[tuple[int, int]]
    crash_sfx: pg.mixer.Sound
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

        dhw = self.diamond_image.get_width() / 2
        dhh = self.diamond_image.get_height() / 2
        for diamond_pos in self.diamond_coords:
            win.blit(self.diamond_image, (diamond_pos[0] - dhw, diamond_pos[1] - dhh))

        self.red_car.draw(win)
        self.blue_car.draw(win)

    def process_step_outcome(self, step_outcome: StepOutcome):
        if step_outcome.collected_diamond:
            self.diamond_sfx.play()
            self.diamond_coords.remove(step_outcome.collected_diamond)
            self.diamond_coords.add(random_pos(self.spawn_mask))

        if step_outcome.crash_velocity and abs(step_outcome.crash_velocity) > 1.0:
            self.crash_sfx.play()
