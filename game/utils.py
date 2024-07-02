import random

import pygame as pg


def scale_image(img, factor):
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return pg.transform.smoothscale(img, size)


def random_pos(spawn_mask: pg.Mask) -> tuple[int, int]:
    w, h = spawn_mask.get_size()
    while True:
        pos = random.randint(0, w - 1), random.randint(0, h - 1)
        if spawn_mask.get_at(pos):
            return pos


def init_diamonds(spawn_mask: pg.Mask):
    return {random_pos(spawn_mask) for _ in range(3)}
