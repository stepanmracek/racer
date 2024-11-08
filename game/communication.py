from typing import Literal, TypedDict, cast

import msgpack

from game.car import SensorReadings


class ReadingsMessage(TypedDict):
    velocity: float
    sensors: SensorReadings


class ControlMessage(TypedDict):
    u: bool
    d: bool
    l: bool
    r: bool


def parse_control_message(message: bytes) -> ControlMessage:
    return cast(ControlMessage, msgpack.loads(message))


def parse_readings_message(message: bytes) -> ReadingsMessage:
    return cast(ReadingsMessage, msgpack.loads(message))


def create_readings_message(
    car: Literal[b"red_car", b"blue_car"], velocity: float, sensors: SensorReadings
):
    message = cast(bytes, msgpack.packb({"sensors": sensors, "velocity": velocity}))
    return car + message
