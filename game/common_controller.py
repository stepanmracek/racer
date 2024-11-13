import numpy as np

from game.communication import ControlMessage, ReadingsMessage, SensorReadings


def parse_sensor(sensors: SensorReadings, letters, max_val=501.0) -> np.ndarray:
    ans = []
    for s in sensors:
        if not s:
            ans.append(1.0)
            continue
        if s[0] not in letters:
            ans.append(1.0)
            continue
        ans.append(s[1] / max_val)
    return np.array(ans, dtype=np.float32)


def create_input(readings: ReadingsMessage, input_size: int) -> np.ndarray:
    velocity = np.array([readings["velocity"]], dtype=np.float32)
    sensors = readings["sensors"]

    if input_size == 49:
        walls = parse_sensor(sensors, ("w", "e"))
        diamonds = parse_sensor(sensors, ("d"))
        return np.array([np.concatenate((velocity, walls, diamonds))])
    elif input_size == 73:
        walls = parse_sensor(sensors, ("w"))
        enemy = parse_sensor(sensors, ("e"))
        diamonds = parse_sensor(sensors, ("d"))
        return np.array([np.concatenate((velocity, walls, enemy, diamonds))])
    else:
        raise RuntimeError("Invalid input size")


POSSIBLE_KEYS: list[ControlMessage] = [
    {"u": True, "d": False, "l": True, "r": False},
    {"u": True, "d": False, "l": False, "r": False},
    {"u": True, "d": False, "l": False, "r": True},
    {"u": False, "d": False, "l": True, "r": False},
    {"u": False, "d": False, "l": False, "r": False},
    {"u": False, "d": False, "l": False, "r": True},
    {"u": False, "d": True, "l": True, "r": False},
    {"u": False, "d": True, "l": False, "r": False},
    {"u": False, "d": True, "l": False, "r": True},
]


def output_to_keys(output: np.ndarray, logits: bool = False) -> ControlMessage:
    output_shape = output.shape
    if output_shape == (1, 4):
        t = 0.0 if logits else 0.5
        return {
            "u": bool(output[0][0] > t),
            "d": bool(output[0][1] > t),
            "l": bool(output[0][2] > t),
            "r": bool(output[0][3] > t),
        }
    elif output_shape[1] == 4:
        merged_output = np.sum(np.sign(output), axis=0)
        return {
            "u": bool(merged_output[0] > 0),
            "d": bool(merged_output[1] > 0),
            "l": bool(merged_output[2] > 0),
            "r": bool(merged_output[3] > 0),
        }
    elif output_shape == (1, 9):
        return POSSIBLE_KEYS[np.argmax(output[0])]
    else:
        raise RuntimeError(f"Invalid output shape: {output_shape}")
