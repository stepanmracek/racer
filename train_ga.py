import random
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from multiprocessing import Pool
from itertools import count

import msgpack
import numpy as np
import pygame as pg
from tqdm import tqdm

from world import World
from car import Car, StepOutcome, SensorReadings


@dataclass
class NumpyModel:
    weights: list[np.ndarray]

    def __call__(self, input) -> np.ndarray:
        hidden = np.tanh(np.matmul(input, self.weights[0]) + self.weights[1])
        output = np.matmul(hidden, self.weights[2]) + self.weights[3]
        return output

    def mutate(self, intensity: float) -> "NumpyModel":
        new_weights = []
        for w in self.weights:
            std = w.std()
            new_weights.append(
                (1.0 - intensity) * w + intensity * np.random.normal(scale=std, size=w.shape)
            )
        return NumpyModel(new_weights)

    def std(self):
        return tuple(w.std() for w in self.weights)

    @classmethod
    def crossover(
        cls, parent1: "NumpyModel", parent2: "NumpyModel", inclination: float, mutation: float
    ) -> "NumpyModel":
        new_weights = []
        for w1, w2 in zip(parent1.weights, parent2.weights):
            std = (w1.std() + w2.std()) / 2.0
            w = inclination * w1 + (1.0 - inclination) * w2
            new_weights.append(w + mutation * np.random.normal(scale=std, size=w.shape))
        return NumpyModel(new_weights)

    def save(self, path) -> None:
        packer = msgpack.Packer()
        with open(path, "wb") as f:
            for w in self.weights:
                f.write(packer.pack(w.shape))
                f.write(packer.pack(w.astype(np.float32).tobytes()))

    @classmethod
    def load(cls, path) -> "NumpyModel":
        with open(path, "rb") as f:
            unpacker = msgpack.Unpacker(f)
            weights = []
            for shape in unpacker:
                weights.append(np.frombuffer(next(unpacker), dtype=np.float32).reshape(shape))
        return NumpyModel(weights=weights)


def parse_sensor(sensors: SensorReadings, letters, max_val=501.0) -> list[float]:
    ans = []
    for s in sensors:
        if not s:
            ans.append(1.0)
            continue
        if s[0] not in letters:
            ans.append(1.0)
            continue
        ans.append(s[1] / max_val)
    return ans


def sensors_to_feature_vec(velocity: float, sensors: SensorReadings) -> list[float]:
    return [velocity] + parse_sensor(sensors, ("w", "e")) + parse_sensor(sensors, ("d",))


def compute_fitness(car: Car, step_outcome: StepOutcome) -> float:
    ans = 0.0

    if abs(car.velocity) < 0.1:
        ans -= 1.0

    if step_outcome.collected_diamond:
        ans += 1000.0

    if step_outcome.crash_velocity:
        ans -= 100.0

    return ans


def init_keys():
    return {
        "red": {"u": False, "d": False, "l": False, "r": False},
        "blue": {"u": False, "d": False, "l": False, "r": False},
    }


def process_init(level: str) -> None:
    global world
    world = World.create(level=level, headless=True)


@dataclass(slots=True)
class EvaluateParams:
    order: int
    seed: int
    model: NumpyModel


def evaluate_model(params: EvaluateParams):
    global world
    random.seed(params.seed)
    world.reset()
    car_keys = init_keys()
    fitness = 0.0
    for frame in range(60 * 30):
        step_outcome = world.step(
            red_up=car_keys["red"]["u"],
            red_down=car_keys["red"]["d"],
            red_left=car_keys["red"]["l"],
            red_right=car_keys["red"]["r"],
            blue_up=car_keys["blue"]["u"],
            blue_down=car_keys["blue"]["d"],
            blue_left=car_keys["blue"]["l"],
            blue_right=car_keys["blue"]["r"],
        )

        fitness += compute_fitness(world.red_car, step_outcome.red_car[1])
        fitness += compute_fitness(world.blue_car, step_outcome.blue_car[1])

        model_input = np.array(
            [
                sensors_to_feature_vec(world.red_car.velocity, step_outcome.red_car[0]),
                sensors_to_feature_vec(world.blue_car.velocity, step_outcome.blue_car[0]),
            ],
            dtype=np.float32,
        )
        model_output = params.model(model_input)
        car_keys = {
            "red": {
                "u": model_output[0][0] > 0,
                "d": model_output[0][1] > 0,
                "l": model_output[0][2] > 0,
                "r": model_output[0][3] > 0,
            },
            "blue": {
                "u": model_output[1][0] > 0,
                "d": model_output[1][1] > 0,
                "l": model_output[1][2] > 0,
                "r": model_output[1][3] > 0,
            },
        }
    return params.order, fitness


def train():
    arg_parser = ArgumentParser(prog="train_ga.py train")
    arg_parser.add_argument("--initial-model", required=True)
    arg_parser.add_argument("--output-model-prefix", required=True)
    arg_parser.add_argument("--level", default="park", choices=["park", "nyan"])
    arg_parser.add_argument("--initial-population-size", type=int, default=200)
    arg_parser.add_argument("--elite-count", type=int, default=20)
    arg_parser.add_argument("--parents-count", type=int, default=40)
    arg_parser.add_argument("--pairs-select-count", type=int, default=20)
    arg_parser.add_argument("--children-per-pair", type=int, default=9)
    arg_parser.add_argument("--mutation", type=float, default=0.25)
    args = arg_parser.parse_args(sys.argv[2:])

    initial_model = NumpyModel.load(args.initial_model)
    first_generation: list[NumpyModel] = []
    for mutation_intensity in np.linspace(0.0, 0.75, num=args.initial_population_size):
        first_generation.append(initial_model.mutate(mutation_intensity))

    populations = [first_generation]
    with Pool(initializer=process_init, initargs=(args.level,)) as pool:
        for generation_index in count():
            population = populations[-1]

            params = [
                EvaluateParams(order=i, seed=generation_index, model=model)
                for i, model in enumerate(population)
            ]
            results = list(
                tqdm(
                    pool.imap_unordered(evaluate_model, params),
                    total=len(population),
                    desc=f"Evaluating generation {generation_index}",
                )
            )
            results.sort(key=lambda index_fitness_pair: index_fitness_pair[0])

            # sort population by fitness (best individuals first)
            sorted_population = sorted(
                ((model, fitness) for (model, (_, fitness)) in zip(population, results)),
                reverse=True,
                key=lambda model_fitness_pair: model_fitness_pair[1],
            )

            # save best model
            sorted_population[0][0].save(f"{args.output_model_prefix}{generation_index:04}.np")

            best_fitness = sorted_population[0][1]
            worst_fitness = sorted_population[-1][1]
            avg_fitness = sum(fitness for _, fitness in sorted_population) / len(sorted_population)
            print(
                f"Best fitness: {best_fitness:.2f}; worst fitness: {worst_fitness}; average fitness: {avg_fitness}"
            )

            # create new generation according to the hyper-params
            new_generation: list[NumpyModel] = []

            # directly add elite
            elite = sorted_population[: args.elite_count]
            new_generation.extend(model for model, _ in elite)

            # select parents
            parents = sorted_population[: args.parents_count]
            parents = [parent for parent, _ in parents]

            # create pairs from those parents
            for _ in range(args.pairs_select_count):
                p1, p2 = np.random.choice(range(args.parents_count), 2, replace=False)

                # generate children
                children = [
                    NumpyModel.crossover(
                        parents[p1], parents[p2], inclination, mutation=args.mutation
                    )
                    for inclination in np.linspace(0.1, 0.9, num=args.children_per_pair)
                ]
                new_generation.extend(children)

            populations.append(new_generation)
            diversity = np.array([i.std() for i in new_generation])
            print("Average diversity per weight matrix in new population:", diversity.mean(0))


def test():
    arg_parser = ArgumentParser(prog="train_ga.py test")
    arg_parser.add_argument("--model", required=True)
    arg_parser.add_argument("--level", default="park", choices=["park", "nyan"])
    arg_parser.add_argument("--timelimit", default=60, type=int)
    arg_parser.add_argument("--scorelimit", default=10, type=int)
    args = arg_parser.parse_args(sys.argv[2:])

    pg.init()
    win = pg.display.set_mode((1280, 768))
    pg.display.set_caption("Racer")
    clock = pg.Clock()

    world = World.create(level=args.level)
    model = NumpyModel.load(args.model)
    car_keys = init_keys()

    frame_limit = args.timelimit * 30
    frame = 0
    fitness = 0.0
    while frame < frame_limit:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return

        just_pressed_keys = pg.key.get_just_pressed()
        if just_pressed_keys[pg.K_r]:
            world.reset()
            fitness = 0.0
            frame = 0

        step_outcome = world.step(
            red_up=car_keys["red"]["u"],
            red_down=car_keys["red"]["d"],
            red_left=car_keys["red"]["l"],
            red_right=car_keys["red"]["r"],
            blue_up=car_keys["blue"]["u"],
            blue_down=car_keys["blue"]["d"],
            blue_left=car_keys["blue"]["l"],
            blue_right=car_keys["blue"]["r"],
        )

        fitness += compute_fitness(world.red_car, step_outcome.red_car[1])
        fitness += compute_fitness(world.blue_car, step_outcome.blue_car[1])

        if world.red_car.score >= args.scorelimit or world.blue_car.score >= args.scorelimit:
            break

        world.draw(win, step_outcome.red_car[0], step_outcome.blue_car[0])
        time_left = frame / frame_limit * 1280
        pg.draw.rect(win, (255, 255, 0), (0, 760, time_left, 768))
        pg.display.update()
        frame += 1

        clock.tick(30)

        model_input = np.array(
            [
                sensors_to_feature_vec(world.red_car.velocity, step_outcome.red_car[0]),
                sensors_to_feature_vec(world.blue_car.velocity, step_outcome.blue_car[0]),
            ],
            dtype=np.float32,
        )
        model_output = model(model_input)
        car_keys = {
            "red": {
                "u": model_output[0][0] > 0,
                "d": model_output[0][1] > 0,
                "l": model_output[0][2] > 0,
                "r": model_output[0][3] > 0,
            },
            "blue": {
                "u": model_output[1][0] > 0,
                "d": model_output[1][1] > 0,
                "l": model_output[1][2] > 0,
                "r": model_output[1][3] > 0,
            },
        }
    pg.quit()


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "test":
        test()
