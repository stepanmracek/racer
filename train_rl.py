import random
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tf2onnx
from tqdm import tqdm

from game.communication import ControlMessage
from game.world import World
from game.car import Car, StepOutcome, SensorReadings
from game.common_controller import create_input, output_to_keys

FPS = 30


def create_optimizer():
    return tf.keras.optimizers.Adam(learning_rate=0.0001)


def create_model():
    input = tf.keras.Input(shape=(49,), name="input")
    hidden = tf.keras.layers.Dense(64, activation="tanh", name="hidden1")(input)
    output = tf.keras.layers.Dense(9, activation="softmax", name="output")(hidden)
    model = tf.keras.Model(name="ff_model", inputs=input, outputs=output)
    model.compile(optimizer=create_optimizer(), loss="categorical_crossentropy")
    return model


def load_model(path):
    print("Loading model from:", path)
    model = tf.keras.models.load_model(path)
    model.compile(optimizer=create_optimizer(), loss="categorical_crossentropy")
    return model


def save_onnx_model(model, path):
    spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input"),)
    tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=path)


def guided_reward(
    car: Car,
    sensor_readings: SensorReadings,
    step_outcome: StepOutcome,
    prev_step_closest_diamond: Optional[int],
) -> tuple[float, Optional[int]]:
    ans = 0.0

    # discourage when car is not moving
    if abs(car.velocity) < 0.1:
        ans -= 1.0

    closest_diamond = min((r[1] for r in sensor_readings if r and r[0] == "d"), default=None)
    # encourage exploring
    if not closest_diamond:
        ans -= -0.2
    ans += car.velocity

    if prev_step_closest_diamond is not None and closest_diamond is not None:
        # encourage approaching to dimaonds
        if closest_diamond < prev_step_closest_diamond:
            ans += 1.0
        # discourage going away from diamonds
        if prev_step_closest_diamond > closest_diamond:
            ans -= 0.2

    # extra price for collecting diamonds
    if step_outcome.collected_diamond:
        ans += 1000.0

    # discourage crashing into walls (or other car)
    if step_outcome.crash_velocity:
        ans -= 100.0

    return ans, closest_diamond


def raw_reward(
    car: Car,
    sensor_readings: SensorReadings,
    step_outcome: StepOutcome,
    prev_step_closest_diamond: Optional[int],
) -> tuple[float, Optional[int]]:
    return 1.0 if step_outcome.collected_diamond else 0.0, None


REWARD_FUNCS = {"guided": guided_reward, "raw": raw_reward}


def init_keys() -> dict[str, ControlMessage]:
    return {
        "red": {"u": False, "d": False, "l": False, "r": False},
        "blue": {"u": False, "d": False, "l": False, "r": False},
    }


@dataclass(slots=True)
class EvaluateParams:
    epoch: int
    seed: int
    model: tf.keras.Model
    reward_func: str
    discount: float


@dataclass(slots=True)
class EvaluateResult:
    states: list
    actions: list
    rewards: list
    future_discounted_reward: list


def compute_future_discounted_reward(rewards: list[float], discount: float):
    ans = []
    for i in range(len(rewards)):
        future = rewards[i:]
        ans.append(sum((r * discount**k for k, r in enumerate(future)), 0.0))

    # return ans
    return list((np.array(ans) - np.mean(ans)) / np.std(ans))


def evaluate_model(world: World, params: EvaluateParams) -> EvaluateResult:
    random.seed(params.seed)
    reward_func = REWARD_FUNCS[params.reward_func]
    world.reset()
    car_keys = init_keys()
    red_prev_diamond = None
    states = []
    actions = []
    rewards = []

    def step():
        return world.step(
            red_up=car_keys["red"]["u"],
            red_down=car_keys["red"]["d"],
            red_left=car_keys["red"]["l"],
            red_right=car_keys["red"]["r"],
            blue_up=car_keys["blue"]["u"],
            blue_down=car_keys["blue"]["d"],
            blue_left=car_keys["blue"]["l"],
            blue_right=car_keys["blue"]["r"],
        )

    # initial state
    step_outcome = step()

    for frame in tqdm(range(60 * FPS), desc=f"Epoch {params.epoch}"):
        model_input = create_input(
            {"velocity": world.red_car.velocity, "sensors": step_outcome.red_car[0]},
            params.model.inputs[0].shape[1],
        )
        model_output = params.model(model_input)
        action_probs = tfp.distributions.Categorical(probs=model_output)
        action = action_probs.sample()
        car_keys["red"] = output_to_keys(action)

        step_outcome = step()

        reward, red_prev_diamond = reward_func(
            world.red_car, *step_outcome.red_car, red_prev_diamond
        )

        states.append(model_input)
        actions.append(action)
        rewards.append(reward)

    # rewards[-1] = sum(rewards) ???

    return EvaluateResult(
        actions=actions,
        states=states,
        rewards=rewards,
        future_discounted_reward=compute_future_discounted_reward(rewards, params.discount),
    )


def train_model(model: tf.keras.Model, result: EvaluateResult):
    with tf.GradientTape() as tape:
        loss = 0
        for a, g, s in tqdm(
            zip(result.actions, result.future_discounted_reward, result.states),
            desc="Computing loss",
            total=len(result.actions),
        ):
            probs = model(s)
            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(a)
            loss += -g * tf.squeeze(log_prob)

    print("Policy gradient ascent")
    gradient = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply(gradient, model.trainable_variables)


def train():
    arg_parser = ArgumentParser(prog=f"{sys.argv[0]} train")
    arg_parser.add_argument("--initial-model")
    arg_parser.add_argument("--output-onnx-model-prefix", required=True)
    arg_parser.add_argument("--level", default="park", choices=["park", "nyan"])
    arg_parser.add_argument("--epochs", type=int, default=10)
    arg_parser.add_argument("--reward", default="guided", choices=list(REWARD_FUNCS))
    args = arg_parser.parse_args(sys.argv[2:])

    tf.random.set_seed(0)
    np.random.seed(0)

    model = load_model(args.initial_model) if args.initial_model else create_model()
    world = World.create(level=args.level, headless=True)

    for epoch in range(args.epochs):
        params = EvaluateParams(
            epoch=epoch, seed=epoch, model=model, reward_func=args.reward, discount=0.999
        )
        result = evaluate_model(world, params)
        print("Total reward:", sum(result.rewards))
        train_model(model, result)
        save_onnx_model(model, f"{args.output_onnx_model_prefix}-{epoch + 1:04}.onnx")
        print("-" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage:", sys.argv[0], "{train}")
    elif sys.argv[1] == "train":
        train()


if __name__ == "__main__":
    main()
