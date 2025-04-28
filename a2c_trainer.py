import os
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from poke_env.player import SimpleHeuristicsPlayer
from rl_player import SimpleRLPlayer

# Creates an A2C model, trains it, and then logs the results into a tensorboard directory.
# After the model is finished training, it is saved into the file a2c_pokemon_model.zip
# Env is a gymnasium environment; should input the SimpleRLPlayer here.
# During or after training, you can view the model's progress with tensorboard
# https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html
class A2CTrainer:
    def __init__(self, opponent_class, battle_format="gen8randombattle"):
        self.battle_format = battle_format

        # Set up environment and opponent
        self.opponent = opponent_class(battle_format=battle_format)
        self.env = SimpleRLPlayer(opponent=self.opponent, battle_format=battle_format)

        self.log_dir = "./a2c_pokemon_tensorboard"
        os.makedirs(self.log_dir, exist_ok=True)  # Make log folder if it doesn't exist

        # Create the A2C model
        self.model = A2C(
            "MlpPolicy",
            self.env,
            verbose=1,
            tensorboard_log=self.log_dir,
        )

        # Set up evaluation environment for callbacks
        self.eval_env = SimpleRLPlayer(opponent=SimpleHeuristicsPlayer(battle_format=battle_format))
        self.eval_env = Monitor(self.eval_env)

        self.eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=os.path.join(self.log_dir, "best_model"),
            log_path=self.log_dir,
            eval_freq=10_000,
            deterministic=True,
            render=False,
            n_eval_episodes=20,
        )

    def train(self, total_timesteps=20_000):
        self.model.learn(total_timesteps=total_timesteps, callback=self.eval_callback)
        self.model.save("./models/a2c_pokemon_model")

    # Evaluates the model by seeing how many times it can win against the SimpleHeuristicBot
    # battles is the number of battles to evaluate with
    def evaluate(self, battles=100):
        finished_battles = 0
        self.env.reset_battles()
        obs, _ = self.env.reset()

        while finished_battles < battles:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, _, info = self.env.step(action)

            if done:
                finished_battles += 1
                obs, _ = self.env.reset()

        print(
            "A2C Evaluation: %d victories out of %d episodes"
            % (self.env.n_won_battles, battles)
        )
