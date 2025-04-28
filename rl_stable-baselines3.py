import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from gymnasium.spaces import Box
from poke_env.data import GenData

from poke_env.player import RandomPlayer, Gen8EnvSinglePlayer, SimpleHeuristicsPlayer


# We define our RL player
# It needs a state embedder and a reward computer, hence these two methods
class SimpleRLPlayer(Gen8EnvSinglePlayer):
    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GEN_8_DATA.type_chart

                )

        # We count how many pokemons have not fainted in each team
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [remaining_mon_team, remaining_mon_opponent],
            ]
        )

    def calc_reward(self, last_state, current_state) -> float:
        return self.reward_computing_helper(
            current_state, fainted_value=2, hp_value=1, victory_value=30
        )
    
    def describe_embedding(self):
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        return Box(
            np.array(low, dtype=np.float32),
            np.array(high, dtype=np.float32),
            dtype=np.float32,
        )


# class MaxDamagePlayer(RandomPlayer):
#     def choose_move(self, battle):
#         # If the player can attack, it will
#         if battle.available_moves:
#             # Finds the best move among available ones
#             best_move = max(battle.available_moves, key=lambda move: move.base_power)
#             return self.create_order(best_move)

#         # If no attack is available, a random switch will be made
#         else:
#             return self.choose_random_move(battle)

# np.random.seed(0)

# This is the function that will be used to train the a2c
def a2c_train(env, total_timesteps):
    env = Monitor(env)

    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log='./a2c_pokemon_tensorboard')
    model.learn(total_timesteps=total_timesteps)
    model.save('a2c_pokemon_model')
    
def a2c_evaluation(env: SimpleRLPlayer, model: A2C, battles):
    finished_battles = 0

    env.reset_battles()
    obs, _ = env.reset()
    
    while finished_battles < battles:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        if done:
            finished_battles += 1
            obs, _ = env.reset()
            if finished_battles >= battles:
                break

    print(
        "A2C Evaluation: %d victories out of %d episodes"
        % (env.n_won_battles, battles)
    )


NB_TRAINING_STEPS = 20_000
TEST_EPISODES = 100
GEN_8_DATA = GenData.from_gen(8)

if __name__ == "__main__":
    simpleHeuristicsPlayer = SimpleHeuristicsPlayer(battle_format='gen8randombattle')

    env = SimpleRLPlayer(opponent=simpleHeuristicsPlayer)

    # train the bot
    a2c_train(env, NB_TRAINING_STEPS)

    # model = A2C.load('a2c_pokemon_model')

    # a2c_evaluation(env, model, 100)

