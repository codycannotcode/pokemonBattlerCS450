import asyncio

from poke_env import AccountConfiguration, ShowdownServerConfiguration
from poke_env.player import RandomPlayer, Player, SimpleHeuristicsPlayer

# class MaxDamagePlayer(Player):
#     def choose_move(self, battle):
#         # Chooses a move with the highest base power when possible
#         if battle.available_moves:
#             # Iterating over available moves to find the one with the highest base power
#             best_move = max(battle.available_moves, key=lambda move: move.base_power)

#             if battle.can_tera:
#                 return self.create_order(best_move, terastallize=True)
#             # Creating an order for the selected move
#             return self.create_order(best_move)
#         else:
#             # If no attacking move is available, perform a random switch
#             # This involves choosing a random move, which could be a switch or another available action
#             return self.choose_random_move(battle)

async def main():
    random_player = RandomPlayer()
    random_player_2 = RandomPlayer()

    # max_damage_player = MaxDamagePlayer()

    await random_player.battle_against(random_player_2, n_battles=50)

    print(
        f'Player {random_player.username} won {random_player.n_won_battles} out of {random_player.n_finished_battles} played'
    )

if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())