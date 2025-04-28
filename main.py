import argparse
from poke_env.player import SimpleHeuristicsPlayer

# Number of training steps and evaluation episodes
NB_TRAINING_STEPS = 20_000
TEST_EPISODES = 100

# Entrypoint: Train and evaluate the bot
if __name__ == "__main__":
    # Parse command-line arguments to choose the training algorithm
    parser = argparse.ArgumentParser(description="Run Pokémon AI training")
    parser.add_argument(
        "--algorithm",
        choices=["a2c", "ppo"],
        required=True,
        help="Training algoirithm selection: a2c, ppo, ...(future implementations)",
    )
    args = parser.parse_args()

    # Initialize the trainer based on the chosen algorithm
    if args.algorithm == "a2c":
        from a2c_trainer import A2CTrainer
        trainer = A2CTrainer(SimpleHeuristicsPlayer)
    elif args.algorithm == "ppo":
        from ppo_trainer import PPOTrainer
        trainer = PPOTrainer(SimpleHeuristicsPlayer)
    else:
        raise ValueError(f"Unknown training algorithm: {args.algorithm}")

    # Train the bot
    trainer.train(total_timesteps=NB_TRAINING_STEPS)

    # Evaluate the bot
    trainer.evaluate(battles=TEST_EPISODES)