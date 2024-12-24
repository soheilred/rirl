# import gym

# from stable_baselines import GAIL, SAC
# from stable_baselines.gail import ExpertDataset, generate_expert_traj

# # Generate expert trajectories (train expert)
# model = SAC('MlpPolicy', 'Pendulum-v0', verbose=1)
# generate_expert_traj(model, 'expert_pendulum', n_timesteps=100, n_episodes=10)

# # Load the expert dataset
# dataset = ExpertDataset(expert_path='expert_pendulum.npz', traj_limitation=10, verbose=1)

# model = GAIL('MlpPolicy', 'Pendulum-v0', dataset, verbose=1)
# # Note: in practice, you need to train for 1M steps to have a working policy
# model.learn(total_timesteps=1000)
# model.save("gail_pendulum")

# del model # remove to demonstrate saving and loading

# model = GAIL.load("gail_pendulum")

# env = gym.make('Pendulum-v0')
# obs = env.reset()
# while True:
#   action, _states = model.predict(obs)
#   obs, rewards, dones, info = env.step(action)
#   env.render()

"""This is a simple example demonstrating how to clone the behavior of an expert.
"""

# import gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.ppo import MlpPolicy

# from imitation.algorithms import bc
# from imitation.data import rollout
# from imitation.data.wrappers import RolloutInfoWrapper


# env = gym.make("CartPole-v1")

# def train_expert():
#     print("Training a expert.")
#     expert = PPO(
#         policy=MlpPolicy,
#         env=env,
#         seed=0,
#         batch_size=64,
#         ent_coef=0.0,
#         learning_rate=0.0003,
#         n_epochs=10,
#         n_steps=64,
#     )
#     expert.learn(100000)  # Note: change this to 100000 to trian a decent expert.
#     return expert


# def sample_expert_transitions():
#     expert = train_expert()

#     print("Sampling expert transitions.")
#     rollouts = rollout.rollout(
#         expert,
#         DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
#         rollout.make_sample_until(n_timesteps=None, n_episodes=50),
#     )
#     return rollout.flatten_trajectories(rollouts)

# # transitions = sample_expert_transitions()


# env = DummyVecEnv([lambda: RolloutInfoWrapper(env)])

# expert = train_expert()
# transitions = rollout.generate_trajectories(expert, env,
#                                             sample_until=rollout.min_episodes(50))
# trajectories = rollout.flatten_trajectories(transitions)
# # import ipdb; ipdb.set_trace()
# bc_trainer = bc.BC(
#     observation_space=env.observation_space,
#     action_space=env.action_space,
#     expert_data=trajectories,
# )


# reward, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=3, render=True)
# print(f"Reward before training: {reward}")

# print("Training a policy using Behavior Cloning")
# bc_trainer.train(n_epochs=100)

# reward, _ = evaluate_policy(bc_trainer.policy, env, n_eval_episodes=3, render=True)
# print(f"Reward after training: {reward}")


# from stable_baselines3 import PPO
# from stable_baselines3.ppo import MlpPolicy
# import gym
# import seals
# from imitation.data import rollout
# from imitation.data.wrappers import RolloutInfoWrapper
# from imitation.algorithms.adversarial import GAIL
# from imitation.rewards.reward_nets import BasicRewardNet
# from imitation.util.networks import RunningNorm
# from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
# import matplotlib.pyplot as plt
# import numpy as np



# env = gym.make("seals/CartPole-v0")
# expert = PPO(
#     policy=MlpPolicy,
#     env=env,
#     seed=0,
#     batch_size=64,
#     ent_coef=0.0,
#     learning_rate=0.0003,
#     n_epochs=10,
#     n_steps=64,
# )

# expert.learn(1000)  # Note: set to 100000 to train a proficient expert
# venv = DummyVecEnv([lambda: RolloutInfoWrapper(env)])
# rollouts = rollout.generate_trajectories(expert, venv,
#                                          sample_until=rollout.min_episodes(50))
# trajectories = rollout.flatten_trajectories(rollouts)

# # rollouts = rollout.rollout(
# #     expert,
# #     DummyVecEnv([lambda: RolloutInfoWrapper(gym.make("seals/CartPole-v0"))] * 5),
# #     rollout.make_sample_until(min_timesteps=None, min_episodes=60),
# # )

# venv = DummyVecEnv([lambda: gym.make("seals/CartPole-v0")] * 8)

# reward_net = BasicRewardNet(
#     venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
# )

# import ipdb; ipdb.set_trace()
# gail_trainer = GAIL(
#     venv=env,
#     expert_data=trajectories,
#     # gen_replay_buffer_capacity=2048,
#     expert_batch_size=1024,
#     # n_disc_updates_per_round=4,
#     gen_algo=expert,
#     reward_net=reward_net,
# )

# expert_rewards_before_training, _ = evaluate_policy(
#     expert, venv, 100, return_episode_rewards=True
# )

# gail_trainer.train(20000)  # Note: set to 300000 for better results
# expert_rewards_after_training, _ = evaluate_policy(
#     expert, venv, 100, return_episode_rewards=True
# )

# print(np.mean(expert_rewards_after_training))
# print(np.mean(expert_rewards_before_training))

# plt.hist(
#     [expert_rewards_before_training, expert_rewards_after_training],
#     label=["untrained", "trained"],
# )
# plt.legend()
# plt.show()
