import numpy as np
from collections import defaultdict


def sample_policy(observation):
    score, dealer_score, usable_ace = observation
    return 0 if score >=20 else 1


def generate_episode(env):
    states, actions, rewards = [], [], []
    observation = env.reset()
    while True:
        states.append(observation)
        action = sample_policy(observation)
        actions.append(action)
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
            break
    return states, actions, rewards


def first_visit_mc_prediction(env, n_episodes, gamma=1.0):
    value_table = defaultdict(float)
    N = defaultdict(float)
    for _ in range(n_episodes):
        states, actions, rewards = generate_episode(env)
        returns = 0
        for t in range(len(states)-1,-1,-1):
            R, S = rewards[t], states[t]
            returns = gamma*returns+R
            if S not in states[:t]:
                N[S] += 1
                value_table[S] += (returns-value_table[S])/N[S]


