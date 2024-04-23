import numpy as np
from RL.value_iteration import extract_policy


def compute_value_function(env, policy, gamma=1.0):
    value_table = np.zeros(env.nS)
    threshold = 1e-10
    while True:
        updated_value_table = np.copy(value_table)
        for state in range(env.nS):
            action = policy[state]

            value_table[state] = \
                sum([trans_prob * (reward_prob+gamma*updated_value_table[next_state])
                    for trans_prob,reward_prob,next_state,_ in env.P[state][action]])

        if np.sum((np.fabs(updated_value_table - value_table))) <= threshold:
            break
    return value_table


def policy_iteration(env, gamma=1.0):
    old_policy = np.zeros(env.observation_space.n)
    no_of_iteration = 10000
    for i in range(no_of_iteration):
        new_value_table = compute_value_function(env, old_policy, gamma)
        new_policy = extract_policy(env, new_value_table, gamma)
        if np.all(old_policy==new_policy):
            break
        old_policy = new_policy
    return new_policy
