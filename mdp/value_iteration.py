import numpy as np


def value_iteration(env, gamma=1.0,no_of_iterations=10000):
    value_table = np.zeros(env.observation_space.n)
    threshold = 1e-20

    for i in range(no_of_iterations):
        updated_value_table = np.copy(value_table)
        for state in range(env.observation_space.n):
            Q_value = []
            for action in range(env.action_space.n):
                next_states_results = []
                for next_sr in env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_sr
                    next_states_reward = \
                        trans_prob*(reward_prob+gamma*updated_value_table[next_state])
                    next_states_results.append(next_states_reward)
                Q_value.append(np.sum(next_states_results))
            value_table[state] = max(Q_value)

        if np.sum(np.fabs(updated_value_table-value_table)) <= threshold:
            break
    return value_table


def extract_policy(env, value_table, gamma=1):
    policy = np.zeros(env.observation_space.n)

    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += \
                    trans_prob*(reward_prob+gamma*value_table[next_state])

        policy[state] = np.argmax(Q_table)



