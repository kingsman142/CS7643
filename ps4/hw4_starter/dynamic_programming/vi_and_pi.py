### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time
from lake_envs import *

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

    P: nested dictionary
        From gym.core.Environment
        For each pair of states in [0, nS-1] and actions in [0, nA-1],
        P[state][action] is a tuple of the form
        (probability, nextstate, reward, terminal) where
            - probability: float
                the probability of transitioning from "state" to "nextstate" with "action"
            - nextstate: int
                denotes the state we transition to (in range [0, nS - 1])
            - reward: int
                either 0 or 1, the reward for transitioning from "state" to
                "nextstate" with "action"
            - terminal: bool
              True when "nextstate" is a terminal state (hole or goal), False otherwise
    nS: int
        number of states in the environment
    nA: int
        number of actions in the environment
    gamma: float
        Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """

    value_function = np.zeros(nS)
    #####################################################################
    # YOUR IMPLEMENTATION HERE
    #####################################################################
    iter = 0
    while True:
        prev_value_function = value_function # store the previous value function values so we can check for convergence later
        for s in range(nS): # get a value for each state
            action_vals = []

            # for this given state, calculate the value of each action
            state_val = 0.0
            for out in P[s][policy[s]]:
                probability, nextstate, reward, terminal = out # if we perform this action, where do we go next?
                curr_val = reward
                if not terminal: # if this is not a terminal node, add a penalty for future rewards
                    curr_val += gamma * prev_value_function[nextstate]
                curr_val *= probability # calculate the current value for this s' from this action (multiple s' states in the stochastic case, but only one in the deterministic case)
                state_val += curr_val

            # update the 'future' value reward for this state, as well as the 'policy' for this state (which action to take)
            value_function[s] = state_val # for this given state, take the action with the highest reward

        # check if the values have converged
        # NOTE: Since multiple actions can have the same reward, and we choose a random action to favor exploration, we need to add a threshold
        #       on the number of iterations before breaking this while loop because on some iterations, due to randomness, the value iteration
        #       might not have changed early on.
        if np.max(np.absolute(np.subtract(value_function, prev_value_function))) < tol and iter >= 100:
            break
        iter += 1
    #####################################################################
    #                             END OF YOUR CODE                      #
    #####################################################################
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    policy: np.array
        The previous policy.

    Returns
    -------
    new_policy: np.ndarray[nS]
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """

    new_policy = np.zeros(nS, dtype='int')

    #####################################################################
    # YOUR IMPLEMENTATION HERE
    #####################################################################
    for s in range(nS):
        action_vals = [] # keep track of value of each action for this state
        for a in range(nA):
            action_val = 0.0
            for out in P[s][a]:
                probability, nextstate, reward, terminal = out # if we perform this action, where do we go next?
                curr_val = reward
                if not terminal: # if this is not a terminal node, add a penalty for future rewards
                    curr_val += gamma * value_from_policy[nextstate]
                curr_val *= probability # calculate the current value for this s' from this action (multiple s' states in the stochastic case, but only one in the deterministic case)
                action_val += curr_val # keep track of this action's value
            action_vals.append(action_val)

        # for this state, find the best action
        max_indices = np.argwhere(action_vals == np.max(action_vals)).flatten().tolist() # find ALL actions with the highest reward, not just the first one
        best_action_index = np.random.choice(max_indices) # if there are multiple actions with max reward, randomly choose one
        new_policy[s] = best_action_index # set the best action for this state
    #####################################################################
    #                             END OF YOUR CODE                      #
    #####################################################################
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    #####################################################################
    # YOUR IMPLEMENTATION HERE
    #####################################################################
    while True:
        # policy evaluation
        value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)

        # policy improvement
        old_policy = policy # keep track of this to check for convergence
        policy = policy_improvement(P, nS, nA, value_function, policy, gamma)

        if np.max(np.absolute(np.subtract(policy, old_policy))) < tol: # the policy hasn't changed at all, so we terminate
            break
    #####################################################################
    #                             END OF YOUR CODE                      #
    #####################################################################
    return value_function, policy

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)

    #####################################################################
    # YOUR IMPLEMENTATION HERE
    #####################################################################
    iter = 0
    while True:
        prev_value_function = value_function # store the previous value function values so we can check for convergence later
        for s in range(nS): # get a value for each state
            action_vals = []

            # for this given state, calculate the value of each action
            for a in range(nA): # get a value for each action so we can take the best action later on
                running_val = 0.0
                for out in P[s][a]:
                    probability, nextstate, reward, terminal = out # if we perform this action, where do we go next?
                    curr_val = reward
                    if not terminal: # if this is not a terminal node, add a penalty for future rewards
                        curr_val += gamma * prev_value_function[nextstate]
                    curr_val *= probability # calculate the current value for this s' from this action (multiple s' states in the stochastic case, but only one in the deterministic case)
                    running_val += curr_val # keep track of this action's value
                action_vals.append(running_val)

            # for this state, find the best action
            max_indices = np.argwhere(action_vals == np.max(action_vals)).flatten().tolist() # find ALL actions with the highest reward, not just the first one
            best_action_index = np.random.choice(max_indices) # if there are multiple actions with max reward, randomly choose one
            best_action_val = action_vals[best_action_index] # actual reward value of the best action

            # update the 'future' value reward for this state, as well as the 'policy' for this state (which action to take)
            value_function[s] = best_action_val # for this given state, take the action with the highest reward
            policy[s] = best_action_index # set the best action for this state

        # check if the values have converged
        # NOTE: Since multiple actions can have the same reward, and we choose a random action to favor exploration, we need to add a threshold
        #       on the number of iterations before breaking this while loop because on some iterations, due to randomness, the value iteration
        #       might not have changed early on.
        if np.max(np.absolute(np.subtract(value_function, prev_value_function))) < tol and iter >= 100:
            break
        iter += 1
    #####################################################################
    #                             END OF YOUR CODE                      #
    #####################################################################
    return value_function, policy

def render_single(env, policy, max_steps=100, show_rendering=True):
    """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
    """

    episode_reward = 0
    ob = env.reset()
    for t in range(max_steps):
        if show_rendering:
            env.render()
        time.sleep(0.25)
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    if show_rendering:
        env.render()
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: %f" % episode_reward)

def evaluate(env, policy, max_steps=100, max_episodes=32):
    """
    This function does not need to be modified,
    evaluates your policy over multiple episodes.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
    """

    episode_rewards = []
    dones = []
    for _ in range(max_episodes):
        episode_reward = 0
        ob = env.reset()
        for t in range(max_steps):
            a = policy[ob]
            ob, rew, done, _ = env.step(a)
            episode_reward += rew
            if done:
                break

        episode_rewards.append(episode_reward)
        dones.append(done)

    episode_rewards = np.array(episode_rewards).mean()
    success = np.array(dones).mean()

    print(f"> Average reward over {max_episodes} episodes:\t\t\t {episode_rewards}")
    print(f"> Percentage of episodes goal reached:\t\t\t {success * 100:.0f}%")
