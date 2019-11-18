#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gridworld
import numpy as np
import random
import copy

class DynamicProgramming:
    def __init__(self, environmentName):
        """
        Class for performing value interation in the given environment
        Parameters
        ----------
        environmentName : string
            Name of gym environment to utilize.
        Returns
        -------
        None.
        """
        self.env = gridworld.GridworldEnv()
        self.theta = 0.0001
        self.discount_factor = 0.9

    def One_Step_LookAhead(self, state, V):
        """
        Function for calculating the value of a given state
        Parameters
        ----------
        state : int
            Current state.
        V : Value array
            list.
        Returns
        -------
        A : List
            Action Value Array.
        """
        A = np.zeros(self.env.nA)
        for a in range(self.env.nA):
            for prob, next_state, reward, done in self.env.P[state][a]:
                A[a] += prob * (reward + self.discount_factor*V[next_state])
        return A


    def run(self):
        V = np.zeros(self.env.nS)
        while True:
            delta = 0
            for s in range(self.env.nS):
                A = self.One_Step_LookAhead(s, V)
                best_action_value = np.max(A)
                delta = max(delta, np.abs(best_action_value - V[s]))
                V[s] = best_action_value
            if delta < self.theta:
                break
        # Create a deterministic policy using the optimal value function
        policy = np.zeros([self.env.nS, self.env.nA])
        for s in range(self.env.nS):
            # One step lookahead to find the best action for this state
            A = self.One_Step_LookAhead(s, V)
            best_action = np.argmax(A)
            # Always take the best action
            policy[s, best_action] = 1.0
        return policy, V

if __name__ == '__main__':
    dp = DynamicProgramming('FrozenLake-v0')
    pi, v = dp.run()

    print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
    print(np.reshape(np.argmax(pi, axis=1), dp.env.shape))
    print("")

    print("Reshaped Grid Value Function:")
    print(v.reshape(dp.env.shape))
    print("")
