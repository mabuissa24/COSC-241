# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # Create local variables:
        # U, U' (vectors of utilties for all states) - use a dict associated to the states?
        # delta - max change in utility for any state
        newUtilities = util.Counter()
        # delta = 0
        # I'VE GOTTEN IT TO THE POINT WHERE EVERYTHING IS JUST A LITTLE BIT TOO FAR AWAY FROM 0 (abs val too high)
        # Loop over the following while delta >= epsilon (1 - lambda) / lambda:
        index = 0
        # print 'is index less than iterations?', index < iterations, ' with index ', index, ' and iterations ', iterations
        while (index < iterations):
            # print 'do we enter this while loop?'
            index += 1
            # Set U to U' and delta to 0
              # is this in the right place?
            # delta = 0

            # Go through the states in S and do the following:
            # print 'at index ', index, ' we set the states as following:'
            for s in mdp.getStates():
                # U'[S] = reward at S + lambda (max over actions of sum(prob going to new state)(utility of new state))
                qValues = []
                for a in mdp.getPossibleActions(s):
                    # for next_state, prob in mdp.getTransitionStatesAndProbs(s, a):
                    qValue = 0
                    for stateValue in mdp.getTransitionStatesAndProbs(s, a):
                        # qValue += mdp.getReward(s, a, stateValue[0])
                        qValue += (mdp.getReward(s, a, stateValue[0]) + self.discount * self.values[stateValue[0]]) * stateValue[1]
                    qValues.append(qValue)
                if len(qValues) == 0:
                    newUtilities[s] = 0
                else:
                    newUtilities[s] = max(qValues)
                # print 'the old utility of ', self.values[s], ' becomes ', newUtilities[s]
                # print 'newUtilities at s should be', max(rValues), ' it is ', newUtilities[s]
                # If the utility at S changed by more than delta, set delta to that change
                # change = abs(newUtilities[s] - self.values[s])
                # if change > delta:
                #    delta = change
            self.values = newUtilities.copy()

        # Return U

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        QValue = 0
        for stateValue in self.mdp.getTransitionStatesAndProbs(state, action):
            # print 'QValue is ', QValue
            # mdp.getReward(state, action, stateValue[0])
            QValue += (self.discount * self.values[stateValue[0]]) * stateValue[1] + self.mdp.getReward(state, action, stateValue[0])  # Consider reward for each action
            # print 'reward is ', self.mdp.getReward(state, action, stateValue[0])
            # print 'product of ', self.values[stateValue[0]], ' and ', stateValue[1]

        return QValue

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        maxValue = float('-inf')
        bestAction = None
        for action in self.mdp.getPossibleActions(state):
            # to calculate value, look at the possible results and weight the utilities of those states accordingly
            value = self.computeQValueFromValues(state, action)
            if value > maxValue:
                maxValue = value
                bestAction = action

        return bestAction

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
