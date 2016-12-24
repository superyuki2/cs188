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
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
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
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
          temp = util.Counter()
          for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
              temp[state] = 0
            else:
              curmax = float('-inf')
              for action in self.mdp.getPossibleActions(state):
                update = 0
                for next, chance in self.mdp.getTransitionStatesAndProbs(state, action):
                  update = update + chance * (self.mdp.getReward(state, action, next) + (self.discount * self.values[next]))
                if update > curmax:
                  curmax = update
                temp[state] = curmax
          self.values = temp


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
        ret = 0
        for next, chance in self.mdp.getTransitionStatesAndProbs(state, action):
          ret = ret + chance * (self.mdp.getReward(state, action, next) + (self.discount * self.values[next]))
        return ret

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestval = float('-inf')
        bestpol = None
        for policy in self.mdp.getPossibleActions(state):
          if bestval < self.computeQValueFromValues(state, policy):
            bestval = self.computeQValueFromValues(state, policy)
            bestpol = policy
        return bestpol

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        i = 0
        while i < self.iterations:
            state = self.mdp.getStates()[i % len(self.mdp.getStates())]
            if self.mdp.isTerminal(state):
                i += 1
                continue
            all_vals = []
            for action in self.mdp.getPossibleActions(state):
                update = self.computeQValueFromValues(state, action)
                all_vals.append(update)
            self.values[state] = max(all_vals)
            i += 1

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = {}
        for state in self.mdp.getStates():
          predecessors[state] = set()
        for state in self.mdp.getStates():
          for action in self.mdp.getPossibleActions(state):
            for next, chance in self.mdp.getTransitionStatesAndProbs(state, action):
              if chance > 0:
                predecessors[next].add(state)
        queue = util.PriorityQueue()
        for state in self.mdp.getStates():
          if not self.mdp.isTerminal(state):
            all_vals = []
            for action in self.mdp.getPossibleActions(state):
              update = self.computeQValueFromValues(state, action)
              all_vals.append(update)
            diff = abs(self.getValue(state) - max(all_vals))
            queue.push(state, -diff)
        for x in range(self.iterations):
          if queue.isEmpty():
            break
          curr_state = queue.pop()
          if not self.mdp.isTerminal(curr_state):
            all_vals = [self.getQValue(curr_state, action) for action in self.mdp.getPossibleActions(curr_state)]
            self.values[curr_state] = max(all_vals)
            for predecessor in predecessors[curr_state]:
              p_vals = [self.getQValue(predecessor, action) for action in self.mdp.getPossibleActions(predecessor)]
              diff = abs(self.getValue(predecessor) - max(p_vals))
              if diff > self.theta:
                queue.update(predecessor, -diff)







