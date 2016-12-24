# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        newGhostPos = successorGameState.getGhostPositions()
        capsules = successorGameState.getCapsules()

        # print ("newPOS: " + str(newPos))
        # print ("dir: " + str(successorGameState.getPacmanState().getDirection()))
        # print ("newFood: ")
        # print (str(newFood))
        # print (str(newGhostStates[0]))
        # print (str(newGhostPos[0]))
        eat_capsule = 0
        if len(capsules) == 0:
            eat_capsule += 100
        elif len(capsules) > 0:
            eat_capsule -= 100
        if newPos in currentGameState.getCapsules():
            eat_capsule += 1000
             
        at_food = 0
        if newPos in currentGameState.getFood().asList():
            at_food += 10

        direction_change = 0
        if action == 'West':
            if successorGameState.getPacmanState().getDirection() == 'East':
                direction_change -= 100
            elif successorGameState.getPacmanState().getDirection() == 'North':
                direction_change -= 50
            elif successorGameState.getPacmanState().getDirection() == 'South':
                direction_change -= 50
            else:
                direction_change += 100
        elif action == 'North':
            if successorGameState.getPacmanState().getDirection() == 'South':
                direction_change -= 100
            elif successorGameState.getPacmanState().getDirection() == 'East':
                direction_change -= 50
            elif successorGameState.getPacmanState().getDirection() == 'West':
                direction_change -= 50
            else:
                direction_change += 100
        elif action == 'South':
            if successorGameState.getPacmanState().getDirection() == 'North':
                direction_change -= 100
            elif successorGameState.getPacmanState().getDirection() == 'West':
                direction_change -= 50
            elif successorGameState.getPacmanState().getDirection() == 'East':
                direction_change -= 50
            else:
                direction_change += 100
        elif action == 'East':
            if successorGameState.getPacmanState().getDirection() == 'West':
                direction_change -= 100
            elif successorGameState.getPacmanState().getDirection() == 'North':
                direction_change -= 50
            elif successorGameState.getPacmanState().getDirection() == 'South':
                direction_change -= 50
            else:
                direction_change += 100
        
        eat_food = 0
        x_coord, y_coord = newPos
        for row in newFood[x_coord-1:x_coord+2]:
            for col in row[y_coord-1:y_coord+3]:
                if col:
                    eat_food += 1

        eat_ghost = 0
        for i in range(len(newGhostPos)):
            dist = util.manhattanDistance(newPos, newGhostPos[i])
            if dist == 0:
                dist = 1e-10
            if dist == 0 and newScaredTimes[i]:
                eat_ghost += 1000
            elif dist <= newScaredTimes[i] / 2:
                eat_ghost += (1/dist)*100
            
            elif dist <= newScaredTimes[i]:
                eat_ghost += (1/dist)
            elif dist <= 4:
                eat_ghost -= (1/dist)*10

        return eat_food + at_food + eat_ghost + direction_change + eat_capsule
  
def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return (self.minimax(0, gameState))[0]

    def minimax(self, depth, gameState):
      if gameState.isWin() or gameState.isLose() or depth == self.depth * gameState.getNumAgents():
        return (None, self.evaluationFunction(gameState))
      if depth % gameState.getNumAgents() == 0:
        return self.maxf(gameState, depth)
      else:
        return self.minf(gameState, depth)

    def maxf(self, state, depth):
      actions = state.getLegalActions()
      if not actions:
        return (None, self.evaluationFunction(state))
      retval = (None, float("-inf"))
      for action in actions:
        next = state.generateSuccessor(0, action)
        result = self.minimax(depth + 1, next)
        if result[1] > retval[1]:
          retval = (action, result[1])
      return retval

    def minf(self, state, depth):
      actions = state.getLegalActions(depth % state.getNumAgents())
      if not actions:
        return self.evaluationFunction(state)
      retval = (None, float("inf"))
      for action in actions:
        next = state.generateSuccessor(depth % state.getNumAgents(), action)
        result = self.minimax(depth + 1, next)
        if result[1] < retval[1]:
          retval = (action, result[1]) 
      return retval

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

