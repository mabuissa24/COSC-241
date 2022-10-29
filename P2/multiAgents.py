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

        width = newFood.packBits()[0]
        height = newFood.packBits()[1]
       #  print "printing"
        # print "pacman's pos", newPos
        # print(newFood)
        # for a in successorGameState.getGhostPositions():
            # print a
        # print newScaredTimes
        # print(successorGameState.getScore())
        walls = currentGameState.getWalls()
        if walls[newPos[0]][newPos[1]]:
            # print "wall alert"
            solution = -10000
            # print "solution is:", solution
            return solution

        if newPos == currentGameState.getPacmanPosition():
            solution = -111111111111
            return solution

        numGhosts = 0
        ghostDist = width + height
        for pos in successorGameState.getGhostPositions():
            distance = util.manhattanDistance(pos, newPos)
            if distance < ghostDist:
                ghostDist = distance
            #ghostDist += util.manhattanDistance(pos, newPos)
            numGhosts += 1

        foodDist = closestFood(newPos, newFood, width, height)  # We should maybe test this further
        # print "foodDist is", foodDist
        foodTotal = totalFood(newFood, width, height)
        # print "foodTotal is", foodTotal
        # print "ghostDist is", ghostDist
        # print "time scared is", newScaredTimes

        # if food distance = 1 and ghost is far, eat the damn food

        # if ghostDist > (width + height) / (4 * numGhosts) and foodDist <= 1:
        #     solution = 11111111
        #     print "solution", solution

        # ADD GHOST POWER PELLET CASE
        # If you can eat a power pellet, DO
        minScaredTime = min(newScaredTimes)

        # When there are any ghosts who have scared time left, pursue them
        # Until there's only one scared amount left, then go back to running away
        if (minScaredTime > 1):
            solution = 3 * (width + height) - 15 * ghostDist - foodDist - foodTotal
        # elif ghostDist > (width + height) / (5 * numGhosts): # Will generally be lower than the else case; makes exiting this range of the ghost look bad
        elif ghostDist > 3:
            # print "ghost is far enough away"
            solution = 3 * (width + height) + ghostDist - 2 * foodDist - (width + height) * foodTotal
        else:
            # print "ghost is very close"
            solution = (width + height) * ghostDist - foodDist - (width + height) * foodTotal

        # solution = util.manhattanDistance(successorGameState.getGhostPositions(),
        # successorGameState.getPacmanPosition())
        # print "distances", ghostDist


        # print "solution is", solution
        return solution
        # return successorGameState.getScore()

def closestFood(position, food, w, h):
    positionOfClosestFood = (-1, -1)
    minDist = w + h
    runningDist = 0
    hasFood = False
    for i in range(w):
        for j in range(h):
            if food[i][j]:
                hasFood = True
                dist = util.manhattanDistance(position, (i, j))
                if dist < minDist:
                    minDist = dist
                    positionOfClosestFood = (i, j)
    if not hasFood:
        return 0
    return util.manhattanDistance(position, positionOfClosestFood)


    return runningDist

def totalFood(food, w, h):

    runningTotal = 0
    for i in range(w):
        for j in range(h):
            if food[i][j]:
                runningTotal += 1

    return runningTotal

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
        """

        utilities = []
        for action in gameState.getLegalActions(0):
            newState = gameState.generateSuccessor(0, action)
            utilities.append(self.minimax(newState, 1, 1))

        max = utilities[0]
        actionIndex = 0

        for i in range(len(utilities)):
            if utilities[i] > max:
                max = utilities[i]
                actionIndex = i

        return gameState.getLegalActions(0)[actionIndex]

    def minimax(self, gameState, agentIndex, currDepth):
        newAgentIndex = (agentIndex + 1) % gameState.getNumAgents()

        if agentIndex == 0:
            currDepth += 1

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if currDepth > self.depth:
            return self.evaluationFunction(gameState)

        values = []
        for action in gameState.getLegalActions(agentIndex):
            state = gameState.generateSuccessor(agentIndex, action)
            values.append(self.minimax(state, newAgentIndex, currDepth))

        if agentIndex == 0:
            return max(values)
        elif agentIndex > 0:
            return min(values)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        a = float('-inf')
        b = float('inf')
        v = float('-inf')
        utilities = []
        for action in gameState.getLegalActions(0):
            newState = gameState.generateSuccessor(0, action)
            currValue = self.minValue(newState, a, b, 1, 1) # 10
            utilities.append(currValue)
            v = max(v, currValue)
            a = max(a, v)
            # print "a is", a

        maxVal = utilities[0]
        actionIndex = 0

        for i in range(len(utilities)):
            if utilities[i] > maxVal:
                maxVal = utilities[i]
                actionIndex = i

        return gameState.getLegalActions(0)[actionIndex]

    def maxValue(self, gameState, a, b, agentIndex, currDepth):
        currDepth += 1
        newAgentIndex = (agentIndex + 1) % gameState.getNumAgents()

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if currDepth > self.depth:
            return self.evaluationFunction(gameState)

        v = float('-inf')

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            v = max(v, self.minValue(successor, a, b, newAgentIndex, currDepth))
            if v > b:
                # print "v is", v
                # print "v g than or quela to beta"
                # print "beta is", b
                return v

            a = max(a, v)
            # print "a is", a

        return v

    def minValue(self, gameState, a, b, agentIndex, currDepth):
        newAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
        # print 'new'

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if currDepth > self.depth:
            return self.evaluationFunction(gameState)

        v = float('inf')

        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            if newAgentIndex == 0:
                v = min(v, self.maxValue(successor, a, b, newAgentIndex, currDepth))
            else:
                v = min(v, self.minValue(successor, a, b, newAgentIndex, currDepth))
            if v < a:
                return v
            b = min(b, v)

        return v

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
        utilities = []
        for action in gameState.getLegalActions(0):
            newState = gameState.generateSuccessor(0, action)
            utilities.append(self.expectimax(newState, 1, 1))

        max = utilities[0]
        actionIndex = 0

        for i in range(len(utilities)):
            if utilities[i] > max:
                max = utilities[i]
                actionIndex = i

        return gameState.getLegalActions(0)[actionIndex]

        util.raiseNotDefined()


    def expectimax(self, gameState, agentIndex, currDepth):
        newAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
        if agentIndex == 0:
            currDepth += 1

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if currDepth > self.depth:
            return self.evaluationFunction(gameState)

        values = []
        for action in gameState.getLegalActions(agentIndex):
            successor = gameState.generateSuccessor(agentIndex, action)
            values.append(self.expectimax(successor, newAgentIndex, currDepth))
        if agentIndex == 0:
            return max(values)
        elif agentIndex > 0:
            average = sum(values) / len(values)
            return average


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    width = newFood.packBits()[0]
    height = newFood.packBits()[1]
    #  print "printing"
    # print "pacman's pos", newPos
    # print(newFood)
    # for a in successorGameState.getGhostPositions():
    # print a
    # print newScaredTimes
    # print(successorGameState.getScore())
    walls = currentGameState.getWalls()
    if walls[newPos[0]][newPos[1]]:
        # print "wall alert"
        solution = -10000
        # print "solution is:", solution
        return solution

    if newPos == currentGameState.getPacmanPosition(): # This was meant to stop pacman from sitting in a corner and not moving
        solution = -111111111111
        return solution

    numGhosts = 0
    ghostDist = width + height
    for pos in currentGameState.getGhostPositions():
        distance = util.manhattanDistance(pos, newPos)
        if distance < ghostDist:
            ghostDist = distance
        # ghostDist += util.manhattanDistance(pos, newPos)
        numGhosts += 1

    foodDist = closestFood(newPos, newFood, width, height)  # We should maybe test this further
    # print "foodDist is", foodDist
    foodTotal = totalFood(newFood, width, height)
    # print "foodTotal is", foodTotal
    # print "ghostDist is", ghostDist
    # print "time scared is", newScaredTimes

    # if food distance = 1 and ghost is far, eat the damn food

    # if ghostDist > (width + height) / (4 * numGhosts) and foodDist <= 1:
    #     solution = 11111111
    #     print "solution", solution

    # ADD GHOST POWER PELLET CASE
    # If you can eat a power pellet, DO
    minScaredTime = min(newScaredTimes)

    # When there are any ghosts who have scared time left, pursue them
    # Until there's only one scared amount left, then go back to running away
    if (minScaredTime > 1):
        solution = 3 * (width + height) - 15 * ghostDist - foodDist - foodTotal
    # elif ghostDist > (width + height) / (5 * numGhosts): # Will generally be lower than the else case; makes exiting this range of the ghost look bad
    elif ghostDist > 3:
        # print "ghost is far enough away"
        solution = 3 * (width + height) + ghostDist - 2 * foodDist - (width + height) * foodTotal
    else:
        # print "ghost is very close"
        solution = (width + height) * ghostDist - foodDist - (width + height) * foodTotal

    # solution = util.manhattanDistance(successorGameState.getGhostPositions(),
    # successorGameState.getPacmanPosition())
    # print "distances", ghostDist

    # print "solution is", solution
    return solution

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

