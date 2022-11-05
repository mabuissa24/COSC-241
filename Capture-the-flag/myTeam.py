# myTeam.py
# ---------
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

# Final Capture the Flag project completed in Fall 2021 by Claire Jensen and Maryam Abuissa for
# Professor Scott Alfeld's Artificial Intelligence course


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='SmartyAgent', second='SmartyAgent'):
    """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

particles = [[], []]


class SmartyAgent(CaptureAgent):
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.carrying = 0
        self.foodToEat = self.countFood(gameState)
        self.initialFoodCount = self.countFood(gameState)

        # Create n particles in their start state - can find this as part of the map
        if gameState.isOnRedTeam(self.index):
            self.enemies = gameState.getBlueTeamIndices()
        else:
            self.enemies = gameState.getRedTeamIndices()

        self.numParticles = 600

        # Distribute particles at each enemy's start state
        for i in range(len(self.enemies)):
            for index in range(self.numParticles):
                particles[i].append(gameState.getInitialAgentPosition(self.enemies[i]))

        # Calculate middle of the board so we can determine which half
        # of the board something is on
        positions = [pos for pos in gameState.getWalls().asList(False) if pos[1] > 1]
        x1, y1 = max(positions)
        x0, y0 = min(positions)
        self.middle = x0 + float(x1 - x0) / 2

        # Fixes error dividing the board when on blue team
        if not gameState.isOnRedTeam(self.index):
            self.middle += 1

    def initializeParticles(self, gameState):
        # Determine which agents are enemies
        if gameState.isOnRedTeam(self.index):
            self.enemies = gameState.getBlueTeamIndices()
        else:
            self.enemies = gameState.getRedTeamIndices()

        # Create n particles randomly distributed
        positions = [pos for pos in gameState.getWalls().asList(False) if pos[1] > 1]
        for i in range(len(self.enemies)):
            for index in range(self.numParticles):
                pos = positions[index % len(positions)]
                particles[i].append(pos)

    def chooseAction(self, gameState):
        dis1 = self.getDistribution(0)
        dis2 = self.getDistribution(1)
        self.displayDistributionsOverPositions([dis1, dis2, util.Counter(), util.Counter()])

        # Update for each agent before updating using observe
        self.particleObserve(gameState)
        if self.index == 0 or self.index == 3:
            self.particleElapseTime(gameState, 1)
        if self.index == 1 or self.index == 2:
            self.particleElapseTime(gameState, 0)

        # Append particles to each enemy's position
        enemyPositions = self.getEnemyPositions(gameState)
        for index in range(len(self.enemies)):
            if enemyPositions[index] is not None:
                particles[index] = []
                for i in range(self.numParticles):
                    particles[index].append(enemyPositions[index])

        return self.assessSituation(gameState)

    def onOurSide(self, gameState, position):
        # Method for determining if a position is on our side,
        # depending on which team we are
        if gameState.isOnRedTeam(self.index):
            return position[0] < self.middle - 1
        else:
            return position[0] > self.middle + 1

    def assessSituation(self, gameState):  # TODO: Add case where we're right next to a food (even if we're carrying 5)
        myPos = gameState.getAgentPosition(self.index)

        directions = [Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST]
        enemyPositions = [pos for pos in self.getEnemyPositions(gameState) if pos]
        positions = [pos for pos in gameState.getWalls().asList(False) if pos[1] > 1]
        x1, y1 = max(positions)
        x0, y0 = min(positions)

        # If agent is not on home side, and any next action will take it
        # close (within 1 or less maze distance) to a ghost, do not take that action.
        # Instead, return to home side
        if not self.onOurSide(gameState, myPos):
            for index in range(4):
                action = directions[index]
                nextPos = self.getNextPosition(action, myPos)
                if x0 - 1 <= nextPos[0] <= x1 and y0 - 1 <= nextPos[1] <= y1 and not gameState.getWalls()[nextPos[0]][nextPos[1]]:
                    for enemyPos in enemyPositions:
                        dist = self.getMazeDistance(enemyPos, nextPos)
                        if (nextPos is enemyPos or dist <= 1) and self.onOurSide(gameState, nextPos):
                            return self.returnHomeGetAction(gameState)

        scaredTimer = gameState.getAgentState(self.index).scaredTimer
        # Agent should eat food when enemy has eaten a power pellet so
        # as not to get killed. When the scaredTimer gets below 10,
        # just return home
        if scaredTimer > 10:
            return self.eatClosestFoodGetAction(gameState)
        if scaredTimer > 0:
            return self.returnHomeGetAction(gameState)

        # Case for defensive agent
        if self.isDefense(gameState):
            return self.defenseGetAction(gameState)

        # Keep track of how much food is being carried
        currFood = self.countFood(gameState)
        self.carrying = self.initialFoodCount - self.getScore(gameState) - currFood

        enemyPositions = self.getEnemyPositions(gameState)
        # Determines whether to chase an enemy -
        # If enemy is on our side, chase, otherwise
        # return home because you are on the enemy side and
        # could get killed
        for enemyPos in enemyPositions:
            if enemyPos is not None:
                if self.onOurSide(gameState, enemyPos):
                    return self.eatEnemyGetAction(gameState)
                elif not self.onOurSide(gameState, myPos) and self.getMazeDistance(enemyPos, myPos) < 4:
                    return self.returnHomeGetAction(gameState)

        # Return home if our agents are carrying 5 or more
        # food pellets collectively
        if self.carrying >= 5:
            action = self.returnHomeGetAction(gameState)
            if action is None:
                return self.eatClosestFoodGetAction(gameState)
            else:
                return action
        # If agents have nothing else to do, eat food
        else:
            return self.eatClosestFoodGetAction(gameState)

    def isDefense(self, gameState):
        for index in self.getTeam(gameState):
            if index is self.index:
                myPos = gameState.getAgentPosition(index)
            else:
                teamPos = gameState.getAgentPosition(index)
        # TODO: fix so that this works regardless of team (because blue team accrues negative score)
        # If score is negative, need more points!
        if gameState.getScore() < 0:
            return False
        # Case where we're winning by enough to just do defense
        if gameState.getScore() > 7:
            return True

        if gameState.isOnRedTeam(self.index):
            return myPos[0] < teamPos[0]
        else:
            return myPos[0] > teamPos[0]

    def getEnemyPositions(self, gameState):
        enemyPositions = []
        for agent in self.getOpponents(gameState):
            enemyPositions.append(gameState.getAgentPosition(agent))
        return enemyPositions

    def eatClosestFoodGetAction(self, gameState):
        path = self.aStarSearch(gameState, self.isGoalState, self.heuristic, 15)
        return path[0]

    def returnHomeGetAction(self, gameState):
        path = self.aStarSearch(gameState, self.isGoalStateReturn, self.heuristicReturn, 15)
        if not path:
            return None

        return path[0]

    def defenseGetAction(self, gameState):  # TODO: Make sure we actually chase the right ghost
        # Want something that searches towards the ghost closest to our side, staying towards the middle
        distributions = self.getDistribution(0), self.getDistribution(1)
        found = False

        # Determine whether we should chase after an enemy
        # If it is in our 3/4 of the board, start chasing
        for dist in distributions:
            maxD = max(dist)
            maxP = dist.argMax()
            if gameState.isOnRedTeam(self.index):
                if maxD > 0.1 and maxP[0] < 0.75 * self.middle:
                    found = True
            else:
                if maxD > 0.1 and maxP[0] > 0.75 * self.middle:
                    found = True

        if not found:
            path = self.aStarSearch(gameState, self.isGoalStateReturn, self.heuristicReturn, 0)
        else:
            path = self.aStarSearch(gameState, self.isGoalStateDefense, self.heuristicDefense, -1)

        if path:
            return path[0]
        else:
            actions = gameState.getLegalActions(self.index)
            return random.choice(actions)

    def eatEnemyGetAction(self, gameState):
        path = self.aStarSearch(gameState, self.isGoalStateDefense, self.heuristicDefense, -1)
        return path[0]

    def countFood(self, gameState):
        if gameState.isOnRedTeam(self.index):
            food = gameState.getBlueFood()
        else:
            food = gameState.getRedFood()

        positions = [pos for pos in gameState.getWalls().asList(False) if pos[1] > 1]
        x1, y1 = max(positions)
        x0, y0 = min(positions)
        return len(food.asList(True))

    def aStarSearch(self, gameState, goalCheck, heuristic, costWeight):
        # Search the node that has the lowest combined cost and heuristic first
        myState = gameState.getAgentState(self.index)
        pos = myState.getPosition()

        current = self.MyNode(pos, None, 0, None)

        # Create the frontier (stack)
        frontier = util.PriorityQueue()
        frontier.push(current, current.getPathCost() + heuristic(gameState, pos))

        # Create an empty set ("explored")
        explored = []

        # Start loop (if frontier is not empty)
        while not frontier.isEmpty():
            # Pop from the frontier and assign that to node
            current = frontier.pop()
            # Check if the node.getState is a goal state, if so return plan (list)
            if goalCheck(gameState, current.getPos()):
                return self.getPlan(current)

            # Check if node.state is not in explored
            if current.getPos() not in explored:
                # Add node to explored, add node's successors to frontier
                explored.append(current.getPos())

                # While loop to loop through tuples and create children
                successors = self.getSuccessors(gameState, current.getPos(), costWeight)
                for tpl in successors:
                    child = self.MyNode(tpl[0], current, current.getPathCost() + tpl[2], tpl[1])
                    frontier.push(child, child.getPathCost() + heuristic(gameState, child.getPos()))

        # Return failure
        return []

    def isGoalState(self, gameState, pos):
        if gameState.isOnRedTeam(self.index):
            food = gameState.getBlueFood()
        else:
            food = gameState.getRedFood()
        x, y = int(pos[0]), int(pos[1])

        if food[x][y]:
            return True

        return False

    def isGoalStateReturn(self, gameState, pos):
        if pos is None:
            return None

        return pos[0] is int(self.middle)

    def isGoalStateDefense(self, gameState, pos):
        # Find position with maximum probability
        distributions = [self.getDistribution(0), self.getDistribution(1)]
        maxPositions = [distributions[0].argMax(), distributions[1].argMax()]

        if pos in maxPositions:
            return True
        return False

    def heuristic(self, gameState, pos):
        if gameState.isOnRedTeam(self.index):
            food = gameState.getBlueFood()
        else:
            food = gameState.getRedFood()

        distances = []
        positions = [pos for pos in gameState.getWalls().asList(False) if pos[1] > 1]
        x1, y1 = max(positions)
        x0, y0 = min(positions)

        for x in range(x0, x1):
            for y in range(y0, y1):
                if food[x][y]:
                    distances.append(self.getMazeDistance(pos, (x, y)))

        return min(distances)

    def heuristicReturn(self, gameState, pos):
        # Find fastest way home
        walls = gameState.getWalls()
        positions = [pos for pos in gameState.getWalls().asList(False) if pos[1] > 1]
        x1, y1 = max(positions)
        x0, y0 = min(positions)
        middle = int(self.middle)
        dists = [self.getMazeDistance(pos, (middle, y)) for y in range(y0, y1) if walls[middle][y] is False]

        return min(dists)

    def heuristicDefense(self, gameState, pos):  # TODO: Improve if we have time
        distributions = [self.getDistribution(0), self.getDistribution(1)]
        positions = [distributions[0].argMax(), distributions[1].argMax()]
        enemyDists = []

        for enemyPos in positions:
            if gameState.getWalls()[enemyPos[0]][enemyPos[1]]:
                return abs(enemyPos[0] - pos[0]) + abs(enemyPos[1] - pos[1])
            enemyDists.append(self.getMazeDistance(pos, enemyPos))

        return min(enemyDists)

    def getPlan(self, current):
        # Pop from the frontier and assign that to node
        # Check if the node.getState is a goal state, if so return plan (list)
        plan = []

        while current.parent is not None:
            plan.append(current.getAction())
            current = current.getParent()

        plan.reverse()

        return plan

    class MyNode:

        def __init__(self, position, parent, pathCost, action):
            self.position = position
            self.parent = parent
            self.pathCost = pathCost
            self.action = action

        def getPos(self):
            return self.position

        def getParent(self):
            return self.parent

        def getPathCost(self):
            return self.pathCost

        def getAction(self):
            return self.action

    def getSuccessors(self, gameState, pos, costWeight):
        """
    Returns successor states, the actions they require, and a cost of 1.

     As noted in search.py:
        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost'
        is the incremental cost of expanding to that successor
    """

        x, y = int(pos[0]), int(pos[1])
        successors = []

        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            nextx, nexty = self.getNextPosition(action, (x, y))
            if not gameState.getWalls()[nextx][nexty]:
                nextPos = nextx, nexty
                dist0 = self.getDistribution(0)
                dist1 = self.getDistribution(1)

                if self.onOurSide(gameState, nextPos):
                    cost = 1
                else:
                    cost = costWeight * (dist0[nextPos] + dist1[nextPos])
                successors.append((nextPos, action, cost))

        return successors

    def getNextPosition(self, action, pos):
        x, y = pos
        if action == Directions.NORTH:
            nextx = x
            nexty = y + 1
        elif action == Directions.SOUTH:
            nextx = x
            nexty = y - 1
        elif action == Directions.EAST:
            nextx = x + 1
            nexty = y
        else:
            nextx = x - 1
            nexty = y

        return nextx, nexty

    def particleObserve(self, gameState):  # TODO: Debug issues with particleFilter
        noisyDistances = gameState.getAgentDistances()

        # Create 2000 particles distributed uniformly,
        # flipping weighted coin based on transition model
        # Move particles accordingly
        # Weigh particles based on how compatible they are with evidence (emission model)
        # Use weighted particles to create probability distribution
        # Sample 2000 new particles from that distribution

        for i in range(len(self.enemies)):
            distribution = util.Counter()
            myPos = gameState.getAgentPosition(self.index)

            for particlePos in particles[i]:
                distance = util.manhattanDistance(particlePos, myPos)
                noisyDistance = noisyDistances[self.enemies[i]]
                # Weight each particle's probability based on the likeliness it gave me the evidence I have
                distribution[particlePos] += gameState.getDistanceProb(distance, noisyDistance)

            distribution.normalize()
            if distribution.totalCount() is 0:  # Special case
                self.initializeParticles(gameState)
                return

            # Sample particles uniformly from the current distribution
            particles[i] = []
            for j in range(self.numParticles):
                particles[i].append(util.sample(distribution))

    def getDistribution(self, index):
        distribution = util.Counter()

        for particlePos in particles[index]:
            distribution[particlePos] += 1
        distribution.normalize()

        return distribution

    def particleElapseTime(self, gameState, index):
        # Move each particle according to the transition model
        positions = [pos for pos in gameState.getWalls().asList(False) if pos[1] > 1]
        x1, y1 = max(positions)
        x0, y0 = min(positions)
        particles1 = []

        for particle in particles[index]:
            nextPositions = []
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                nextx, nexty = self.getNextPosition(action, particle)
                if x0 - 1 <= nextx <= x1 and y0 - 1 <= nexty <= y1:
                    nextPositions.append((nextx, nexty))

            transition = self.getPositionDistribution(gameState, nextPositions)
            particles1.append(util.sample(transition))

        particles[index] = particles1

    def getPositionDistribution(self, gameState, positions):
        toReturn = util.Counter()
        for pos in positions:
            # Calculate distance from middle
            dist = abs(self.middle - pos[0])
            # Weight the distribution based on how close to the middle it is
            toReturn[pos] += 1 / (dist + 1)
            if gameState.getWalls()[pos[0]][pos[1]]:
                toReturn[pos] *= 0.01

        toReturn.normalize()
        return toReturn
