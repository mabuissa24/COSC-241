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

class DummyAgent(CaptureAgent):
    """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

    def registerInitialState(self, gameState):
        """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

        '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
    Your initialization code goes here, if you need any.
    '''

    def chooseAction(self, gameState):
        """
    Picks among actions randomly.
    """
        actions = gameState.getLegalActions(self.index)

        '''
    You should change this in your own agent.
    '''

        return random.choice(actions)


particles = [[], []]


class SmartyAgent(CaptureAgent):
    def registerInitialState(self, gameState):

        # print 'in registerInitialState'
        CaptureAgent.registerInitialState(self, gameState)
        self.carrying = 0
        self.foodToEat = self.countFood(gameState)
        self.initialFoodCount = self.countFood(gameState)
        # Create n particles in their start state - can find this as part of the map
        if gameState.isOnRedTeam(self.index):
            self.enemies = gameState.getBlueTeamIndices()
        else:
            self.enemies = gameState.getRedTeamIndices()
        for i in range(len(self.enemies)):
            for index in range(1000):
                particles[i].append(gameState.getInitialAgentPosition(self.enemies[i]))

        positions = [pos for pos in gameState.getWalls().asList(False) if pos[1] > 1]
        x1, y1 = max(positions)
        x0, y0 = min(positions)
        self.middle = x0 + float(x1 - x0) / 2

    def initializeParticles(self, gameState):
        if gameState.isOnRedTeam(self.index):
            self.enemies = gameState.getBlueTeamIndices()
        else:
            self.enemies = gameState.getRedTeamIndices()
        # print 'resetting particles'
        # Create n particles randomly distributed
        positions = [pos for pos in gameState.getWalls().asList(False) if pos[1] > 1]
        for i in range(len(self.enemies)):
            for index in range(1000):
                pos = positions[index % len(positions)]
                particles[i].append(pos)

        # print 'in initializeParticles, distribution is', self.getDistribution(0), 'for index 0'

    def chooseAction(self, gameState):  # TODO: Debug defense, account for power pellets, make personalities better?
        # print 'type of self.getDistribution is', type(self.getDistribution(0))
        dis1 = self.getDistribution(0)
        dis2 = self.getDistribution(1)
        self.displayDistributionsOverPositions([dis1, dis2, util.Counter(), util.Counter()])
        # Update in time for the agent before you
        # Update using observe

        self.particleObserve(gameState)
        if self.index == 0 or self.index == 3:
            self.particleElapseTime(gameState, 1)
        if self.index == 1 or self.index == 2:
            self.particleElapseTime(gameState, 0)

        # TODO: Debug special case when we can see the ghosts
        enemyPositions = self.getEnemyPositions(gameState)
        for index in range(len(self.enemies)):
            if enemyPositions[index] is not None:
                particles[index] = []
                for i in range(1000):
                    particles[index].append(enemyPositions[index])

        # dist = self.getDistribution(0)
        # maxKey = dist.argMax()
        # print 'most likely particle for', self.enemies[0], 'is', maxKey, 'with', dist[maxKey]
        # dist = self.getDistribution(1)
        # maxKey = dist.argMax()
        # print 'most likely particle for', self.enemies[1], 'is', maxKey, 'with', dist[maxKey]

        return self.assessSituation(gameState)

    def onOurSide(self, gameState, position):
        if gameState.isOnRedTeam(self.index):
            return position[0] < self.middle - 1
        else:
            return position[0] > self.middle + 1

    def assessSituation(self, gameState):  # TODO: Add never kill yourself condition
        # How to order priority
        # Can use chain of if elses
        # Could compute things like danger level
        # Commit to some kind of string of actions
        # Under no circumstance kill yourself

        # Can Q Learn a function that takes a state and many features and gives a personality

        # Have a distribution over each personality, experiment with how weights work (overly balanced, aggressive, etc)

        if self.isDefense(gameState):
            return self.defenseGetAction(gameState)

        currFood = self.countFood(gameState)
        self.carrying = self.initialFoodCount - self.getScore(gameState) - currFood

        myPos = gameState.getAgentPosition(self.index)

        enemyPositions = self.getEnemyPositions(gameState)
        for enemyPos in enemyPositions:
            if enemyPos is not None:
                if self.onOurSide(gameState, enemyPos):
                    return self.eatEnemyGetAction(gameState)
                elif not self.onOurSide(gameState, myPos) and self.getMazeDistance(enemyPos, myPos) < 4:
                    return self.returnHomeGetAction(gameState)

        if self.carrying >= 5:
            action = self.returnHomeGetAction(gameState)
            if action is None:
                return self.eatClosestFoodGetAction(gameState)
            else:
                return action
        else:
            return self.eatClosestFoodGetAction(gameState)

    def isDefense(self, gameState):
        for index in self.getTeam(gameState):
            if index is self.index:
                myPos = gameState.getAgentPosition(index)
            else:
                teamPos = gameState.getAgentPosition(index)
        if gameState.getScore() < 0:  # Not sure if I really want this case -- maybe better if we're close to middle
            return False
        if gameState.getScore() > 7:  # Case where we're winning by enough to just do defense
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

    def defenseGetAction(self, gameState):
        # Want something that searches towards the ghost closest to our side, staying towards the middle
        # TODO: Change so that it stays towards the middle but on our side of the board
        distributions = self.getDistribution(0), self.getDistribution(1)
        found = False

        for dist in distributions:
            maxD = max(dist)
            maxP = dist.argMax()
            if gameState.isOnRedTeam(self.index):
                if maxD > 0.25 and maxP[0] < 0.75 * self.middle:
                    found = True
            else:
                if maxD > 0.25 and maxP[0] > 0.75 * self.middle:
                    found = True
        if not found:
            path = self.aStarSearch(gameState, self.isGoalStateReturn, self.heuristicReturn, 0)
        else:
            path = self.aStarSearch(gameState, self.isGoalStateDefense, self.heuristicDefense, 0)

        if path:
            return path[0]
        else:
            print "taking random action"
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
        """Search the node that has the lowest combined cost and heuristic first."""

        # Nodes need to have a parent (so that they can get the path), a gameState (position), an action, and path cost

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
            # Check if the node.getState is a goal state, if so return # return plan (list)
            if goalCheck(gameState, current.getPos()):
                return self.getPlan(current)

            # Check if node.state is not in explored
            if current.getPos() not in explored:
                # Add node to explored, add node's successors to frontier
                explored.append(current.getPos())

                # while loop to loop through tuples and create children
                successors = self.getSuccessors(gameState, current.getPos(), costWeight)
                for tpl in successors:
                    child = self.MyNode(tpl[0], current, current.getPathCost() + tpl[2], tpl[1])
                    enemyPos = self.getEnemyPositions(gameState)
                    if child.getPos() not in enemyPos:
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

    def isGoalStateReturn(self, gameState, pos):  # Might need to have multiple options for this?
        if pos is None:
            return None
        return pos[0] is int(self.middle)


    # def isGoalStateDefense(self, gameState, pos):
    #     enemyPositions = self.getEnemyPositions(gameState)
    #     for enemyPos in enemyPositions:
    #         if pos == enemyPos:
    #             return True
    #     return False
    #
    # def heuristicDefense(self, gameState, pos):
    #     enemyPositions = self.getEnemyPositions(gameState)
    #     enemyDists = []
    #     for enemyPos in enemyPositions:
    #         #print 'pos is', pos, 'and enemy pos is', enemyPos
    #         if enemyPos is not None:
    #             enemyDists.append(self.getMazeDistance(pos, enemyPos))
    #
    #     return min(enemyDists)

    def isGoalStateDefense(self, gameState, pos):  # TODO: FIX
        enemyPositions = []
        distributions = [self.getDistribution(0), self.getDistribution(1)]
        maxProbs = [max(distributions[0].values()), max(distributions[1].values())]
        # print 'maxProbs are', maxProbs[0], maxProbs[1]
        for key in distributions[1].keys():
            # print 'key is', key
            key = int(key[0]), int(key[1])
            value = distributions[1][key]
            # print "key is", key, 'and value is', value
            if abs(value - maxProbs[1]) < .05:  # Instead want to find the max ON OUR SIDE and take within .05 of that
                #print 'appending', key
                enemyPositions.append(key)
        for key in distributions[0].keys():
            # print 'key is', key
            value = distributions[0][key]
            if abs(value - maxProbs[0]) < .05:  # TODO: See comment above
                #print 'appending', key
                enemyPositions.append(key)
        # TODO: Fix - not actually adding anything to enemyPositions sometimes
        # Also it's searching towards the enemy that's further away and on the enemy side for some reason?

        if pos in enemyPositions:
            print 'passed goaltest for defense with pos', pos
            # print 'and enemyPositions', enemyPositions
            return True
        #print 'enemyPositions is', enemyPositions
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

    def heuristicReturn(self, gameState, pos):  # Might need to have multiple options for this?

        # self.distancer.getMazeDistances()  # I think this enables mazeDistance to work?
        walls = gameState.getWalls()
        # for x, y in range()
        # legalPositions = x, y in walls
        # max[pos[0] for pos in gameState.getWalls() if gameState.getWalls()[pos] == (False)]
        positions = [pos for pos in gameState.getWalls().asList(False) if pos[1] > 1]
        x1, y1 = max(positions)
        x0, y0 = min(positions)
        middle = int(self.middle)
        dists = [self.getMazeDistance(pos, (middle, y)) for y in range(y0, y1) if walls[middle][y] is False]
        return min(dists)

    def heuristicDefense(self, gameState, pos):  # TODO: Check this works
        distributions = [self.getDistribution(0), self.getDistribution(1)]
        positions = [distributions[0].argMax(), distributions[1].argMax()]
        enemyDists = []

        for enemyPos in positions:
            enemyDists.append(self.getMazeDistance(pos, enemyPos))

        return min(enemyDists)

    def getPlan(self, current):
        # Pop from the frontier and assign that to node
        # Check if the node.getState is a goal state, if so return # return plan (list)
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

        # for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
        #
        #     successor = gameState.generateSuccessor(self.index, action)
        #     pos = successor.getAgentState(self.index).getPosition()
        #     x, y = pos
        #     dx, dy = game.Actions.directionToVector(action)  # Is this reference to game legal?
        #     nextx, nexty = int(x + dx), int(y + dy)
        #     if not self.walls[nextx][nexty]:
        #         nextState = (nextx, nexty)
        #         cost = self.costFn(nextState)
        #         successors.append((nextState, action, cost))

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

    def particleObserve(self, gameState):  # TODO: Debug issues with particleFilter if we have time
        # print "in particleObserve"
        noisyDistances = gameState.getAgentDistances()
        # print 'I am', self.index
        # print 'noisyDistances are', noisyDistances

        # Create 2000 particles distributed uniformly
        # flipping weighted coin based on transition model
        # move particles accordingly
        # weigh particles based on how compatible they are with evidence (emission model)
        # Use weighted particles to create probability distribution
        # Sample 2000 brand new particles from that distribution

        for i in range(len(self.enemies)):
            distribution = util.Counter()
            myPos = gameState.getAgentPosition(self.index)

            for particlePos in particles[i]:  # Sometimes particlePos is None??????
                # print 'particlePos is', particlePos, 'and myPos is', myPos
                if particlePos is None:
                    print 'particles at', i, 'are', particles[i]
                distance = util.manhattanDistance(particlePos, myPos)
                noisyDistance = noisyDistances[self.enemies[i]]
                # Weight each particle's probability based on the likeliness it gave me the evidence I have

                # print 'disProb for', distance, noisyDistance, 'is', gameState.getDistanceProb(distance, noisyDistance)
                distribution[particlePos] += gameState.getDistanceProb(distance, noisyDistance)

            distribution.normalize()
            if distribution.totalCount() is 0:  # Special case
                # print 'distribution total count is 0'
                # print distribution
                # print self.particles[i]
                self.initializeParticles(gameState)
                return

            # Sample particles uniformly from the current distribution
            particles[i] = []
            for j in range(1000):
                particles[i].append(util.sample(distribution))

    def getDistribution(self, index):
        distribution = util.Counter()
        for particlePos in particles[index]:
            distribution[particlePos] += 1
        distribution.normalize()
        # print 'distribution is', distribution
        return distribution

    def particleElapseTime(self, gameState, index):
        # move each particle according to the transition model
        # Go through each legal action
        # Two extremes: equal probability, do what we would do
        # Middle ground: go towards middle with more probability
        # print 'distribution is', self.getDistribution(index), 'for index', index

        positions = [pos for pos in gameState.getWalls().asList(False) if pos[1] > 1]
        x1, y1 = max(positions)
        x0, y0 = min(positions)
        particles1 = []
        for particle in particles[index]:
            nextPositions = []
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                nextx, nexty = self.getNextPosition(action, particle)
                if x0 - 1 <= nextx <= x1 and y0 - 1 <= nexty <= y1:
                    # print 'action:', action, 'is legal'
                    # print 'this action takes us from', particle, 'to', nextx, nexty
                    nextPositions.append((nextx, nexty))
            # print 'nextPositions is', nextPositions
            # for x, y in range()
            # legalPositions = x, y in walls
            # max[pos[0] for pos in gameState.getWalls() if gameState.getWalls()[pos] == (False)]

            transition = self.getPositionDistribution(gameState, nextPositions)
            particles1.append(util.sample(transition))

        particles[index] = particles1

    def getPositionDistribution(self, gameState, positions):  # TODO: Check that this works
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

