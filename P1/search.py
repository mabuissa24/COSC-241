# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

class MyNode:
    #state
    #most recent action
    #pointer to parent
    #total pathcost

    def __init__(self, state, parent, pathCost, action):
        self.state = state
        self.parent = parent
        self.pathCost = pathCost
        self.action = action



    def getState(self):
        return self.state

    def getParent(self):
        return self.parent
    def getPathCost(self):
        return self.pathCost
    def getAction(self):
        return self.action





def getPlan(current, problem):
    # Pop from the frontier and assign that to node
    # Check if the node.getState is a goal state, if so return # return plan (list)
    plan = []
    while current.parent is not None:
        plan.append(current.getAction())
        current = current.getParent()

    plan.reverse()
    return plan


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    ## pacman is a sequence of plans
    ## path , store it explicitly in the node class, or
    ## have the node remember what their parent was ( have  a parent paramete in node) --> what ever my action was and go ask my parent
    #having path makes debugging easier
    ## we have to store state in the node. Store pointer in parent node. TO add plan: what is the last actionw e took
    # Create a node that stores problem

    current = MyNode(problem.getStartState(), None, 0, None)



    # Create the frontier (stack)

    frontier = util.Stack()
    frontier.push(current)

    # Create an empty set ("explored")
    explored = []

    # Start loop (if frontier is not empty)
    while not frontier.isEmpty():
        # Pop from the frontier and assign that to node
        current = frontier.pop()
        # Check if the node.getState is a goal state, if so return # return plan (list)
        if problem.isGoalState(current.getState()):
            return getPlan(current, problem)

        # Check if node.state is not in explored
        if current.getState() not in explored:


            # Add node to explored, add node's sucessors to frontier
            explored.append(current.getState())

            ## while loop to loop through touples and create children
            listOfChildren = problem.getSuccessors(current.getState())
            for tpl in listOfChildren:
                child = MyNode(tpl[0],current,current.getPathCost()+tpl[2],tpl[1])
                frontier.push(child)

                ## for loop through list
                ## for x in list
                    ## child = myNode(,x,x)
                    ## state from tuple (at index 0)
                    # parent is current
                    # pathcost is current.getPathCost + cost from tuple (index 2)
                    # action from tuple (index 1)
                    # push child onto frontier

    # Return failure

    return []

   # print(problem.getStartState())
   # print "Is the start a goal?", problem.isGoalState(problem.getStartState())
   # print "Start's successors:", problem.getSuccessors(problem.getStartState())
   # print problem.getSuccessors(problem.getSuccessors(problem.getStartState())[0][0])

    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    current = MyNode(problem.getStartState(), None, 0, None)

    # Create the frontier (queue)

    frontier = util.Queue()
    frontier.push(current)

    # Create an empty set ("explored")
    explored = []

    # Start loop (if frontier is not empty)
    while not frontier.isEmpty():
        # Pop from the frontier and assign that to node
        current = frontier.pop()
        # Check if the node.getState is a goal state, if so return # return plan (list)
        if problem.isGoalState(current.getState()):
            return getPlan(current, problem)

        # Check if node.state is not in explored
        if current.getState() not in explored:

            # Add node to explored, add node's sucessors to frontier
            explored.append(current.getState())

            ## while loop to loop through touples and create children
            listOfChildren = problem.getSuccessors(current.getState())
            for tpl in listOfChildren:
                child = MyNode(tpl[0], current, current.getPathCost() + tpl[2], tpl[1])
                frontier.push(child)

                ## for loop through list
                ## for x in list
                ## child = myNode(,x,x)
                ## state from tuple (at index 0)
                # parent is current
                # pathcost is current.getPathCost + cost from tuple (index 2)
                # action from tuple (index 1)
                # push child onto frontier

    # Return failure

    return []
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # algorithm seems like it doesn't fully take advantage of cheap paths (dottedMaze)
    # expands too many nodes (doesn't reach goal state before exploring more expensive nodes)

    current = MyNode(problem.getStartState(), None, 0, None)

    # Create the frontier (stack)

    frontier = util.PriorityQueue()
    frontier.push(current, current.getPathCost())

    # Create an empty set ("explored")
    explored = []

    # Start loop (if frontier is not empty)
    while not frontier.isEmpty():
        # Pop from the frontier and assign that to node
        current = frontier.pop()
        # Check if the node.getState is a goal state, if so return # return plan (list)
        if problem.isGoalState(current.getState()):
            return getPlan(current, problem)

        # Check if node.state is not in explored
        if current.getState() not in explored:

            # Add node to explored, add node's sucessors to frontier
            explored.append(current.getState())

            ## while loop to loop through touples and create children
            listOfChildren = problem.getSuccessors(current.getState())
            for tpl in listOfChildren:
                child = MyNode(tpl[0], current, current.getPathCost() + tpl[2], tpl[1])
                frontier.push(child, child.getPathCost())

                ## for loop through list
                ## for x in list
                ## child = myNode(,x,x)
                ## state from tuple (at index 0)
                # parent is current
                # pathcost is current.getPathCost + cost from tuple (index 2)
                # action from tuple (index 1)
                # push child onto frontier



    # Return failure

    return []
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    print("START")

    current = MyNode(problem.getStartState(), None, 0, None)
    #print("HEURISTIC BELOW")
    #print(heuristic(current.getState(), problem))

    # Create the frontier (stack)

    frontier = util.PriorityQueue()
    frontier.push(current, current.getPathCost() + heuristic(current.getState(), problem))

    # Create an empty set ("explored")
    explored = []

    # Start loop (if frontier is not empty)
    while not frontier.isEmpty():
        # Pop from the frontier and assign that to node
        current = frontier.pop()
        # Check if the node.getState is a goal state, if so return # return plan (list)
        if problem.isGoalState(current.getState()):
            return getPlan(current, problem)

        # Check if node.state is not in explored
        if current.getState() not in explored:

            # Add node to explored, add node's sucessors to frontier
            explored.append(current.getState())

            ## while loop to loop through touples and create children
            listOfChildren = problem.getSuccessors(current.getState())
            for tpl in listOfChildren:
                child = MyNode(tpl[0], current, current.getPathCost() + tpl[2], tpl[1])
                frontier.push(child, child.getPathCost() + heuristic(tpl[0], problem))

                ## for loop through list
                ## for x in list
                ## child = myNode(,x,x)
                ## state from tuple (at index 0)
                # parent is current
                # pathcost is current.getPathCost + cost from tuple (index 2)
                # action from tuple (index 1)
                # push child onto frontier

    # Return failure

    return []


    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
