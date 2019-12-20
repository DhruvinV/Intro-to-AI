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
    st = util.Stack()
    s1 = problem.getStartState()
    st.push(([s1],[]))
    flag = False
    visited = set()
    while(st.isEmpty() ==  False):
        popped = st.pop()
        s = popped[0][-1]
        if(problem.isGoalState(s)== True):
            # popped[1].append(popped[1][-1])
            return popped[1]
        if(s not in visited):
            visited.add(s)
            successors = problem.getSuccessors(s)
            for i in successors:
                if(i[0] not in visited):
                    l1,l2 = popped
                    l1 = list(l1)
                    l2 = list(l2)
                    l1.append(i[0])
                    l2.append(i[1])
                    st.push(tuple((l1,l2)))
    return False

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    st = util.Queue()
    s1 = problem.getStartState()
    print(s1)
    st.push(([s1],[]))
    flag = False
    visited = set()
    cost_dict = {}
    cost_dict[s1] = 0
    while(st.isEmpty() ==  False):
        popped = st.pop()
        s = popped[0][-1]
        cost_s = len(popped[1])
        if(cost_s <= cost_dict[s]):
            if(problem.isGoalState(s)== True):
                # popped[1].append(popped[1][-1])
                return popped[1]
            successors = problem.getSuccessors(s)
            for i in successors:
                if i[0] not in cost_dict.keys() or (len(popped[1])+1) < cost_dict[i[0]]:
                    # print(cost_dict)
                    l1,l2 = popped
                    l1 = list(l1)
                    l2 = list(l2)
                    l1.append(i[0])
                    l2.append(i[1])
                    st.push(tuple((l1,l2)))
                    cost_dict[i[0]] = len(popped[1])+1
    return False


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    st = util.PriorityQueue()
    s1 = problem.getStartState()
    st.push(([s1],[],[0]),0)
    cost_dict = {}
    cost_dict[s1] = 0
    while(st.isEmpty() ==  False):
        popped = st.pop()
        s = popped[0][-1]
        cost_s = sum(popped[2])
        if(cost_s <= cost_dict[s]):
            if(problem.isGoalState(s)== True):
                return popped[1]
            successors = problem.getSuccessors(s)
            for i in successors:
                if i[0] not in cost_dict.keys() or (((cost_s)+i[2]) < cost_dict[i[0]]):
                    l1,l2,l3 = popped
                    l1 = list(l1)
                    l2 = list(l2)
                    l3 = list(l3)
                    l1.append(i[0])
                    l2.append(i[1])
                    l3.append(i[2])
                    st.push(tuple((l1,l2,l3)),sum(l3))
                    cost_dict[i[0]] = sum(l3)
    return False
def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    st = util.PriorityQueue()
    s1 = problem.getStartState()
    st.push(([s1],[],[0]),0)
    cost_dict = {}
    cost_dict[s1] = 0
    while(st.isEmpty() ==  False):
        popped = st.pop()
        s = popped[0][-1]
        cost_s = sum(popped[2])
        if(cost_s <= cost_dict[s]):
            if(problem.isGoalState(s)== True):
                return popped[1]
            successors = problem.getSuccessors(s)
            for i in successors:
                if i[0] not in cost_dict.keys() or (((cost_s)+i[2])< cost_dict[i[0]]):
                    l1,l2,l3 = popped
                    l1 = list(l1)
                    l2 = list(l2)
                    l3 = list(l3)
                    l1.append(i[0])
                    l2.append(i[1])
                    l3.append(i[2])
                    st.push(tuple((l1,l2,l3)),sum(l3)+heuristic(i[0],problem))
                    cost_dict[i[0]] = sum(l3)
    return False

# class node:
#     def __init__(self,s1):


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
