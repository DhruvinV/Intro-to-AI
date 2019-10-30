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
def eating_all_food(state,gameState):
  position, foodlist = state
  "*** YOUR CODE HERE ***"
  #create graph from food grid.
  foodlist.append(position)
  if(gameState.isWin()or gameState.isLose()or len(foodlist)==0):
      return []
  vertex = {}
  edges = {}
  for i in enumerate(foodlist):
      key = i[1]
      vertex[key] = i[0]
  pq,vertice,edge = convertograph(foodlist,vertex)
  # disjoint for mst
  cluster = {}
  rank = {}
  for i in vertice.values():
      cluster[i] = i
      rank[i] = 0
  estimate = []
  tuple=set()
  while pq.isEmpty() == False:
      edge = pq.pop()
      x,y = edge[0],edge[1]
      # find parents
      while(cluster[x] != x):
          x = cluster[x]
      while(cluster[y]!=y):
          y = cluster[y]
      # update ranks if each disjoint sets
      if(x != y):
          if(rank[x]>rank[y]):
              cluster[x] = y
          elif(rank[x]<rank[y]):
              cluster[y] = x
          elif(rank[x]==rank[y]):
              cluster[x]=y
              rank[x] = rank[x]+1
          estimate += edge
  return estimate
def convertograph(foodlist,vertex,edges={}):
  pq = util.PriorityQueue()
  edges={}
  for j in (foodlist):
      for k in foodlist:
          if(j != k and (((vertex[j],vertex[k]) not in edges.keys()))):
              distance = manhattanDistance(j,k)
              edges[(vertex[j],vertex[k])] = distance
              pq.push((vertex[j],vertex[k],distance),distance)
  return (pq,vertex,edges)
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
        fooddistance = []
        ghosdistance = []
        for i in newGhostStates:
          ghosdistance.append(manhattanDistance(i.getPosition(),newPos))
        dis = 0
        if(len(currentGameState.getFood().asList()) != len(newFood.asList())):
          for i in newFood:
            fooddistance.append(1/manhattanDistance(i,newPos))
          dis = 1/2*(min(fooddistance) + min(ghosdistance)) 
        if(min(ghosdistance) < 2 and dis ==0):
          return -1000
        return successorGameState.getScore() + dis - max(ghosdistance)
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
    #hash_table to store all information
    def DFMiniMax(self,pos,my_table):
      best_move = None
      if(pos.isLose() or pos.isWin() or my_table["moves"] == self.depth * pos.getNumAgents()):
        return (best_move, self.evaluationFunction(pos))
      if(my_table["player"] == "MAX"):
          value = - float("inf")
      if(my_table["player"] == "MIN"):
          value = float("inf")
      actions = pos.getLegalActions(my_table["agent_index"])
      new_my_table = self.updatemydict(my_table,pos)
      for move in actions:
        nxt_pos = pos.generateSuccessor(my_table["agent_index"],move)
        nxt_move,nxt_val = self.DFMiniMax(nxt_pos,new_my_table)
        if my_table["player"] == "MAX" and value < nxt_val: 
            value, best_move = nxt_val, move
        if my_table["player"] == "MIN" and value > nxt_val:
            value, best_move = nxt_val, move
      return (best_move,value)
      
    def updatemydict(self,my_table,gameState):
      new_dict = dict()
      new_dict["player"] = None
      new_dict["agent_index"] = None
      new_dict["moves"] = None
      new_dict["moves"] = my_table["moves"] + 1
      new_dict["agent_index"] = new_dict["moves"] % gameState.getNumAgents()
      if(new_dict["agent_index"] == 0):
        new_dict["player"] = 'MAX'
      else:
        new_dict["player"] = 'MIN'
      return new_dict
  
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
        "*** YOUR CODE HERE ***"
        # build dict to store important info such as moves,player, turn etc
        # total avaialble moves is equal to self.depth*numberofagents
        my_table = dict()
        my_table["player"] = "MAX"
        my_table["moves"] = 0
        my_table["agent_index"] = 0
        bestMove, bestVal = self.DFMiniMax(gameState, my_table)
        return bestMove
     

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def AlphaBeta(self,pos,my_table,beta,alpha):
      best_move = None
      if(pos.isLose() or pos.isWin() or my_table["moves"] == self.depth * pos.getNumAgents()):
        return (best_move, self.evaluationFunction(pos))
      if(my_table["player"] == "MAX"):
          value = - float("inf")
      if(my_table["player"] == "MIN"):
          value = float("inf")
      actions = pos.getLegalActions(my_table["agent_index"])
      for move in actions:
        new_my_table = self.updatemydict(my_table,pos)
        nxt_pos = pos.generateSuccessor(my_table["agent_index"],move)
        nxt_move,nxt_val = self.AlphaBeta(nxt_pos,new_my_table,beta,alpha)
        if(my_table["player"] == "MAX"):
          if(value < nxt_val):
            value,best_move = nxt_val,move
          if(value >= beta):
            return best_move, value
          alpha = max(alpha,value)
        if(my_table["player"] == "MIN"):
          if(value > nxt_val):
            value,best_move = nxt_val,move
          if(value <=  alpha):
            return best_move,value
          beta = min(beta,value)
      return (best_move,value)
    def updatemydict(self,my_table,gameState):
      new_dict = dict()
      new_dict["player"] = None
      new_dict["agent_index"] = None
      new_dict["moves"] = None
      new_dict["moves"] = my_table["moves"] + 1
      new_dict["agent_index"] = new_dict["moves"] % gameState.getNumAgents()
      if(new_dict["agent_index"] == 0):
        new_dict["player"] = 'MAX'
      else:
        new_dict["player"] = 'MIN'
      return new_dict
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # build dict to store important info such as moves,player, turn etc
        # total avaialble moves is equal to self.depth*numberofagents
        my_table = dict()
        my_table["player"] = "MAX"
        my_table["moves"] = 0
        my_table["agent_index"] = 0
        bestMove, bestVal = self.AlphaBeta(gameState, my_table,float('inf'),-float('inf'))
        return bestMove

class ExpectimaxAgent(MultiAgentSearchAgent):

    def Expectimax(self,pos,my_table):
      best_move = None
      if(pos.isLose() or pos.isWin() or my_table["moves"] == self.depth * pos.getNumAgents()):
        return (best_move, self.evaluationFunction(pos))
      if(my_table["player"] == "MAX"):
          value = - float("inf")
      if(my_table["player"] == "MIN"):
          value = 0
      actions = pos.getLegalActions(my_table["agent_index"])
      new_my_table = self.updatemydict(my_table,pos)
      for move in actions:
        nxt_pos = pos.generateSuccessor(my_table["agent_index"],move)
        nxt_move,nxt_val = self.Expectimax(nxt_pos,new_my_table)
        if my_table["player"] == "MAX" and value < nxt_val: 
            value, best_move = nxt_val, move
        if my_table["player"] == "MIN":
            value = value + float(float(float(1.0)/float(len(actions))) * nxt_val)
      return (best_move,value)
    def updatemydict(self,my_table,gameState):
      new_dict = dict()
      new_dict["player"] = None
      new_dict["agent_index"] = None
      new_dict["moves"] = None
      new_dict["moves"] = my_table["moves"] + 1
      new_dict["agent_index"] = new_dict["moves"] % gameState.getNumAgents()
      if(new_dict["agent_index"] == 0):
        new_dict["player"] = 'MAX'
      else:
        new_dict["player"] = 'MIN'
      return new_dict
    def getAction(self, gameState):
      my_table = dict()
      my_table["player"] = "MAX"
      my_table["moves"] = 0
      my_table["agent_index"] = 0
      bestMove, bestVal = self.Expectimax(gameState, my_table)
      return bestMove
      
def betterEvaluationFunction(currentGameState):
  """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      Evalaution function uses weighted average calculation function based different features.
      - Closest Distance to scared ghost(ghost-hunting)
      - Closest Distance to capsules(pellet-nabbing)
      - Distance to closest to food pellet(good-gobbling)
      - Distance to closest non-scared ghost.
      - Number of food pellets left.
      - Number of capsules left.   
  """
  "*** YOUR CODE HERE ***"
  distance = {"scare-Ghost": None,"food-pellet-num": None,"capsules-num": None,"capsules-dis": None,"food-dis": None,"non-scare-Ghost": None}
  position = currentGameState.getPacmanPosition()
  food = currentGameState.getFood()
  ghosts = currentGameState.getGhostStates()
  capsules = currentGameState.getCapsules()
  # number of capsules left
  distance["capsules-num"] = len(capsules)
  if(len(capsules)> 0):
    cap_dis = []
    for i in capsules:
      cap_dis.append(manhattanDistance(i,position))
    distance["capsules-dis"] = 1/min(cap_dis)
  else:
    distance["capsules-dis"] = 0
  # calculate scard ghost dist
  newGhostStates = currentGameState.getGhostStates()
  non_scared_ghosts = [ghosts for ghosts in newGhostStates if ghosts.scaredTimer == 0]
  if(len(non_scared_ghosts)>0):
    non_scar_dis = []
    for i in non_scared_ghosts:
      non_scar_dis.append(manhattanDistance(i.getPosition(),position))
    min_dis = min(non_scar_dis)
    if(min_dis < 1):
      distance["non-scare-Ghost"] = 100000
    else:
      distance["non-scare-Ghost"] = 1/min_dis
  else:
    distance["non-scare-Ghost"] = 0
  # caclulate for scared  ghosts
  newGhostStates = currentGameState.getGhostStates()
  non_scared_ghosts = [ghosts for ghosts in newGhostStates if ghosts.scaredTimer > 0]
  if(len(non_scared_ghosts)>0):
    non_scar_dis = []
    for i in non_scared_ghosts:
      non_scar_dis.append(manhattanDistance(i.getPosition(),position))
    min_dis = min(non_scar_dis)
    if(min_dis < 1):
      distance["scare-Ghost"] = 100
    else:
      distance["scare-Ghost"] = 1/min_dis
  else:
    distance["scare-Ghost"] = 0
  # get esitmate of the eating all the food dots from current postion
  distance["food-pellet-num"] = len(food.asList())
  length = len(food.asList())//4
  foodinput= food.asList()[0:length]
  estimate = eating_all_food((position,foodinput),currentGameState)
  if(len(estimate) == 0):
    distance["food-dis"] = 0
  else:
    distance["food-dis"] = 1/len(estimate)
  # distance = {"scare-Ghost": None,"food-pellet-num": None,"capsules-num": None,"capsules-dis": None,"food-dis": None,"non-scare-Ghost": None}    
  score = 7*distance["scare-Ghost"]-20*distance["food-pellet-num"]-90*distance["capsules-num"]+7*distance["capsules-dis"]+2*distance["food-dis"]-6*distance["non-scare-Ghost"]
  return score
    
# Abbreviation
better = betterEvaluationFunction
