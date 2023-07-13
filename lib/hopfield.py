from enum import Enum
from typing import Iterator
import random
import numpy as np
import scipy.linalg as la

class NetworkMode(Enum):
  LEARN, INFER = range(2)

class HopfieldNet:
  """
  Hopfield network with hebbian learning. Zero-threshold nodes.
  Implemented as an iterator that at each step runs the netwkork in inference or train mode.
  """
  def __init__(self, nodesNumber: int):
    # Below's matrix main diagional will always be zero since autoinhibition
    # and autoactivation are prohibited; moreover must be symmetric
    self.nodeWeights: np.ndarray = np.zeros((nodesNumber, nodesNumber))
    self.nodeValues: np.ndarray = np.zeros((1, nodesNumber))
    self.prevEnergy: float = 0.0
    self.nodesNumber: int = nodesNumber
    self.mode = NetworkMode.INFER
    self.safeguardCounter = 0

  def __iter__(self):
    return self
  
  def __next__(self):
    self.checkPlausibility()

    if self.mode == NetworkMode.INFER:
      return self.__inferStep() 
    elif self.mode == NetworkMode.LEARN:
      return self.__learnStep()
    
  def __inferStep(self):
    """
    Freely run the network (with clamped weights) until it coverges to an attractor
    """
    updatingNodeIdx = np.random.randint(self.nodesNumber)
    self.nodeValues[0][updatingNodeIdx] = np.heaviside(
      self.computeInputValue(updatingNodeIdx),
      0
    )

    if self.isStuckInAttractor():
      self.safeguardCounter += 1
    else:
      self.safeguardCounter = 0
    
    safeguardThresholdCoefficient = 3
    if self.safeguardCounter >= self.nodesNumber*safeguardThresholdCoefficient:
      raise StopIteration("Stuck in an attractor")

    return updatingNodeIdx
  
  def __learnStep(self):
    """
    Hebbian learning rule for binary nodes
    """
    i, j = self.learningIndicesIterator.__next__()
    if i != j:
      self.nodeWeights[i][j] += (2*self.nodeValues[0][i] - 1) * (2*self.nodeValues[0][j] - 1)
      self.nodeWeights[j][i] += (2*self.nodeValues[0][i] - 1) * (2*self.nodeValues[0][j] - 1)

      return i, j
  
  def __learningIndicesGenerator(self) -> Iterator:
    for i in range(self.nodesNumber):
      for j in range(i, self.nodesNumber):
        yield i, j
  
  def __runIndexGenerator(self) -> Iterator:
    indices = list(range(self.nodesNumber))
    random.shuffle(indices)
    for i in indices:
      yield i

  def withPattern(self, pattern: np.matrix):
    self.nodeValues = np.array(pattern)
    return self
  
  def withMode(self, mode: NetworkMode):
    """
    LEARN mode will clamp node values and will update exclusively weights (deterministic updating)
    INFER mode will clamp node weigths and will update exclusively value (stochastic updating)
    """
    self.mode = mode
    self.safeguardCounter = 0

    if self.mode == NetworkMode.LEARN:
      self.learningIndicesIterator = self.__learningIndicesGenerator()
    
    return self

  def isStuckInAttractor(self) -> bool:
    """
    Whenever energy is the same for 2 consecutive iterations the
    network may have reached a potential energy well (local minimum)
    """
    currentEnergy: float = self.computeEnergy()
    if self.prevEnergy == currentEnergy:
      return True
    
    self.prevEnergy = currentEnergy
    return False

  def checkPlausibility(self) -> None:
    """
    Weights matrix must respect the constraints described above (see constructor method)
    """
    assert(
      all(np.diag(self.nodeWeights) == 0)
    )
    
    assert(
      la.issymmetric(self.nodeWeights)
    )
  
  def computeEnergy(self) -> float:
    triuSelector = np.triu(np.ones((self.nodesNumber, self.nodesNumber)))
    partialProducts = self.nodeValues * self.nodeValues.T * triuSelector
    weightedPartialProducts = self.nodeWeights * partialProducts
    # The second addend is ignored since every threshold is zero
    # 1/2 coefficent is ignored thanks to the upper triangular selector
    # and symmetry of weights matrix
    return - np.sum(weightedPartialProducts)
    
  def computeInputValues(self) -> np.ndarray:
    return self.nodeValues @ self.nodeWeights
  
  def computeInputValue(self, nodeIdx: int) -> float:
    return self.nodeValues @ self.nodeWeights[nodeIdx]