from enum import Enum
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
    self.nodeValues: np.ndarray = np.zeros((nodesNumber, 1))
    self.prevEnergy: float = 0.0
    self.nodesNumber: int = nodesNumber
    self.mode = NetworkMode.INFER

  def __iter__(self):
    return self
  
  def __next__(self):
    self.checkPlausibility()

    if self.isStuckInAttractor():
      raise StopIteration("Stuck in an attractor")

    return

  def withPattern(self, pattern: np.matrix):
    self.nodeValues = np.array(pattern)
    return self
  
  def withMode(self, mode: NetworkMode):
    """
    LEARN mode will clamp node values and will update exclusively weights (deterministic updating)
    INFER mode will clamp node weigths and will update exclusively value (stochastic updating)
    """
    self.mode = mode
    return self

  def isStuckInAttractor(self) -> bool:
    """
    Whenever energy is the same for 2 consecutive iterations
    the network reached a potential energy well (local minimum)
    """
    currentEnergy: float = self.computeEnergy()
    if self.prevEnergy == currentEnergy:
      return True
    
    self.prevEnergy = currentEnergy
    return False

  def checkPlausibility(self) -> None:
    """
    Weights matrix must respect the constraints described above
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
    return self.nodeValues.T @ self.nodeWeights
  
  def computeInputValue(self, nodeIdx: int) -> float:
    return self.nodeValues.T @ self.nodeWeights[nodeIdx]