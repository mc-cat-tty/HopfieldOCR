import numpy as np

class HopfieldNet:
  """
  Hopfield network with hebbian learning. Zero-threshold nodes.
  """
  def __init__(self, nodesNumber: int):
    nodeWeights: np.ndarray = np.zeros((nodesNumber, nodesNumber))
    nodeValues: np.ndarray = np.zeros((nodesNumber))

  def runStep(self):
    """
    clamp weights
    random node extraction
    update node's value
    check termination
    """

  def trainStep():
    """
    clamp Values
    random node extraction
    update node's input weights
    check termination
    """
  
  def getEnergy() -> float:
    energy: float = 0
    # TODO