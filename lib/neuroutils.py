import numpy as np

def biasedHeaviside(value, threshold):
  return np.heaviside(value - threshold)

def hopfieldStorablePattern(nodesNumber: int) -> int:
  STORING_RATE = 0.14
  return int(nodesNumber * STORING_RATE)
