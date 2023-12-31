import numpy as np

def biasedHeaviside(value, threshold):
  return np.heaviside(value - threshold)

def hopfieldStorablePattern(nodesNumber: int) -> int:
  RETAIN_RATE = 0.14
  return int(nodesNumber * RETAIN_RATE)

def pngToPattern(a, *_):
  ...