from lib.hopfield import HopfieldNet, NetworkMode
from lib.neuroutils import pngToPattern

if __name__ == "__main__":
  import numpy as np
  network = HopfieldNet(16).withMode(NetworkMode.LEARN).withPattern(
    np.array(
      [
        [1, 0, 0, 1,
        0, 1, 1, 0,
        0, 1, 1, 0,
        1, 0, 0, 1]
      ]
    )
  )

  for _ in network:
    print(network.computeEnergy())
  
  print("infer")

  for _ in network.withMode(NetworkMode.INFER).withPattern(
    np.array(
      [
        [0, 0, 0, 0,
        0, 0, 1, 0,
        0, 1, 0, 0,
        0, 0, 0, 1]
      ]
    )
  ):
    print(network.nodeValues)
    print(network.computeEnergy())
  