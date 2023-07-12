from lib.hopfield import HopfieldNet, NetworkMode
from lib.neuroutils import pngToPattern

if __name__ == "__main__":
  network = HopfieldNet(4).withMode(NetworkMode.LEARN)

  for _ in network.withMode(NetworkMode.INFER):
    print(network.computeEnergy())