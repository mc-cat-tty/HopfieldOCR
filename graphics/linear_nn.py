import re
from manim import *
from lib import hopfield
from typing import List
from random import random

class LinearNN:
  def __init__(self, network: hopfield.HopfieldNet, cols: int, rows: int, scale = 3):
    self.net = network
    self.nodes = [f"a{i}" for i in range(network.nodesNumber)]
    self.edges = [(self.nodes[i], self.nodes[j]) for i in range(network.nodesNumber) for j in range(network.nodesNumber) if i > j]
    self.partitions = [
      [self.nodes[cols*j + i] for i in range(cols)] for j in range(rows)
    ]
    self.scale = scale
  
  def getGraph(self) -> Graph:
    activeNodes = [i for i in range(self.net.nodesNumber) if self.net.nodeValues[0][i] == 1]
    self.graph = Graph(
      self.nodes,
      self.edges,
      partitions = self.partitions,
      layout = 'partite',
      layout_scale = self.scale,
      labels = True,
      vertex_config = {"radius": 0.5} | {
        self.nodes[i]: {"fill_color": BLUE, "radius": 0.5} for i in activeNodes
      },
      edge_config = {
        (self.nodes[i], self.nodes[j]): {
          "stroke_color": GREEN
        } for i in activeNodes for j in activeNodes
      }
    )

    return self.graph

  def getEdgeLabels(self) -> List:
    return [
      Text(str(int(
        self.net.nodeWeights[
          int(re.findall(r'\d+', edge[0])[0])
        ][
          int(re.findall(r'\d+', edge[1])[0])
        ])),
        font_size = 24
      ).move_to(
        self.graph.edges[edge].get_center_of_mass() - self.graph.edges[edge].get_right()/6 + self.graph.edges[edge].get_top()/6 + UP/5 - RIGHT/5
      ) for edge in self.edges
    ]