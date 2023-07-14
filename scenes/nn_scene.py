from manim import *

class NNScene(Scene):
  def updateGraph(self):
    self.remove(self.g)
    self.g = self.nn.getGraph().to_corner(DR)
    self.add(self.g)
  
  def updateEdgeLabels(self):
    self.remove(*self.el)
    self.el = self.nn.getEdgeLabels()
    self.add(*self.el)