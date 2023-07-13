from manim import *
from graphics.linear_nn import *
from lib.hopfield import *

class DemoNN(Scene):
  def construct(self):
    self.hopfieldNet = HopfieldNet(9)
    self.nn = LinearNN(self.hopfieldNet, 3, 3, 2.3)

    titleTxt = Text("Toy model demo").to_edge(UP)
    learningTxt = Text("Learning phase", font_size=30).next_to(titleTxt, DOWN)
    inferringTxt = Text("Inferring phase", font_size=30).next_to(titleTxt, DOWN)

    self.g = self.nn.getGraph().to_corner(DR)
    self.el = self.nn.getEdgeLabels()

    self.play(Write(titleTxt))
    self.add(self.g)
    self.add(*self.el)
    self.wait()

    vanNet = Text("Vanilla network before learning any pattern", font_size=30).to_edge(LEFT)
    self.play(Write(vanNet))
    self.wait()
    
    code = Code(
      code = """
      N: int = 9

      hopfieldNetwork = HopfieldNetwork(neuronsNumber = 9)
        .withMode(LEARN)
        .withPattern(X_PATTERN)
      
      A: numpy.ndarray = hopfieldNetwork.nodeValues
      W: numpy.matrix = hopfieldNetwork.nodeWeights
      
      for i in range(N):
        for j in range(i+1, N):
          W[i][j] += (2 * A[i] - 1) * (2 * A[j] - 1)
          W[j][i] += (2 * A[i] - 1) * (2 * A[j] - 1)
      """,
      language = "Python",
      background = "window",
      tab_width = 4,
      background_stroke_color = WHITE,
      background_stroke_width = 1,
      insert_line_no = True
    )
    self.play(Unwrite(vanNet))
    self.play(
      Create(code.scale(0.6).to_edge(LEFT)),
      Create(learningTxt)
    )
    self.wait()

    self.hopfieldNet.withMode(NetworkMode.LEARN).withPattern(
      [[
        1, 0, 1,
        0, 1, 0,
        1, 0, 1
      ]]
    )
    self.updateGraph()
    self.wait()

    for _ in self.hopfieldNet:  # Learning
      self.updateEdgeLabels()
      self.wait()

    self.play(FadeOut(*self.el))    
    self.wait()

    self.play(Uncreate(code))
    code = Code(
      code = """
      hopfieldNetwork
        .withMode(INFER)
        .withPattern(RANDOM_PATTERN)
      
      SAFEGUARD_INTERATIONS: int = N * 3
      
      for itCounter in range(SAFEGUARD_ITERATIONS):
        updatingNodeIdx = numpy.random.randint(N)
        A[updatingNodeIdx] = numpy.heaviside(
        \tA @ W[updatingNodeIdx],  # Dot product
        \t0  # Zero-threshold
        )
      """,
      language = "Python",
      background = "window",
      tab_width = 4,
      background_stroke_color = WHITE,
      background_stroke_width = 1,
      insert_line_no = True
    )
    self.play(
      Create(code.scale(0.6).to_edge(LEFT)),
      Transform(learningTxt, inferringTxt)
    )
    self.wait()

    self.hopfieldNet.withMode(NetworkMode.INFER).withPattern(
      [[
        1, 0, 0,
        0, 0, 0,
        0, 0, 1
      ]]
    )
    for idx in self.hopfieldNet:  # Inferring
      self.play(Flash(self.g.vertices[f"a{idx}"], flash_radius = 0.6))
      self.updateGraph()
      self.wait()
  
  def updateGraph(self):
    self.remove(self.g)
    self.g = self.nn.getGraph().to_corner(DR)
    self.add(self.g)
  
  def updateEdgeLabels(self):
    self.remove(*self.el)
    self.el = self.nn.getEdgeLabels()
    self.add(*self.el)