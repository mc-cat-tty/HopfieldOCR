from manim import *
from graphics.linear_nn import *
from lib.hopfield import *
from nn_scene import *

class DemoNN(NNScene):
  def construct(self):
    self.hopfieldNet = HopfieldNet(9)
    self.nn = LinearNN(self.hopfieldNet, 3, 3, 2.3)

    titleTxt = Text("Toy model demo").to_edge(UP)
    learningTxt = Text("Hebbian learning", font_size=30).next_to(titleTxt, DOWN)
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

      hopfieldNetwork = HopfieldNetwork(neuronsNumber = 9) \\
        .withMode(LEARN) \\
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
      self.updateGraph()
      self.wait()

    self.play(FadeOut(*self.el))    
    self.wait()

    self.play(Uncreate(code))
    code = Code(
      code = """
      hopfieldNetwork \\
        .withMode(INFER) \\
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
    ).scale(0.6).to_edge(LEFT + UP)
    potentialAxes = Axes(
      x_range = [0, 60, 6],
      y_range = [0, -11, 2],
      x_length = 4,
      y_length = 2.5
    ).next_to(code, DOWN).shift(DOWN)
    labels = potentialAxes.get_axis_labels(
      Text("Iter").scale(0.4),
      Text("Energy").scale(0.4)
    )

    self.hopfieldNet.withMode(NetworkMode.INFER).withPattern(
      [[
        1, 0, 0,
        0, 0, 0,
        0, 0, 1
      ]]
    )
    self.updateGraph()

    self.line = None
    self.energyVal = None
    noisyPatternTxt = Text(
      """
      The noisy pattern given as input will be
      led to the nearest attractor.
      Network's potential energy will
      decrease towards the potential well.
      """, font_size=30).next_to(code, DOWN).shift(DOWN)

    self.play(
      Transform(learningTxt, inferringTxt),
    )
    self.play(
      Create(code),
      learningTxt.animate.shift(RIGHT*3.5),
      titleTxt.animate.shift(RIGHT*3.5),
      Write(noisyPatternTxt)
    )
    self.wait(3)

    self.play(Unwrite(noisyPatternTxt))
    self.play(
      Create(potentialAxes),
      Create(labels)
    )
    self.wait()

    inputValue = Text(f"Neuron {0} input value: {10}", font_size=30).next_to(code, DOWN)
    self.add(inputValue)

    E = []
    it = [0]
    for idx in self.hopfieldNet:  # Inferring
      inputVal = int(self.hopfieldNet.computeInputValue(idx)[0])
      self.play(
        Flash(self.g.vertices[f"a{idx}"], flash_radius = 0.6),
        Transform(
          inputValue,
          Text(
            f"Neuron {idx} input value: {'+' if inputVal >= 0 else ''}{inputVal}",
            font_size=30
          ).next_to(code, DOWN)
        )
      )
      it.append(it[-1] + 1)
      E.append(self.hopfieldNet.computeEnergy())
      if self.line is not None:
        self.remove(self.line)
      if self.energyVal is not None:
        self.remove(self.energyVal)
      self.line = potentialAxes.plot_line_graph(it, E, add_vertex_dots=False)
      if len(it) >= 3:
        self.energyVal = Text(f"{E[-1]}", font_size = 15, color = YELLOW)
        self.add(self.energyVal.next_to(
          self.line.get_corner(DR)
        ))
      self.add(self.line)
      self.updateGraph()
      self.wait()
