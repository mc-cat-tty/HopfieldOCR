from manim import *
from graphics.linear_nn import *
from lib.hopfield import *
from nn_scene import *
from PIL import Image, ImageOps
from random import choice

class DemoOcr(NNScene):
  def construct(self):
    self.potentialAxes = None
    self.line = None
    self.labels = None
    self.hopfieldNet = HopfieldNet(196)
    self.nn = LinearNN(self.hopfieldNet, 14, 14, 2.5, 0.19, False)
    self.g = self.nn.getGraph().to_edge(RIGHT)

    titleTxt = Text("Real-world demo").to_edge(UP)
    learningTxt = Text("Learning", font_size = 30).next_to(titleTxt, DOWN)
    inferringTxt = Text("Inferring", font_size = 30).next_to(titleTxt, DOWN)

    self.play(
      Write(titleTxt)
    )
    self.wait()
    self.updateGraph()
    self.wait()
    img = self.fromNumbersToCanvas(2)
    caption = Text("*These patterns are not (too) overlapping", font_size = 20).next_to(img, DOWN)
    self.play(
      FadeIn(img),
      FadeIn(caption),
      Write(learningTxt)
    )
    self.wait()

    LEARNED_PATTERNS = [2, 5, 7]
    for num in LEARNED_PATTERNS:
      self.play(Transform(img, self.fromNumbersToCanvas(num)))
      self.hopfieldNet \
        .withMode(NetworkMode.LEARN) \
        .withPattern(
          self.fromNumbers(num)
        )
      self.updateGraph()
      self.wait()
      [_ for _ in self.hopfieldNet]  # learn
    
    self.play(FadeOut(img), FadeOut(caption))

    self.play(Transform(learningTxt, inferringTxt))
    self.wait()
    for num in LEARNED_PATTERNS:
      self.copeWithNoise(num)

  def halveResolution(self, img: np.ndarray) -> np.ndarray:
    return img[::2, ::2]

  def fromNumbers(self, n: int) -> np.ndarray:
    return np.reshape(
      np.array(Image.open(f"numbers/Number{n}.png").convert('1')),
      (1, 196),
      'F'
    )

  def fromNumbersToCanvas(self, n: int) -> ImageMobject:
    img = ImageMobject(
      np.array(
        Image.open(f"numbers/Number{n}.png")
      )
    )
    img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["box"])
    return img.scale(40).to_edge(LEFT)

  def copeWithNoise(self, n: int):
    imgN = self.fromNumbers(n)
    
    NOISE_RATE_ZERO = 0.30
    noiseZero = np.random.randint(196, size = int(196 * NOISE_RATE_ZERO / 2))
    imgN[0][noiseZero] = 0

    NOISE_RATE_ONE = 0.20
    noiseOne = np.random.randint(196, size = int(196 * NOISE_RATE_ONE / 2))
    imgN[0][noiseOne] = 1

    self.hopfieldNet.withMode(NetworkMode.INFER).withPattern(imgN)
    self.updateGraph()
    self.wait()
    count = 0
    t = []
    E = []
    for _ in self.hopfieldNet:
      count += 1
      t.append(count)
      E.append(self.hopfieldNet.computeEnergy())
    
    if self.potentialAxes is not None:
      self.remove(self.potentialAxes)
      self.remove(self.line)
      self.remove(self.labels)
    
    self.potentialAxes = Axes(
      x_range = [0, t[-1], int(t[-1]/5)],
      y_range = [0, E[-1], int(t[-1]/5)],
      x_length = 5,
      y_length = 6,
      x_axis_config={"include_numbers": True},
      y_axis_config={"include_numbers": True},
      axis_config={"include_numbers": True},
    ).to_edge(LEFT)
    self.labels = self.potentialAxes.get_axis_labels(
      Text("Iteration").scale(0.6),
      Text("Energy").scale(0.6)
    )
    self.line = self.potentialAxes.plot_line_graph(t, E, add_vertex_dots=False)

    self.updateGraph()
    self.add(self.potentialAxes, self.labels)
    self.play(Create(self.potentialAxes))
    self.add(self.potentialAxes, self.line)
    self.wait(2)