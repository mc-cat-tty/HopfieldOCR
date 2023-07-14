from manim import *
from graphics.linear_nn import *
from lib.hopfield import *
from nn_scene import *
from PIL import Image, ImageOps
from random import choice

class DemoOcr(NNScene):
  def construct(self):
    self.hopfieldNet = HopfieldNet(196)
    self.nn = LinearNN(self.hopfieldNet, 14, 14, 2.5, 0.19, False)
    self.g = self.nn.getGraph().to_edge(RIGHT)

    titleTxt = Text("Real-world demo").to_edge(UP)
    self.play(Write(titleTxt))
    self.wait()
    self.updateGraph()
    self.wait()
    img = self.fromNumbersToCanvas(2)
    self.play(FadeIn(img))
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
    
    self.play(FadeOut(img))

    for num in LEARNED_PATTERNS:
      self.copeWithNoiseSequence(num)
    

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

  def copeWithNoiseSequence(self, n: int):
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
    [_ for _ in self.hopfieldNet]
    self.updateGraph()
    self.wait()