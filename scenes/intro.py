from manim import *

class Intro(ZoomedScene):
  def construct(self):
    title = Text("Hopfield Network Energy Landscape")
    subtitle = Text(
      "Visualization of Hopfield network's potential energy landscape and attractor wells",
      t2c = {
        "Visualization": BLUE,
        "potential energy": BLUE,
        "attractor wells": BLUE
      }
    )

    self.play(Write(title))
    self.wait()

    self.play(
      Transform(title, subtitle),
      self.camera.frame.animate.scale(1.7))
    self.wait(4)

    self.play(Uncreate(title))
    self.wait()