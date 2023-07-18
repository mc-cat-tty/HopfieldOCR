from manim import *

class Infra1(Scene):
  def construct(self):
    Text.set_default(font_size = 22)
    r1 = Text(
      """
      A Hopfield network is a trivial kind of NN whose main application is pattern reconstruction.
      """,
      t2c = {
        "pattern reconstruction": GREEN
      }
    ).shift(UP)
    r2 = Text(
      """
      Like any other network it has to been submitted to a learning phase.
      """
    ).next_to(r1, DOWN, buff=0.2)
    r3 = Text(
      """
      Two updating strategies are generally available: synchronous and asynchronous updating;
      in this example, an asynchronous neuron updating has been chosen.
      """,
      t2c = {
        "asynchronous neuron updating": GREEN
      }
    ).next_to(r2, DOWN, buff=0.2)
    self.play(LaggedStart(*[Write(r) for r in (r1, r2, r3)], lag_ratio = 0.3))
    self.wait(13)
    self.play(*[FadeOut(r) for r in (r1, r2, r3)])
    self.wait(.5)