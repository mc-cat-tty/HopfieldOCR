from manim import *

class Infra1(Scene):
  def construct(self):
    Text.set_default(font_size = 22)
    r1 = Text(
      """
      Hopfield networks behaviour can be analyzed as a dynamical system through their energy function.
      """,
      t2c = {
        "dynamical system": GREEN,
        "energy function": GREEN
      }
    ).shift(UP)
    r2 = Text(
      """
      In the light of this interpretation, learning can be seen as the act of shaping network's energy landscape,
      while inferring can be seen as the act of descending energy hills towards lower energy spots (energy wells),
      namely network's attractors (known patterns).
      """,
      t2c = {
        "shaping": GREEN,
        "energy landscape": GREEN,
        "descending": GREEN,
        "hills": GREEN,
        "energy wells": GREEN,
        "attractors": GREEN
      }
    ).next_to(r1, DOWN, buff=0.2)
    self.play(LaggedStart(*[Write(r) for r in (r1, r2)], lag_ratio = 0.7))
    self.wait(8)
    self.play(*[FadeOut(r) for r in (r1, r2)])
    self.wait(.5)