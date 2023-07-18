from manim import *

class Outro(Scene):
  def construct(self):
    Text.set_default(font_size = 22)
    r1 = Text(
      """
      Representing more than 4 nodes graphs on a surface would be topologically infeasable,
      since each square graph has exactly 4 nighbours (1 for each side) on a plane.
      """,
      t2c = {
        "more than 4": GREEN,
        "topologically infeasable": GREEN,
        "4 nighbours": GREEN
      }
    ).shift(UP)
    r2 = Text(
      """
      Here's why in the previous example has been chosen such a small network.
      """,
      t2c = {
        "previous": GREEN,
        "small": GREEN
      }
    ).next_to(r1, DOWN, buff=0.2)
    self.play(LaggedStart(*[Write(r) for r in (r1, r2)], lag_ratio = 0.7))
    self.wait(8)
    self.play(*[FadeOut(r) for r in (r1, r2)])
    self.wait(.5)