from manim import *

class Infra2(Scene):
  def construct(self):
    Text.set_default(font_size = 22)
    r1 = Text(
      """
      Hopfield networks are remarkably good at coping with noise.
      """,
      t2c = {
        "coping": GREEN,
        "noise": GREEN
      }
    ).shift(UP)
    r2 = Text(
      """
      The next network will act like an OCR: it will get a distorted pattern as input and
      will bring it to the most similar one it has learned until that moment.
      """,
      t2c = {
        "OCR": GREEN,
        "distorted pattern": GREEN,
        "most similar": GREEN
      }
    ).next_to(r1, DOWN, buff=0.2)
    self.play(LaggedStart(*[Write(r) for r in (r1, r2)], lag_ratio = 0.7))
    self.wait(6)
    self.play(*[FadeOut(r) for r in (r1, r2)])
    self.wait(.5)