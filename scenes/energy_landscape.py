from manim import *
from graphics.linear_nn import *
from lib.hopfield import *
from nn_scene import *

# Toroidal topology
CARTESIAN_CHECKERBOARD_MAPPING = {  # (x, y)
  (-2, 0): 9,
  (-1, 0): 8,
  (0, 0): 0,
  (1, 0): 1,
  (2, 0): 9,

  (-2, 1): 13,
  (-1, 1): 12,
  (0, 1): 4,
  (1, 1): 5,
  (2, 1): 13,

  (-2, 2): 15,
  (-1, 2): 14,
  (0, 2): 6,
  (1, 2): 7,
  (2, 2): 15,

  (-2, -1): 11,
  (-1, -1): 10,
  (0, -1): 2,
  (1, -1): 3,
  (2, -1): 11,

  (-2, -2): 15,
  (-1, -2): 14,
  (0, -2): 6,
  (1, -2): 7,
  (2, -2): 15
}

def checkerboardFromInt(n: int):
  checkerboardList = list(map(
    int,
    list(np.binary_repr(n).zfill(4))
  ))
  return np.array(checkerboardList).reshape((1, len(checkerboardList)))

def paramSurface(x, y, net: HopfieldNet):
  prevPattern = net.nodeValues
  checkerboardVal = CARTESIAN_CHECKERBOARD_MAPPING[(np.around(x), np.around(y))]
  checkerboard = checkerboardFromInt(checkerboardVal)
  energy = net.withPattern(checkerboard).computeEnergy()
  print(checkerboard, energy)
  net.withPattern(prevPattern)
  return np.array([
    x,
    y,
    energy
  ])

class EnergyLandscape(ThreeDScene):
  def construct(self):
    title = Text("Energy Landscape Visualization").to_edge(UP)
    self.hopfieldNet = HopfieldNet(4)
    self.nn = LinearNN(self.hopfieldNet, 2, 2, 1.8, 0.8, True)
    self.g = self.nn.getGraph()

    axes = ThreeDAxes(
      x_range = [-2.5, 2.5],
      x_length = 7,
      y_range = [-2.5, 2.5],
      y_length = 7,
      z_range = [-5, 5],
      z_length = 5
    )

    surface = Surface(
      func = lambda x, y: axes.c2p(*paramSurface(x, y, self.hopfieldNet)),
      u_range = (-2, 2),
      v_range = (-2, 2),
      resolution = (4, 4)
    ).set_fill_by_value(axes = axes, colorscale=[(RED_C, -0.5), (YELLOW_C, 0), (GREEN_C, 0.5)], axis=2)
    z_label = axes.get_z_axis_label(Tex("Potential energy"))

    self.set_camera_orientation(zoom = 0.4)

    graphs = []
    prevPattern = self.hopfieldNet.nodeValues
    for coord, val in CARTESIAN_CHECKERBOARD_MAPPING.items():
      self.hopfieldNet.withPattern(checkerboardFromInt(val))
      graphs.append(
        self.nn
          .getGraph() \
          .scale(0.13) \
          .set_opacity(0.5) \
          .shift(
            RIGHT * 1.35 * coord[0] +
            UP * 1.35 * coord[1]
          )
        )
    self.hopfieldNet.withPattern(prevPattern)

    self.add_fixed_in_frame_mobjects(title)
    hammingDistCaption = Text(
        """
        On the XY plane a toroidal surface is represented as a series of
        network configurations with a hamming distance of 1
        """, font_size = 15
      )
    self.add_fixed_in_frame_mobjects(hammingDistCaption.next_to(axes, DOWN).shift(UP*0.7))
    self.play(
      FadeIn(axes),
      FadeIn(z_label),
      FadeIn(*graphs),
      Write(title),
      Write(hammingDistCaption)
    )
    # self.wait(0.5)

    # First pattern
    self.move_camera(phi=50 * DEGREES, theta=-90 * DEGREES, zoom=0.8, run_time=1.5)
    self.begin_ambient_camera_rotation(90*DEGREES/5, about='theta')
    self.begin_ambient_camera_rotation(90*DEGREES/80, about='phi')
  
    
    footerTxt = Text("Learning a pattern is the process of modelling the energy landscape of the network", font_size = 30).to_edge(DOWN).shift(DOWN*0.4)
    self.add_fixed_in_frame_mobjects(self.g.to_edge(RIGHT))
    self.add_fixed_in_frame_mobjects(footerTxt)
    self.play(
      FadeIn(surface),
      Write(footerTxt)
    )
    # self.wait(0.5)

    for _ in self.hopfieldNet \
      .withMode(NetworkMode.LEARN) \
      .withPattern([[
        1, 0,
        0, 1
      ]]):  # Learning
      self.remove(surface, axes)
      
      axes = ThreeDAxes(
        x_range = [-2.5, 2.5],
        x_length = 7,
        y_range = [-2.5, 2.5],
        y_length = 7,
        z_range = [-5, 5],
        z_length = 5
      )
      
      surface = Surface(
        func = lambda x, y: axes.c2p(*paramSurface(x, y, self.hopfieldNet)),
        u_range = (-2, 2),
        v_range = (-2, 2),
        resolution = (4, 4)
      ) \
      .set_fill_by_value(axes = axes, colorscale=[(RED_C, -0.5), (YELLOW_C, 0), (GREEN_C, 0.5)], axis=2)
      
      self.remove(self.g)
      self.g = self.nn.getGraph().to_edge(RIGHT)
      self.add_fixed_in_frame_mobjects(self.g)
      self.add(surface, axes)

      # self.wait(2)
    
    # self.wait(5)

    ball_path = FunctionGraph(lambda x: 2, x_range = (0, 2)).rotate(90*DEGREES)
    ball = Sphere(radius = 0.3, color = BLUE).move_to(ball_path[0])
    self.play(GrowFromCenter(ball))
    self.play(
      MoveAlongPath(ball, ball_path),
      run_time = 3,
      rate_function = linear
    )
    self.wait(5)
    
    # Second pattern
    self.move_camera(phi=50 * DEGREES, theta=-90 * DEGREES, zoom=0.8, run_time=1.5)
    self.begin_ambient_camera_rotation(90*DEGREES/5, about='theta')
    self.begin_ambient_camera_rotation(90*DEGREES/80, about='phi')
  
    self.remove(self.g, surface)
    self.hopfieldNet = HopfieldNet(4)
    self.nn = LinearNN(self.hopfieldNet, 2, 2, 2, 0.8, True)
    self.g = self.nn.getGraph().to_edge(RIGHT)
    self.add_fixed_in_frame_mobjects(self.g)
    surface = Surface(
      func = lambda x, y: axes.c2p(*paramSurface(x, y, self.hopfieldNet)),
      u_range = (-2, 2),
      v_range = (-2, 2),
      resolution = (4, 4)
    ) \
    .set_fill_by_value(axes = axes, colorscale=[(RED_C, -0.5), (YELLOW_C, 0), (GREEN_C, 0.5)], axis=2)
    self.play(FadeIn(surface))
    self.wait(0.5)

    for _ in self.hopfieldNet \
      .withMode(NetworkMode.LEARN) \
      .withPattern([[
        1, 1,
        1, 1
      ]]):  # Learning
      self.remove(surface, axes)
      
      axes = ThreeDAxes(
        x_range = [-2.5, 2.5],
        x_length = 7,
        y_range = [-2.5, 2.5],
        y_length = 7,
        z_range = [-5, 5],
        z_length = 5
      )
      
      surface = Surface(
        func = lambda x, y: axes.c2p(*paramSurface(x, y, self.hopfieldNet)),
        u_range = (-2, 2),
        v_range = (-2, 2),
        resolution = (4, 4)
      ) \
      .set_fill_by_value(axes = axes, colorscale=[(RED_C, -0.5), (YELLOW_C, 0), (GREEN_C, 0.5)], axis=2)
      
      self.remove(self.g)
      self.g = self.nn.getGraph().to_edge(RIGHT)
      self.add_fixed_in_frame_mobjects(self.g)
      self.add(surface, axes)

      self.wait(2)
    
    self.wait(5)



