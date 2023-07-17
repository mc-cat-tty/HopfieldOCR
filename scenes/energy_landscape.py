from manim import *
from graphics.linear_nn import *
from lib.hopfield import *
from nn_scene import *
from scipy.interpolate import *

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
    self.title = Text("Energy Landscape Visualization").to_edge(UP)
    self.subtitle = Text("Learning pattern 1", font_size = 30).next_to(self.title, DOWN)
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

    self.surface = Surface(
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
          ) \
          .set_z_index(0)
        )
    self.hopfieldNet.withPattern(prevPattern)

    self.add_fixed_in_frame_mobjects(self.title)
    hammingDistCaption = Text(
        """
        On the XY plane a toroidal surface is represented as a series of
        network configurations with a hamming distance of 1
        """, font_size = 15
      )
    self.add_fixed_in_frame_mobjects(
      hammingDistCaption.next_to(axes, DOWN).shift(UP*0.7),
      self.subtitle
    )
    self.play(
      FadeIn(axes),
      FadeIn(z_label),
      FadeIn(*graphs),
      Write(self.title),
      Write(self.subtitle),
      Write(hammingDistCaption)
    )
    self.wait(0.5)

    #First pattern
    self.move_camera(phi=50 * DEGREES, theta=-90 * DEGREES, zoom=0.8, run_time=1.5)
    self.begin_ambient_camera_rotation(90*DEGREES/16, about='theta')
    self.begin_ambient_camera_rotation(90*DEGREES/60, about='phi')

    self.surface = Surface(
      func = lambda x, y: axes.c2p(*paramSurface(x, y, self.hopfieldNet)),
      u_range = (-2, 2),
      v_range = (-2, 2),
      resolution = (4, 4)
    ) \
    .set_fill_by_value(axes = axes, colorscale=[(RED_C, -0.5), (YELLOW_C, 0), (GREEN_C, 0.5)], axis=2)

    footerTxt = Text("Learning a pattern is the process of modelling the energy landscape of the network", font_size = 30).to_edge(DOWN).shift(DOWN*0.4)
    self.add_fixed_in_frame_mobjects(self.g.to_edge(RIGHT))
    self.add_fixed_in_frame_mobjects(footerTxt)
    self.play(
      FadeIn(self.surface),
      Write(footerTxt),
      Transform(self.subtitle, Text("Learning pattern 2", font_size = 30).next_to(self.title, DOWN))
    )
    self.wait(0.5)


    for _ in self.hopfieldNet \
      .withMode(NetworkMode.LEARN) \
      .withPattern([[
        1, 1,
        1, 1
      ]]):  # Learning
      self.remove(self.surface, axes)
      
      axes = ThreeDAxes(
        x_range = [-2.5, 2.5],
        x_length = 7,
        y_range = [-2.5, 2.5],
        y_length = 7,
        z_range = [-5, 5],
        z_length = 5
      )
      
      self.surface = Surface(
        func = lambda x, y: axes.c2p(*paramSurface(x, y, self.hopfieldNet)),
        u_range = (-2, 2),
        v_range = (-2, 2),
        resolution = (4, 4)
      ) \
      .set_fill_by_value(axes = axes, colorscale=[(RED_C, -0.5), (YELLOW_C, 0), (GREEN_C, 0.5)], axis=2)
      
      self.remove(self.g)
      self.g = self.nn.getGraph().to_edge(RIGHT)
      self.add_fixed_in_frame_mobjects(self.g)
      self.add(self.surface, axes)

      self.wait(2)
    
    self.wait(5)

    # Second pattern
    self.stop_ambient_camera_rotation(about='phi')
    self.stop_ambient_camera_rotation(about='theta')
    self.move_camera(phi=50 * DEGREES, theta=-90 * DEGREES, zoom=0.8, run_time=1.5)
    self.begin_ambient_camera_rotation(90*DEGREES/16, about='theta')
    self.begin_ambient_camera_rotation(90*DEGREES/120, about='phi')
  
    self.remove(self.g, self.surface)
    self.hopfieldNet = HopfieldNet(4)
    self.nn = LinearNN(self.hopfieldNet, 2, 2, 1.8, 0.8, True)
    self.g = self.nn.getGraph().to_edge(RIGHT)
    self.add_fixed_in_frame_mobjects(self.g)

    self.surface = Surface(
      func = lambda x, y: axes.c2p(*paramSurface(x, y, self.hopfieldNet)),
      u_range = (-2, 2),
      v_range = (-2, 2),
      resolution = (4, 4)
    ) \
    .set_fill_by_value(axes = axes, colorscale=[(RED_C, -0.5), (YELLOW_C, 0), (GREEN_C, 0.5)], axis=2)
    self.play(FadeIn(self.surface))
    self.wait(0.5)
    
    for _ in self.hopfieldNet \
      .withMode(NetworkMode.LEARN) \
      .withPattern([[
        1, 0,
        0, 1
      ]]):  # Learning
      self.remove(self.surface, axes)
      
      axes = ThreeDAxes(
        x_range = [-2.5, 2.5],
        x_length = 7,
        y_range = [-2.5, 2.5],
        y_length = 7,
        z_range = [-5, 5],
        z_length = 5
      )
      
      self.surface = Surface(
        func = lambda x, y: axes.c2p(*paramSurface(x, y, self.hopfieldNet)),
        u_range = (-2, 2),
        v_range = (-2, 2),
        resolution = (4, 4)
      ) \
      .set_fill_by_value(axes = axes, colorscale=[(RED_C, -0.5), (YELLOW_C, 0), (GREEN_C, 0.5)], axis=2)
      
      self.remove(self.g)
      self.g = self.nn.getGraph().to_edge(RIGHT)
      self.add_fixed_in_frame_mobjects(self.g)
      self.add(self.surface, axes)

      self.wait(2)
    
    self.wait(5)


    self.rollBall1()


  def rollBall1(self):
    self.play(Transform(self.subtitle, Text("Free running", font_size = 30).next_to(self.title, DOWN)))
    self.wait()

    sPoints = self.surface.get_all_points()
    f = interp2d(sPoints[:, 0], sPoints[:, 1], sPoints[:, 2], kind = 'cubic')
    
    ballRadius = 0.3
    ballPath = FunctionGraph(
      lambda x: f(x-3, 3)[0],
      x_range = (0, 3)
    ) \
      .rotate(90 * DEGREES, axis = X_AXIS, about_point = (0, 0, 0)) \
      .rotate(90 * DEGREES, axis = Z_AXIS, about_point = (0, 0, 0)) \
      .rotate(90 * DEGREES, axis = Z_AXIS, about_point = (0, 3, 0)) \
      .rotate(360 * DEGREES, axis = Z_AXIS) \
      .shift((0, -ballRadius/2, ballRadius/2)) \
      .set_shade_in_3d(True) \
      .set_z_index(100)
    
    ball = Sphere(radius = ballRadius, color = BLUE).move_to(ballPath[0]).set_z_index(100)
    
    def rollUpdater(obj: Sphere):
      obj.rotate(-3*DEGREES, axis = Y_AXIS)

    ball.add_updater(rollUpdater)
    self.stop_ambient_camera_rotation(about = 'theta')
    self.stop_ambient_camera_rotation(about = 'phi')
    self.begin_ambient_camera_rotation(90*DEGREES/105, about='theta')
    self.begin_ambient_camera_rotation(90*DEGREES/600, about='phi')
    self.play(
      MoveAlongPath(ball, ballPath),
      run_time = 10,
      rate_function = smooth
    )
    self.wait(10)


