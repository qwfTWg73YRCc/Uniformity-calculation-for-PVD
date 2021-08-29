from raybox3 import  RayPlane, RayScene, RayPlaneSource
import numpy as np

"""
This experiment shows, how you can create your own form of the radiation (light/deposition)
sources by adding the coordinates of new point sources to the existing ones... For this you have to
make the plane source instance. "Source_x" are parallel to the x line, analogically to "source_y".
So this is mostly useful to create rectangular sources.
"""


x = (-250, 250, 250, -250)
y = (-400, -400, 400, 400)
z = 100
normal = (0, 0, 1)

origin = (0, 0, 400)
source_normal = (0, 0, -1)

plane = RayPlane(x=x, y=y, normal=normal, depth=z, power=5)

source_x = np.concatenate((np.linspace(-32.5, 32.5, 5), [32.5]*20, np.linspace(32.5, -32.5, 5), [-32.5] * 20), axis=0)
source_y = np.concatenate(([425]*5, np.linspace(425, -425, 20), [-425]*5, np.linspace(-425, 425, 20)), axis=0)
source = RayPlaneSource(origin=origin, x=source_x, y=source_y, normal=source_normal, filled=False)

scene = RayScene()
scene.push_object(plane)
scene.push_source(source)

if __name__ == '__main__':
    scene.calculate(multiproc=0)
    #scene.calculate_on_rotation(V=(0, 1, 0), point=(0, 0, 0), steps=70, multiproc=0)
    plane.normalize_colors()
    plane.plot()
    plane.plot_on_line(axis=0, line=0, normalize=1)
