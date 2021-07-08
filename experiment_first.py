import numpy as np

from raybox3 import RayPlane, RayScene, RayPointSource

"""
The essence of the first numerical experiment is to validate the 
workability of the algorithm. So we put a single cosine source and 
the substrate under it.

Result is positive - numerical calculation relates the analytical predictions (See [Wang2018]).

Else you can see, that using easy NumPy calculations 
we can create our own figures: for example, a right polygon,
which can be easily approximated to the circle. 
"""


# Reactangle substrate
#x = (-60, 60, 60, -60)
#y = (-31, -31, 31, 31)
# Right polygon substrate
R = 200
alpha = np.linspace(0, 2 * np.pi, 20)
x = R * np.cos(alpha)
y = R * np.sin(alpha)
z = 0
normal = (0, 0, 1)

source_origin = (-17, 0, 100)
source_normal = (0, 0, -1)

plane = RayPlane(x=x, y=y, normal=normal, depth=z, power=5)
source = RayPointSource(origin=source_origin, normal=source_normal)
scene = RayScene()
scene.push_object(plane)
scene.push_source(source)

if __name__ == '__main__':
    scene.calculate()
    plane.normalize_colors()
    plane.plot()
    plane.plot_on_line(axis=0, line=0)
    # plane.save_on_line(filename='first_experiment.txt') # deleted function
    # Checked: OK
