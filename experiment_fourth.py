from raybox3 import RayObject, RayPlane,\
    RayScene,\
    RayPointSource, RayCircleSource
import numpy as np
"""
Numerical experiment 4: shows how uniformity depends on the rotation around the own axis

Also changing the "power" of the substrate you can see how the result changes.
The more you have triangles on the substrate the more accurate becomes the result. 
(uniformity distribution).
"""


x = (-200, 200, 200, -200)
y = (-200, -200, 200, 200)
z = 0
normal = (0, 0, 1)

origin = (0, 140, 114)
source_normal = (0, 0, -1)

plane = RayPlane(x=x,
                 y=y,
                 normal=normal,
                 depth=z,
                 power=5)
plane.plot_mesh()

source = RayCircleSource(origin=origin,
                         normal=source_normal,
                         radius=21,
                         filled=False)

scene = RayScene()
scene.push_object(plane)
scene.push_source(source)
if __name__ == '__main__':
    # scene.calculate()
    scene.calculate_on_narrow_rotation(alpha = np.pi*2, V=normal, point=(0,0,0), steps=32)
    plane.normalize_colors()
    plane.plot(contours=80)
    plane.write_to_wolfram('fourth_experiment.txt')
    # Checked: OK!
