from raybox3 import RayObject, RayPlane, RayScene, RayPointSource, RayCircleSource
import numpy as np

"""
Third experiment: "Comparing point source and circle source"
As we can see here circle source gives another uniformity distribution
(it is a bit better than in the case of using point one).

If you`d like to increase the uniformity rotating the plane around its` axis
you can use the "rotate" method.
"""
x = np.array((-30, -30, 30, 30))*4.5-200
y = np.array((-24, 24, 24, -24))*3
z = 100
normal = (0, 0, 1)
v = (0, 0, 1)  # counterclockwise

origin = (-100, 0, 0)
source_normal = (0, 0, 1)
plane = RayPlane(x=x, y=y, normal=normal, depth=z, power=6)
# plane.rotate()

source = RayPointSource(origin=origin, normal=source_normal)
#source = RayCircleSource(origin=origin, normal=source_normal, radius=85, filled=False)

scene = RayScene()
scene.push_object(plane)
scene.push_source(source)
if __name__ == '__main__':
    # scene.calculate()
    scene.calculate_on_narrow_rotation(alpha = 2*np.pi, V=normal, point=(0,0,0), steps=64)
    plane.normalize_colors()
    plane.plot(contours=40)





