from raybox3 import RayObject, RayPlane, RayScene, RayCircleMaskScene, RayPointSource, RayCircleSource, RayPlaneSource
from raybox3 import RayService
import numpy as np

x = (-200, 200, 200, -200)
y = (-200, -200, 200, 200)
z = 0
normal = (0, 0, 1)

origin = (0, 0, 120)
source_origin = (0, 145, 120)
#source_normal = (0, 0, -1)
source_normal = (0, 0, -1)

plane = RayPlane(x=x, y=y, normal=normal, depth=z, power=5)
source = RayPointSource(origin=source_origin, normal=source_normal)
scene = RayCircleMaskScene(ray_object=plane, ray_sources=[source], origin=source_origin)

'''
scene.calculate_on_rotation(V=normal, point=(0, 0, 0), steps=128)
plane.plot()
plane.normalize_colors()
plane.save_on_line(axis=1, line=0, filename='cuprum.txt')
'''
mask = scene.calculate_symmetric_mask(V=normal, point=(0, 0, 0), points=150, depth=15, filter_angle=np.pi, cut_angle=3/2*np.pi, start_point=0)

scene.calculate()
plane.plot()

scene.calculate_on_rotation(V=normal, point=(0, 0, 0), steps=64)
plane.normalize_colors()
plane.plot()
plane.plot_on_line()

