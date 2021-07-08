from raybox3 import RayPlane, RayCircleMaskScene, RayPointSource
import numpy as np

"""
Mask cannot be generated automatically... Unfortunately,
I don`t have a code for the universal algorithm. 
Only some analytical formulas - see "alternative formulas" file.
They can give the close solution but not the ideal result.
"""

x = (-200, 200, 200, -200)
y = (-200, -200, 200, 200)
z = 0
normal = (0, 0, 1)

origin = (0, 0, 120)
source_origin = (0, 140, 114)
source_normal = (0, 0, -1)

plane = RayPlane(x=x, y=y, normal=normal, depth=z, power=5)
source = RayPointSource(origin=source_origin, normal=source_normal)
scene = RayCircleMaskScene(ray_object=plane, ray_sources=[source])

if __name__ == '__main__':
    mask = scene.calculate_symmetric_mask(V=normal, point=(0, 0, 0), points=150, depth=0, filter_angle=np.pi, cut_angle=1.5*56/180*np.pi)
    scene.calculate_on_rotation(V=normal, point=(0, 0, 0), steps=64)
    plane.normalize_colors()
    plane.plot()
    plane.plot_on_line()
