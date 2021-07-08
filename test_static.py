from raybox3 import RayPlane, RayScene, RayPointSource
import numpy as np

"""
This experiment was created in order to check the ability of
creating a plenty of sources with the "for" loop.
"""


x = (-200, 200, 200, -200)
y = (-200, -200, 200, 200)
z = 0
normal = (0, 0, 1)
origin = (0, 150, 114)
source_normal = (0, 0, -1)

plane = RayPlane(x=x, y=y, normal=normal, depth=z, power=6)
source = RayPointSource(origin=origin, normal=source_normal)

scene = RayScene()
scene.push_object(plane)
scene.push_source(source)

for angle in range(0, 10, 1):
    source = RayPointSource(
        origin=(
            np.cos(np.pi * 2 / 10 * angle)*50,
            140 + np.sin(np.pi * 2 / 10 * angle)*50,
            100
               ),
            normal=source_normal
                           )
    scene.push_source(source)

if __name__ == '__main__':
    scene.calculate()
    #scene.calculate_on_rotation(V=normal, point=(0,0,0), steps=32)
    plane.plot()
    plane.normalize_colors()
    plane.save_on_line(axis=1, line=0, filename='res/magnetron150_Ti_static.txt')
    # Checked: OK!


