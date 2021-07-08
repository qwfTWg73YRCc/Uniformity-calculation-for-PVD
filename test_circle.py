from raybox3 import RayScene, RayPointSource, RayCircle

"""
This test is failed: the circle substrate cannot be calculated...
But it seems that this functionality is not needed
"""


x = (-200, 200, 200, -200)
y = (-200, -200, 200, 200)
z = 0
normal = (0, 0, 1)
origin = (0, 140, 100)

circle = RayCircle(center=(0, 0, 0), normal=(0, 0, -1), r=300, Rn=10, Tn=10)
source = RayPointSource(origin=origin, normal=normal)
scene = RayScene()
scene.push_object(circle)
scene.push_source(source)

if __name__ == '__main__':
    scene.calculate()
    # scene.calculate_on_rotation(V=normal, point=(0,0,0), steps=256, multiproc=1)
    circle.plot()