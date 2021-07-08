from raybox3 import RayObject, RayPlane, RayScene, RayPointSource, RayCircle, RayCircleSource, RayCircleMaskScene
from raybox3 import RayService
import numpy as np

x = (-200, 200, 200, -200)
y = (-200, -200, 200, 200)
z = 0
normal = (0, 0, 1)
source_normal = (0, 0, -1)
origin = (0, 120, 100)

plane = RayPlane(x=x, y=y, normal=normal, depth=z, power=5)
#plane.rotate(V=(1,0,0),point=(0,0,0),angle=-np.pi/20)
source31 = RayPointSource(origin=(-50, 175, 95), normal=source_normal)
source32 = RayPointSource(origin=(50, 175, 95), normal=source_normal)
source = RayPointSource(origin=(0, 175, 90), normal=source_normal)
source11 = RayPointSource(origin=(-70, 160, 95), normal=source_normal)
source12 = RayPointSource(origin=(70, 160, 95), normal=source_normal)
source2 = RayPointSource(origin=(0, 80, 100), normal=source_normal)
source21 = RayPointSource(origin=(30, 100, 95), normal=source_normal)
source22 = RayPointSource(origin=(-30, 100, 95), normal=source_normal)
source3 = RayPointSource(origin=(50, 120, 95), normal=source_normal)
source4 = RayPointSource(origin=(-50, 120, 95), normal=source_normal)
source5 = RayPointSource(origin=(30, 140, 95), normal=source_normal)
source6 = RayPointSource(origin=(-30, 140, 95), normal=source_normal)

scene = RayCircleMaskScene(ray_object=plane, ray_sources=[source, source2, source3, source4, source5, source6, source21, source22, source31, source32], origin=(0,140,120))

scene.calculate_on_rotation(V = normal,point=(0,0,0),steps=40)

mask = scene.calculate_symmetric_mask(
    V=normal,
    point=(0, 0, 120),
    points=150,
    depth=15,
    filter_angle=np.pi,
    cut_angle=1.2*np.pi,
    start_point=0,
    filename='mask2.dxf')