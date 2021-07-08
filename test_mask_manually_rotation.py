from raybox3 import RayObject,\
    RayMask, RayPlane, RayScene, RayCircleMaskScene,\
    RayPointSource, RayCircleSource, RayPlaneSource
from raybox3 import RayService
import numpy as np

"""
In this experiment we change the mask to the approximately real and
make a rotation calculation.
"""


# Код для тестирования работы функци расчёта нанесения покрытия с учётом маски

# 1. Стартовое расположение подложки
z_0 = 200
#R_s = 515/2  # расстояние от центра подложкодержателя до центра подложки
R_s = 568.4/2  # расстояние от центра подложкодержателя до центра подложки
#y = np.array((-30, 30, 30, -30))
#x = np.array((-24, -24, 24, 24))
gamma = 55
x = np.array((-24, -24, 24, 24))+R_s*(-1+np.cos(gamma*np.pi/180))
y = np.array((30, -30, -30, 30))+R_s*np.sin(gamma*np.pi/180)
z = z_0
normal = (0, 0, -1)

source_origin = (0, -232.5, 0)
source_normal = (0, 0, 1)

# Функция задания подложки - инициализация экзмпляра класса RayPlane
plane = RayPlane(x=x, y=y, normal=normal, depth=z, power=5)
# Функция задания источника
source = RayPointSource(origin=source_origin, normal=source_normal)

# When you need to use mask and RayCircleSource you have to add multiple
# sources in the loop instead of using RayCircleSource
# (like in the "test_static.py" file)
#

# Задание координат маски
# mask with raw line profile
#x = np.array([0, 257.5, 305.5, 257.50, 0])*(-1)
#y = [0, 53.03, 0.000, -53.03, 0]
# mask with complex profile
x_1 = np.array([-305.5, -304.81, -303.75, -302.30, -300.45, -298.20, -295.56,
                -292.55, -289.14, -285.16, -280.59, -275.54, -270.21, -264.75,
               -259.22, -257.5])
x = np.concatenate((x_1, [0], x_1[::-1]))
y_1 = np.array([0, 5.511, 10.96, 16.33, 21.56, 26.64, 31.53,
               36.20, 40.58, 44.64, 47.61, 49.92, 51.49, 52.53,
               53.01, 53.03])
y = np.concatenate((y_1, [0], -y_1[::-1]))
# mask with very big area
#x = np.array([-R_s, R_s,R_s,-R_s,-R_s])
#y = np.array([-R_s, -R_s,R_s,R_s, -R_s])
# инициализация класса маски: располагаем её прямо в плоскости движения подложки
mask = RayMask(x, y, (0, 0, -1), depth=z_0)  # depth нужно для того, чтобы потом спроецировать маску

"""
facets = RayService.round_trip_connect(0, len(mask.points) - 1)
print(facets)
"""

scene = RayCircleMaskScene(ray_object=plane, ray_sources=[source])
scene._mask = mask
scene._mask.plot_mesh()
if __name__ == '__main__':
    # scene.calculate()
    steps = 100
    scene.calculate_on_narrow_rotation(alpha = 100/180 * np.pi,
                                       V = (0, 0, 1),
                                       point = (0, 0, z_0),
                                       steps = steps)
    plane.normalize_colors()
    plane.plot(contours=40)


