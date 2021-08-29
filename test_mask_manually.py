from raybox3 import RayObject, RayMask, RayPlane, RayScene, RayCircleMaskScene, RayPointSource, RayCircleSource, RayPlaneSource
from raybox3 import RayService
import numpy as np

"""
You can add a rectangle mask and make a test static calculation.
Trying "const = 150" case will give you a white screen = zero thickness.
Perspective projection gives new projected mask.
This new mask gives the same thickness distribution on the substrate.
"""


# Код для тестирования работы функци расчёта нанесения покрытия с учётом маски

# 1. Стартовое расположение подложки
x = np.array((-30, 30, 30, -30))*10
# y = np.array((-24, -24, 24, 24))-100
y = np.array((-24, -24, 24, 24))*10
z = 200
normal = (0, 0, -1)

# Положение точечного косинусного источника для последующего расчёта - когда маска готова
source_origin = (0, 0, 0)
source_normal = (0, 0, 1)

# Функция задания подложки - инициализация экзмпляра класса RayPlane
plane = RayPlane(x=x, y=y, normal=normal, depth=z, power=5)
# Функция задания источника
source = RayPointSource(origin=source_origin, normal=source_normal)
# Создаю прямоугольник ABCD
#const = 150
# Вариант с полностью закрытой подожкой
const = 10
x_AB = np.linspace(-const, const, 2)
y_AB = np.array([const] * len(x_AB))
x_BC = y_AB
y_BC = np.linspace(const, -const, 2)
x_CD = y_BC
y_CD = np.array([-const]*len(x_CD))
x_DA = -y_AB
y_DA = x_AB
x_half_1 = np.append(x_AB, x_BC)
x_half_2 = np.append(x_CD, x_DA)
x = np.append(x_half_1, x_half_2)  # абсциссы точек прямоугольника

y_half_1 = np.append(y_AB, y_BC)
y_half_2 = np.append(y_CD, y_DA)
y = np.append(y_half_1, y_half_2)  # ординаты точек прямоугольника
# инициализация класса маски: располагаем её прямо в плоскости движения подложки
mask = RayMask(x, y, (0,0,-1), depth=200)  # depth нужно для того, чтобы потом спроецировать маску

scene = RayCircleMaskScene(ray_object=plane, ray_sources=[source])
#scene._mask = mask
projected_x,projected_y = RayService.perspective_projection(x = mask.coords3[0],
                                                       y = mask.coords3[1], z = mask.coords3[2],
                                                       depth=-20,
                                                       source=source_origin)
projected_mask = RayMask(projected_x,projected_y,(0,0,-1),depth=180)
#scene._mask = projected_mask
if __name__ == '__main__':
    scene.calculate()
    plane.normalize_colors()
    plane.plot()



