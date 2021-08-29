import numpy as np
from multiprocessing import Pool
from .ray_service import timeit_io


def default_calc_function(rays_x, rays_y, rays_z, normal_angles, visible_indexes=None, task_id=None, result_dict=None, power=1):
    """
    Default function of film thickness calculation: 
    delta(color) = cos(a) * cos(b) * cos(n) / r^2 
    :param rays_x: vector with x coordinates of rays to object
    :param rays_y: vector with y coordinates of rays to object
    :param rays_z: vector with z coordinates of rays to object
    :param normal_angles: vector with normal angles to surface for each ray
    :return: calculated colors for rays
    """
    cos_fi2 = rays_z / np.sqrt(rays_y ** 2 + rays_z ** 2)
    cos_teta2 = rays_z / np.sqrt(rays_x ** 2 + rays_z ** 2)
    lens2 = (rays_x ** 2 + rays_y ** 2 + rays_z ** 2)
    #colors = np.abs(cos_fi2 * cos_teta2) * normal_angles / lens2 * 10 ** 4 * 2 * power
    colors = np.abs(cos_fi2 * cos_teta2) * normal_angles / lens2 * 10 ** 4
    colors[colors < 0] = 0
    if task_id is not None and result_dict is not None:
        result_dict[task_id] = colors
    if visible_indexes is not None:
        colors[visible_indexes==True] = 0
    return colors


def DEFAULT_GROUP_CALC(task_group, task_id, result_dict):
    for task in task_group:
        task.calculate(task.id, result_dict)


class RayCalculator:
    def __init__(self, task_list = None):
        self._task_container = task_list if task_list is not None else []

    def add_task(self, task):
        self._task_container.append(task)

    def add_tasks(self, *args):
        for task in args:
            self._task_container.append(task)

    def refresh(self):
        self._task_container = []

    @staticmethod
    def calculate_task(task):
        return task.calculate()

    def calculate_multiproc(self, proc_num=5):
        calculator = Pool(proc_num)
        calculator.map(self.calculate_task, self._task_container)
        return calculator

    @timeit_io
    def calculate(self, multiproc=0):
        if multiproc:
            calculator = Pool.T(multiproc)
            result = calculator.map(self.calculate_task, self._task_container)
            return np.array(result)
        return np.array([result.calculate() for result in self._task_container])


class Task:
    def __init__(self, *argc, calc_function=None):
        self.argc = argc
        if calc_function is None:
            self.calc_function = default_calc_function
        else:
            self.calc_function = calc_function

    def calculate(self):
        return self.calc_function(*self.argc)
