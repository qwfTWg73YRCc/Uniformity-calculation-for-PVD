import numpy as np

from matplotlib.tri import Triangulation, UniformTriRefiner, LinearTriInterpolator

# import test_mask
from .core import RayCalculator, Task, default_calc_function
from .ray_exceptions import ObjectDimensionError
from .ray_service import RayService, timeit_io
from dxfwrite import DXFEngine as dxf
import meshpy.triangle as triangle

AXIS_LIST = {'Ox': 0, 'Oy': 1, 'Oz': 2}
EPS = 1e-15


class RayObject:
    def __init__(self, x=None, y=None, z=None, nx=None, ny=None, nz=None, mesh=None, triangulation=None, *argc):
        if x is None or y is None or z is None:
            raise ObjectDimensionError('Object must be 3 dimension size')
        #vec_1 = np.array((x[1]-x[0],
        #                  y[1]-y[0],
        #                  z[1]-z[0]))
        #vec_2 = np.array((x[2]-x[1],
        #                  y[2]-y[1],
        #                  z[2]-z[1]))
        #i=0
        #normal_1 = np.cross(vec_1,vec_2)
        #check = np.cross(normal_1, np.array((nx[0], ny[0], nz[0])))
        #if check[0]!=0 or check[1]!=0 or check[2]!=0:
        #    raise ValueError('Normal vector is wrong')
        self.id = id(self)
        self._initial_x = np.array(x, dtype='f')
        self._initial_y = np.array(y, dtype='f')
        self._initial_z = np.array(z, dtype='f')
        self._x = np.array(x, dtype='f')
        self._y = np.array(y, dtype='f')
        self._z = np.array(z, dtype='f')
        self._nx = np.array(nx, dtype='f')
        self._ny = np.array(ny, dtype='f')
        self._nz = np.array(nz, dtype='f')
        self._mesh = np.array(mesh, dtype='int')
        self._triangulation = triangulation
        self._tasks = []
        self._colors = np.zeros(len(self._x))
        self._validate()

    def _validate(self):
        lengths = map(len, (self._x, self._y, self._z))
        if len(set(lengths)) > 1:
            raise ObjectDimensionError('Dimension vectors of object must have equal size')

    def __repr__(self):
        return 'Object with id = {id}'.format(id=self.id)

    @property
    def normal(self):
        """

        :return: _nx, _ny, _nz 
        """
        return self._nx, self._ny, self._nz

    @property
    def coords3(self):
        """

        :return: x, y, z 
        """
        return self._x, self._y, self._z

    @property
    def coords4(self):
        """

        :return: x, y, z, colors 
        """
        return self._x, self._y, self._z, self._colors

    @property
    def initial_coords4(self):
        """

        :return: x, y, z, colors 
        """
        return self._initial_x, self._initial_y, self._initial_z, self._colors

    @property
    def coords5(self):
        """

        :return: x, y, z, colors 
        """
        return self._x, self._y, self._z, self._mesh, self._colors

    @property
    def matrix3(self):
        """

        :return: 3 dimensional matrix of object 
        """
        return np.vstack((self._x, self._y, self._z))

    @property
    def matrix4(self):
        """
        Used in 3-d rotation (rotate() method)
        :return: 4 dimensional matrix of object, where last col is 1
        x1 y1 z1 1
        x2 y2 z2 1
        ..........
        xn yn zn 1
        """
        t = np.array([1] * len(self._x))
        t.reshape((len(self._x), 1))
        return np.vstack((self._x, self._y, self._z, t))

    @property
    def get_limits(self):
        xmin = np.min(self._initial_x)
        xmax = np.max(self._initial_x)
        ymin = np.min(self._initial_y)
        ymax = np.max(self._initial_y)
        zmin = np.min(self._initial_z)
        zmax = np.max(self._initial_z)
        return (xmin, xmax, ymin, ymax, zmin, zmax)

    @property
    def get_middles(self):
        xmid = self._x[self._mesh].mean(axis=1)
        ymid = self._y[self._mesh].mean(axis=1)
        zmid = self._z[self._mesh].mean(axis=1)
        color_mid = self._colors[self._mesh].mean(axis=1)
        return xmid, ymid, zmid, color_mid

    def move(self, point):
        # Base for the future development - this method will be added
        # to make the movement more complicated
        point_x, point_y, point_z = point
        self._x += point_x
        self._y += point_y
        self._z += point_z

    def rotate(self, V, point, angle):
        rot_m = RayService.rotation_matrix(V, point, angle)
        coords = RayService.rotate(self.matrix4, rot_m)
        self._x, self._y, self._z = coords

    def clear_colors(self):
        self._colors = np.array([0] * self._colors.size, dtype='f')

    def normalize_colors(self):
        self._colors = self._colors / np.max(self._colors)

    def linearize_colors(self, points=None):
        x, y, z, colors = self.initial_coords4
        triang = Triangulation(x, y)
        interp_lin = LinearTriInterpolator(triang, colors)
        if points is None:
            xmin, xmax, ymin, ymax, _, _ = self.get_limits
            steps = int(xmax - xmin) + 1
            stepX = (xmax - xmin) / steps
            stepY = (ymax - ymin) / steps
            lX, lY = np.meshgrid(np.linspace(xmin, xmax, steps), np.linspace(ymin, ymax, steps))
        else:
            lX, lY = points
        linearized_colors = interp_lin(lX, lY)
        return lX, lY, linearized_colors

    def plot(self, contours=10, middles=True):
        x, y, colors = self.linearize_colors()
        RayService.plot_surface(x, y, colors, contours)

    def plot_mesh(self):
        x, y, _, triangles, colors = self.coords5
        RayService.plot_triangles(x, y, triangles, colors)

    def write_to_wolfram(self, name):
        with open(name, 'w') as f:
            f.write('{')
            obj = ['{{{0},{1},{2}}}'.format(x, y, color) for x, y, z, color in zip(*self.coords4)]
            f.write(','.join(obj))
            f.write('}')

    def sum_colors(self, result_colors):
        if result_colors.shape != self._colors.shape:
            result_colors = np.sum(result_colors, axis=0)
        self._colors += result_colors

    def append_task(self, task_id):
        self._tasks.append(task_id)


class RayPointSource:
    def __init__(self, origin=None, normal=None, power=1, *argc):
        if origin is None or normal is None:
            raise ObjectDimensionError('Object must be 3 dimension size')
        self._origin = np.array(origin, dtype='f')
        self._normal = np.array(normal, dtype='f')
        self._power = power


    def calculate_rays(self, ray_object=None):
        x, y, z = ray_object.coords3
        nx, ny, nz = self._normal
        ox, oy, oz = self._origin
        ray_x = x - ox
        ray_y = y - oy
        ray_z = z - oz
        normals = RayService.cos_btw(ray_x, ray_y, ray_z, nx, ny, nz)
        return ray_x, ray_y, ray_z, normals

    @property
    def origin(self):
        return self._origin

    @property
    def normal(self):
        return self._normal

    @property
    def power(self):
        return self._power


class RayCircleSource(RayPointSource):
    def __init__(self, origin=None, normal=None, radius=1, points=12, filled=True):
        super().__init__(origin=origin, normal=normal)
        self.radius = radius
        self.points = points
        self.filled = filled
        x, y, z = self.origin
        if self.filled:
            # SunFlower algorithm
            points = np.arange(0, self.radius ** 2, 1)
            golden_ratio = (np.sqrt(5) + 1) / 2
            theta = 2 * np.pi * points / golden_ratio ** 2
            points_x = x + np.sqrt(points) * np.cos(theta)
            points_y = y + np.sqrt(points) * np.sin(theta)
            origins_z = np.array([self.origin[2]] * len(points))
            self.origins = np.column_stack((points_x, points_y, origins_z))
            RayService.plot_points(points_x, points_y)
        else:
            angles = np.linspace(0, 2 * np.pi, self.points)[:-1]
            origins_z = np.array([z] * len(angles))
            self.origins = np.column_stack(
                (x + np.cos(angles) * self.radius, y + np.sin(angles) * self.radius, origins_z))
            RayService.plot_points(self.origins[:, 0], self.origins[:, 1])
            pass

    def calculate_rays(self, ray_object=None):
        ray_x, ray_y, ray_z, normal = [], [], [], []
        for origin in self.origins:
            rays_x, rays_y, rays_z, normals = RayPointSource(normal=self.normal, origin=origin).calculate_rays(
                ray_object)
            ray_x.append(rays_x)
            ray_y.append(rays_y)
            ray_z.append(rays_z)
            normal.append(normals)
        return np.array(ray_x), np.array(ray_y), np.array(ray_z), np.array(normal)


class RayPlaneSource(RayPointSource):
    def __init__(self, origin=None, x=None, y=None, normal=None, power=1, filled=True):
        super().__init__(origin=origin, normal=normal)
        self.filled = filled
        ox, oy, oz = self.origin
        if self.filled:
            triang = Triangulation(x + ox, y + oy)
            refiner = UniformTriRefiner(triang)
            my_tri2, index = refiner.refine_triangulation(subdiv=power, return_tri_index=True)
            # After refining
            points_x = my_tri2.x
            points_y = my_tri2.y
            points_z = np.array([oz] * len(points_x))
            self.origins = np.column_stack((points_x, points_y, points_z))
            RayService.plot_points(points_x, points_y)
        else:
            points_x = x
            points_y = y
            points_z = np.array([oz] * len(x))
            self.origins = np.column_stack((points_x, points_y, points_z))
            RayService.plot_points(points_x, points_y)
            pass

    def calculate_rays(self, ray_object=None):
        rays = (RayPointSource(normal=self.normal, origin=origin).calculate_rays(ray_object) for origin in self.origins)
        rays = np.stack(tuple(rays), axis=1)
        return rays


class RayPlane(RayObject):
    def __init__(self, x, y, normal, power=2, depth=0):
        nx, ny, nz = normal
        triang = Triangulation(x, y)
        refiner = UniformTriRefiner(triang)
        my_tri2, index = refiner.refine_triangulation(subdiv=power, return_tri_index=True)
        # After refining
        x = my_tri2.x
        y = my_tri2.y
        z = np.array([depth] * len(x))
        triangles = my_tri2.triangles
        nx, ny, nz = [nx] * len(triangles), [ny] * len(triangles), [nz] * len(triangles)
        super().__init__(x=x, y=y, z=z, nx=nx, ny=ny, nz=nz, mesh=triangles, triangulation=triang)

    def get_colors_on_line(self, axis=0, line=0, normalize=0):
        xmin, xmax, ymin, ymax, _, _ = self.get_limits
        steps = 50
        if (axis == 0 and (line>xmax or line<xmin)) or (axis == 1 and (line>ymax or line<ymin)):
            raise ValueError('Line out of range!')
        if not (axis):
            x = np.linspace(xmin, xmax, steps)
            y = np.array([line] * steps)
            x, _, y = self.linearize_colors(points=(x, y))
            # indexes = np.where(np.abs(self._x) < line + EPS)
            # x = self._y[indexes]
            # y = self._colors[indexes]
            if normalize: y = y / np.max(y)
        else:
            y = np.linspace(ymin, ymax, steps)
            x = np.array([line] * steps)
            _, x, y = self.linearize_colors(points=(x, y))
            if normalize: y = y / np.max(y)
        return x, y

    def plot_on_line(self, axis=0, line=0, normalize=0, axes_names=('x','y')):
        x, y = self.get_colors_on_line(axis, line, normalize)
        RayService.plot_lines(x, y, axes_names)

    """
    def save_on_line(self, axis=0, line=0, normalize=0, filename='out.txt'):
        x, y = self.get_colors_on_line(axis, line, normalize)
        RayService.save_to_wolfram(x, y, filename=filename)
    """

class RayMask(RayObject):
    def __init__(self, x, y, normal, power=2, depth=0):
        """
        :param x: coords in the basic Descartes system
        :param y: coords in the basic Descartes system
        :param normal: -||-
        :param power: discretization rate
        :param depth: height
        """
        nx, ny, nz = normal
        info = triangle.MeshInfo()
        points = np.column_stack((np.array(x, dtype='d'), np.array(y, dtype='d')))[::-1]
        RayService.plot_points_lines(*points.T,axes_names=('Mask_x','Mask_y'))
        facets = RayService.round_trip_connect(0, len(points) - 1)  # connection between sequenced indexes
        # (could be used in future development)
        info.set_points(points)
        info.set_facets(facets)
        mesh = triangle.build(info)
        mesh_points = np.array(mesh.points)
        triangles = np.array(mesh.elements)
        # After refining
        x = mesh_points[:, 0]
        y = mesh_points[:, 1]
        z = np.array([depth] * len(x))
        super().__init__(x=x, y=y, z=z, nx=nx, ny=ny, nz=nz,
                         mesh=triangles, triangulation=mesh)

    def calculate_intersection(self, source_origin, raysX, raysY, raysZ):
        points = np.column_stack((self._x, self._y, self._z))[self._mesh]
        pointsA, pointsB, pointsC = points[:, 0], points[:, 1], points[:, 2]
        non_intersected_indexes = RayService.intersect_muller(pointsA, pointsB,
                                                              pointsC, source_origin,
                                                              raysX, raysY,
                                                              raysZ)
        # Intersected rays
        return non_intersected_indexes


class RayCircle(RayObject):
    def __init__(self, center=None, normal=None, r=1, Rn=36, Tn=36):
        if center is None or normal is None:
            raise ObjectDimensionError('Missing center point or normal vector')
        nx, ny, nz = normal
        x0, y0, z0 = center
        self.R, self.T = np.linspace(0, r, Rn), np.linspace(0, 2 * np.pi, Tn)
        r, t = np.meshgrid(self.R, self.T)
        self.Rn = Rn
        self.Tn = Tn
        x = np.ravel(x0 + r * np.cos(t))
        y = np.ravel(y0 + r * np.sin(t))
        z = np.ravel(np.array([z0] * x.size).reshape(x.shape))
        mesh = np.zeros(x.size)
        nx, ny, nz = [nx] * len(r), [ny] * len(r), [nz] * len(r)
        super().__init__(x=x, y=y, z=z, nx=nx, ny=ny, nz=nz, mesh=mesh)


class RayScene:
    def __init__(self, object_list=None, sources_list=None):
        self._object_list = object_list if object_list is not None else []
        self._sources_list = sources_list if sources_list is not None else []
        self._calculator = RayCalculator()

    def push_object(self, object):
        self._object_list.append(object)

    def push_source(self, *sources):
        for source in sources:
            self._sources_list.append(source)

    def plot_sources(self, *sources):
        pass

    @timeit_io
    def calculate_on_rotation(self, V, point, steps, multiproc=0):
        angle = 2 * np.pi / steps
        for source in self._sources_list:
            for obj in self._object_list:
                for i in range(0, steps, 1):
                    self.calculate()
                    obj.rotate(V, point, angle)
                    print('Step done, {0} %'.format(i/steps*100))
            print('Source done')

    @timeit_io
    def calculate_on_narrow_rotation(self, V, point, steps,alpha=np.pi*2, multiproc=0):
        delta_alpha = alpha / steps
        for source in self._sources_list:
            for obj in self._object_list:
                for i in range(0, steps, 1):
                    self.calculate_2()
                    obj.rotate(V, point, delta_alpha)
                    print('Step done, {0} %'.format(i / steps * 100))
            print('Source done')
    '''
    def calculate_on_narrow_rotation_for_circle_source(self, alpha, V,
                                                       point, steps, multiproc=0):
        """
        Use this method when using RayCircleSource        
        :return: 
        """
        delta_alpha = alpha / steps
        for source in self._sources_list:
            for obj in self._object_list:
                for i in range(0, steps, 1):
                    self.calculate_circle()
                    obj.rotate(V, point, delta_alpha)
                    print('Step done, {0} %'.format(i / steps * 100))
            print('Source done')    
    '''

    @timeit_io
    def calculate(self, multiproc=0):
        for source in self._sources_list:
            for obj in self._object_list:
                ray_x, ray_y, ray_z, normals = source.calculate_rays(obj)
                visible_indexes = None
                colors = default_calc_function(ray_x, ray_y, ray_z, normals, visible_indexes, power=source.power)
                obj.sum_colors(colors)


    def calculate_2(self, multiproc=0):
        for source in self._sources_list:
            for obj in self._object_list:
                ray_x, ray_y, ray_z, normals = source.calculate_rays(obj)
                visible_indexes = None
                if self._mask is not None:
                    visible_rays = np.array([False] * ray_x.size)
                    visible_indexes = self._mask.calculate_intersection(source.origin, ray_x, ray_y, ray_z)
                    visible_rays[visible_indexes==False] = True
                colors = default_calc_function(ray_x, ray_y, ray_z, normals, visible_indexes, power=source.power)
                obj.sum_colors(colors)

    '''
    def calculate_circle(self, multiproc=0):
        for origin in self._origins:
            for obj in self._object_list:
                ray_x, ray_y, ray_z, normals = source.calculate_rays(obj)
                visible_indexes = None
                if self._mask is not None:
                    visible_rays = np.array([False] * ray_x.size)
                    visible_indexes = self._mask.calculate_intersection(source.origin, ray_x, ray_y, ray_z)
                    visible_rays[visible_indexes==False] = True
                colors = DEFAULT_CALC_FUNCTION(ray_x, ray_y, ray_z, normals, visible_indexes, power=source.power)
                obj.sum_colors(colors)
    '''

def check_sum(list_, index, value):
    size = list_.size
    # k - roll param
    k = size // 2 - index
    if k < 0:
        k = 3 * size // 2 - index
    middle = size // 2
    list_ = np.roll(list_, k)
    for i in range(1, (size // 2)):
        if middle - i >= 0 and middle + i <= size:
            a = middle - i
            b = middle + i
            s = np.sum(list_[a:b])
            if s >= value:
                s0 = s - value
                sa = np.sum(list_[a:a + 1])
                sb = np.sum(list_[b - 1:b])
                modf = s0 / (sa + sb)
                return a, b, k, s0, modf
    return a, b, k, s - value, 0


def save_as_dxf(X, Y, name='object.dxf'):
    drawing = dxf.drawing(name)
    drawing.add_layer(name.split('.')[0], color=2)
    polyline = dxf.polyline(linetype='DOT')
    polyline.add_vertices(np.column_stack((X, Y)))
    drawing.add(polyline)
    drawing.save()


class RayCircleMaskScene(RayScene):
    def __init__(self, origin=None, ray_object=None, ray_sources=None):
        if ray_object is None or ray_sources is None:
            raise ValueError('Can\'t initialize object without object or source, check args')
        self._origin = origin if origin is not None else np.array((0, 0, 0))
        self._mask = None
        super().__init__(object_list=[ray_object], sources_list=ray_sources)

    '''
    def calculate_mask(self, V, point, nrm=0.5, radius_points=100, angle_points=100, depth=10):
        # this method doesn`t work properly
        self.calculate()
        ray_object = self._object_list[0]
        rd = max(ray_object.get_limits[:4])
        radiuses = np.linspace(0, rd, radius_points)
        angles = np.linspace(0, 2 * np.pi, angle_points)
        points_angles, points_r = np.meshgrid(angles, radiuses)
        points_x = points_r * np.cos(points_angles)
        points_y = points_r * np.sin(points_angles)
        _, _, colors = ray_object.linearize_colors((points_x, points_y))
        # colors = [ colors on r=0 ,  colors on r = 3, ... ]
        # Find line for cut
        min_sum = np.min(np.sum(colors, axis=1))
        max_sum = np.max(np.sum(colors, axis=1))
        colors_sum = (min_sum + max_sum) * nrm
        roll_index = len(colors) - np.argmax(np.sum(colors, axis=0))
        # counts = np.bincount(arg_list)
        # roll_index = arg_list[np.argmax(counts)]
        left_line = []
        right_line = []
        for i, (radial_colors, p_x, p_y) in enumerate(zip(colors[1:], points_x[1:], points_y[1:])):
            path = 0
            not_finded = True
            radial_colors = np.roll(radial_colors, roll_index)
            p_x = np.roll(p_x, roll_index)
            p_y = np.roll(p_y, roll_index)
            index = RayService.find_range(radial_colors, colors_sum)
            if index:
                left_index, right_index = index * (-1), index
                left_point_x, right_point_x = p_x[left_index], p_x[right_index]
                left_point_y, right_point_y = p_y[left_index], p_y[right_index]
                left_line.append((left_point_x, left_point_y))
                right_line.append((right_point_x, right_point_y))
            
            #while not_finded:
                #path += 1
                #masked_sequence = np.roll(radial_colors, path)[0:path * 2 + 1]
                #summed_colors = np.sum(masked_sequence)
                #if summed_colors > colors_sum or masked_sequence.size == radial_colors.size:
                    #not_finded = False
                    #right_border = path * 2 if path * 2 < radial_colors.size - 1 else None
                    # If colors on path is not enough
                    #if right_border is None:
                        #continue
                    #rolled_x = np.roll(p_x, path)
                    #rolled_y = np.roll(p_y, path)
                    #left_point_x, right_point_x = (rolled_x[0], rolled_x[right_border])
                    #left_point_y, right_point_y = (rolled_y[0], rolled_y[right_border])
                    #left_line.append((left_point_x, left_point_y))
                    #right_line.append((right_point_x, right_point_y))
            
        left_line = np.array(left_line)
        right_line = np.array(right_line)
        _, _, z_mask = ray_object.coords3
        origin = self._origin
        projected_left_line = RayService.perspective_projection(left_line.T[0], left_line.T[1], z_mask[0], depth,
                                                                origin)
        projected_right_line = RayService.perspective_projection(right_line.T[0], right_line.T[1], z_mask[0], depth,
                                                                 origin)
        projected_left_line = np.array(projected_left_line)
        projected_right_line = np.array(projected_right_line).T[::-1].T
        projected_x = np.append(np.concatenate((projected_left_line[0], projected_right_line[0])),
                                projected_left_line[0][0])
        projected_y = np.append(np.concatenate((projected_left_line[1], projected_right_line[1])),
                                projected_left_line[1][0])
        projected_line = np.array([projected_x, projected_y])

        RayService.plot_points_lines(projected_line[0], projected_line[1])
        save_as_dxf(projected_line[0], projected_line[1], 'mask_new.dxf')
        self._mask = RayMask(projected_line[0], projected_line[1], normal=ray_object.normal, depth=depth)
        self._mask.plot_mesh()
        self._object_list[0].clear_colors()
        return self._mask
    '''
    '''
    def calculate_symmetric_mask(self, V, point, nrm=0.5, angle=np.pi / 2, cut_angle=np.pi / 8, filter_angle=2*np.pi,
                                 points=100, depth=10, start_point=0, filename='mask.dxf'):
        # this method doesn`t work properly...
        self.calculate_on_rotation(V=V, point=point, steps=64)
        # self.calculate()
        ray_object = self._object_list[0]
        ray_object.normalize_colors()
        rd = max(ray_object.get_limits[:4])
        radiuses = np.linspace(start_point, rd, points)[1:]
        rd_max = np.argmax(radiuses)
        points_x, points_y = radiuses * np.cos(angle), radiuses * np.sin(angle)
        line_x, line_y, colors = ray_object.linearize_colors((points_x, points_y))
        RayService.plot_points(line_y, colors)
        colors_part = colors / np.max(colors)
        cut_sectors = 1 / colors_part
        cut_angles = cut_sectors / np.max(cut_sectors) * cut_angle
        fltr = np.abs(cut_angles) < 2 * np.pi
        left_points_x = line_x - radiuses * np.sin(cut_angles * .5)
        right_points_x = line_x + radiuses * np.sin(cut_angles * .5)
        points_y = line_y - radiuses * (1 - np.cos(cut_angles * .5))

        _, _, z_mask = ray_object.coords3
        origin = self._origin

        projected_left_line = RayService.perspective_projection(left_points_x[fltr][::-1], points_y[fltr][::-1],
                                                                z_mask[0], depth,
                                                                origin)
        projected_right_line = RayService.perspective_projection(right_points_x[fltr], points_y[fltr], z_mask[0], depth,
                                                                 origin)

        projected_x = np.concatenate((projected_left_line[0], projected_right_line[0]))
        projected_y = np.concatenate((projected_left_line[1], projected_right_line[1]))

        save_as_dxf(projected_x, projected_y, filename)
        # Create circle around profile
        circle_angle = np.pi/2-np.arctan(projected_y[-1] / projected_x[-1]) / 2
        circle_angles = np.linspace(circle_angle, 2 * np.pi - circle_angle, points)
        circle_rad = np.sqrt(projected_y[-1] ** 2 + projected_x[-1] ** 2)
        circle_points_x = circle_rad * np.sin(circle_angles)
        circle_points_y = circle_rad * np.cos(circle_angles)
        projected_x = np.concatenate((projected_x, circle_points_x, [projected_x[0]]))
        projected_y = np.concatenate((projected_y, circle_points_y, [projected_y[0]]))
        projected_line = np.array([projected_x, projected_y])

        self._mask = RayMask(projected_line[0][:-1], projected_line[1][:-1], normal=ray_object.normal, depth=depth)
        self._object_list[0].clear_colors()
        return self._mask
    '''
    '''
    def generate_symmetric_mask(self, points):
        pass
    '''

    def calculate(self, multiproc=0):
        for source in self._sources_list:
            for obj in self._object_list:
                ray_x, ray_y, ray_z, normals = source.calculate_rays(obj)
                visible_indexes = None
                if self._mask is not None:
                    visible_rays = np.array([False] * ray_x.size)
                    visible_indexes = self._mask.calculate_intersection(source.origin, ray_x, ray_y, ray_z)
                    visible_rays[visible_indexes==False] = True
                colors = default_calc_function(ray_x, ray_y, ray_z, normals, visible_indexes, power=source.power)
                obj.sum_colors(colors)
