import numpy as np
import time
from functools import wraps
import matplotlib.pyplot as plt

'''
def timeit(method):
    """
    Decorator to get the execution time
    :param method:Function
    :return:Function
    """
    @wraps(method)
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, te-ts))
        return result
    return timed
'''


def timeit_io(method):
    """
    Timing decorator (not asynchronous)
    :param method: function
    :return: function
    """
    @wraps(method)
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r (%r, %r) %2.2f sec' % \
              (method.__name__, args, kw, te - ts))
        #return result, round((te-ts) * 1000)
        return result
    return timed

class RayService:
    @staticmethod
    def rotation_matrix(vec, point, angle):
        """
        Multiplying the coordinates` matrix by the rotation matrix gives the new coordinates matrix
        :param vec: defines the rotation plane and the direction of the rotation
        :param point: the rotation center
        :param angle: rotation angle
        :return: new coordinates matrix
        """
        u, v, w = vec / np.linalg.norm(vec)
        a, b, c = point
        rot_m = np.array([[u ** 2 + (v ** 2 + w ** 2) * np.cos(angle),
                           u * v * (1 - np.cos(angle)) - w * np.sin(angle),
                           u * w * (1 - np.cos(angle)) + v * np.sin(angle),
                           (a * (v ** 2 + w ** 2) - u * (b * v + c * w)) * (1 - np.cos(angle)) + (
                               b * w - c * v) * np.sin(angle)],
                          [u * v * (1 - np.cos(angle)) + w * np.sin(angle),
                           v ** 2 + (u ** 2 + w ** 2) * np.cos(angle),
                           v * w * (1 - np.cos(angle)) - u * np.sin(angle),
                           (b * (u ** 2 + w ** 2) - v * (a * u + c * w)) * (1 - np.cos(angle)) + (
                               c * u - a * w) * np.sin(angle)],
                          [u * w * (1 - np.cos(angle)) - v * np.sin(angle),
                           v * w * (1 - np.cos(angle)) + u * np.sin(angle),
                           w ** 2 + (u ** 2 + v ** 2) * np.cos(angle),
                           (c * (u ** 2 + v ** 2) - w * (a * u + b * v)) * (1 - np.cos(angle)) + (
                               a * v - b * u) * np.sin(angle)],
                          [0, 0, 0, 1]
                          ])
        return rot_m

    @staticmethod
    def rotate(matrix4, rot_matrix4, tol=1e-14):
        """
        If changing of coordinates is less than the tolerance parameter
        the coordinate leaves old.
        :param matrix4: coordinates matrix
        :param rot_matrix4: rotation matrix
        :param tol: tolerance parameter
        :return: new substrate coordinates
        """
        tol = tol
        rotated = np.dot(rot_matrix4, matrix4)
        rotated[abs(rotated) < tol] = 0.0
        rotated = np.delete(rotated, (len(rotated) - 1), axis=0)
        return rotated

    @staticmethod
    def cos_btw(uX, uY, uZ, vX, vY, vZ):
        """
        cosine between u and v
        :param uX:
        :param uY:
        :param uZ:
        :param vX:
        :param vY:
        :param vZ:
        :return: cosine
        """
        lengths = np.sqrt(uX ** 2 + uY ** 2 + uZ ** 2) * np.sqrt(vX ** 2 + vY ** 2 + vZ ** 2)
        angle = (uX * vX + uY * vY + uZ * vZ) / lengths
        return angle

    @staticmethod
    def perspective_projection(x, y, z, depth, source=None):
        """
        Useful for calculating mask. When we calculate mask in this "library",
        we enter its` coordinates assuming it is located in the substrate plane.
        It is impossible in the real equipment.
        So we project it to the direction of the source origin.
        :param x: mask x coordinates
        :param y: mask y coordinates
        :param z: mask z coordinates
        :param depth: the distance on the line between the normal of the substrate plane and the source origin
        :param source: the source origin
        :return: projected x,y coordinates
        """
        if source is None:
            raise ValueError('No source given')
        source_x, source_y, source_z = source
        projected_x = (x - source_x) / (z - source_z) * (z + depth) + x
        projected_y = (y - source_y) / (z - source_z) * (z + depth) + y
        return projected_x, projected_y


    @staticmethod
    def plot_lines(x, y, axes_names=('x','y')):
        """
        Plotting function
        :param x:  argument
        :param y: value
        :param axes_names: names under x and y lines
        :return: plotted graph
        """
        x_name, y_name = axes_names
        order = x.argsort()
        x = x[order]
        y = y[order]
        plt.grid(True)
        plt.plot(x, y)
        plt.ylabel(y_name)
        plt.xlabel(x_name)
        plt.show()

    @staticmethod
    def plot_points_lines(x, y, axes_names=('x','y')):
        """
        :param x: argument
        :param y: value
        :param axes_names: names under x and y lines
        :return: point graph
        """
        x_name, y_name = axes_names
        plt.plot(x, y)
        plt.ylabel(y_name)
        plt.xlabel(x_name)
        plt.show()

    @staticmethod
    def plot_triangles(x, y, colors, triangles):
        """
        show mesh
        :param x: 1 arg
        :param y: 2 arg
        :param colors: not used
        :param triangles: mesh points
        :return: 2d mesh graph
        """
        plt.figure()
        plt.gca().set_aspect('equal')
        plt.tripcolor(x, y, triangles, edgecolors='k')
        plt.colorbar()
        plt.show()

    @staticmethod
    def plot_points(x, y, axes_names=('x','y')):
        """
        :param x:
        :param y:
        :param axes_names:
        :return: graph with equal scale
        """
        x_name, y_name = axes_names
        plt.scatter(x, y)
        plt.ylabel(y_name)
        plt.xlabel(x_name)
        plt.axes().set_aspect('equal')
        plt.show()

    @staticmethod
    def plot_surface(x, y, colors, contours=10, axes_names=('x','y')):
        """
        2d plot of 3d surface
        :param x: x coords
        :param y: y coords
        :param colors: absolute value of f(x,y) function
        :param contours: -||-
        :param axes_names: text
        :return:
        """
        x_name, y_name = axes_names
        plt.figure()
        im = plt.contourf(x, y, colors, contours, cmap=plt.cm.plasma)
        plt.ylabel(y_name)
        plt.xlabel(x_name)
        plt.colorbar(im)
        plt.axes().set_aspect('equal')
        plt.show()
    '''
    @staticmethod
    def save_to_wolfram(*args, filename=None):
    """
    use this to save data in the wolfram mathematica readable format
    """
        if filename is not None:
            with open(filename, 'w') as f:
                f.write('{')
                tostring = np.vectorize(lambda x: str(x))
                str_args = np.array([tostring(arg) for arg in args])
                write_arr = str_args.T
                obj = ['{{{0}}}'.format(','.join(coord)) for coord in write_arr]
                f.write(','.join(obj))
                f.write('}')
    '''
    '''
    @staticmethod
    def intrsct(pointsA, pointsB, pointsC, ray, origin):
        """
        Muller-Trumbore algorithm and it doesn`t work correctly
        :param pointsA: 
        :param pointsB: 
        :param pointsC: 
        :param ray: 
        :param origin: 
        :return: 
        """
        v0v1 = pointsB - pointsA
        v0v2 = pointsC - pointsA
        pvec = np.cross(ray, v0v2)
        det = np.einsum('ij,ij->i', v0v1, pvec)

        # Get first filter
        first_filter = np.logical_not(det < 1e-7)
        v0v1 = v0v1[first_filter]
        invDet = 1 / det[first_filter]
        tvec = (-1) * pointsA[first_filter] + origin
        u = np.einsum('ij,ij->i', tvec, pvec[first_filter]) * invDet

        # Get second filter
        second_filter = np.logical_not(np.logical_or(u < 0, u > 1))
        qvec = np.cross(tvec[second_filter], v0v1[second_filter])
        v = np.einsum('ij,ij->i', [ray], qvec) * invDet[second_filter]

        # Get thrid filter
        third_filter = np.logical_not(np.logical_or(v < 0, u[second_filter] + v > 1))
        s = np.sum(third_filter)
        return s
    '''
    '''
    @staticmethod
    def intrsct_alternative(pointsA, pointsB, pointsC, ray, origin):
        # returns:
        #    0.0 if ray does not intersect triangle,
        #    1.0 if it will intersect the triangle,
        #    2.0 if starting point lies in the triangle.
        #   This algorithm works sometimes... And sometimes it stops.
        #   Interpreter doesn`t continue calculating - I don`t know why.
        s =[]
        for i in range(len(pointsA)):
            u = pointsB[i] - pointsA[i]
            v = pointsC[i] - pointsA[i]
            normal = np.cross(u, v)
            b = np.inner(normal, ray - origin)
            a = np.inner(normal, pointsA[i] - origin)
            if (b == 0):
                if a != 0:
                    s.append(0)
                else:
                    rI = 0
            else:
                rI = a / b
            if rI < 0:
                s.append(0)
            w = origin + rI * (ray - origin) - pointsA[i]
            denom = np.inner(u, v) * np.inner(u, v) - \
                np.inner(u, u) * np.inner(v, v)
            si = (np.inner(u, v) * np.inner(w, v) -
              np.inner(v, v) * np.inner(w, u)) / denom

            if (si < 0.0) | (si > 1.0):
                s.append(0)
            ti = (np.inner(u, v) * np.inner(w, u) -
              np.inner(u, u) * np.inner(w, v)) / denom
            if (ti < 0.0) | (si + ti > 1.0):
                s.append(0)
            if rI == 0.0:
                s.append(2)
            s.append(1)
        return s
    '''
    @staticmethod
    def intrsct_alternative_2(pointsA, pointsB, pointsC, ray, origin):
        '''
        Assume plane is z = z_0 & ray doesn't lay on the plane
        M is the intersection point between the plane and the ray.
        :param pointsA:
        :param pointsB:
        :param pointsC:
        :param ray:
        :param origin:
        :return: s = 1 if intersected; s = 0 if non-intersected
        '''
        eps = 10**(-1)*8
        i = 0
        s = 0
        M_z = pointsA[0][2]
        M_x = ray[0]/ray[2] * (M_z - origin[2]) + origin[0]
        M_y = ray[1]/ray[2] * (M_z - origin[2]) + origin[1]
        M = np.array([M_x, M_y, M_z])
        while s < 1:
            A = pointsA[i]
            B = pointsB[i]
            C = pointsC[i]
            MA = M - A
            MB = M - B
            MC = M - C
            # угол AMB
            AMB = np.arccos((np.dot(MA, MB)) /
                            (np.linalg.norm(MA)*np.linalg.norm(MB))
                            )
            # угол BMC
            BMC = np.arccos((np.dot(MC, MB)) /
                            (np.linalg.norm(MC) * np.linalg.norm(MB))
                            )
            # угол CMA
            CMA = np.arccos((np.dot(MC, MA)) /
                            (np.linalg.norm(MC) * np.linalg.norm(MA))
                            )
            if AMB == float("NaN"):
                AMB = 0
            if BMC == float("NaN"):
                BMC = 0
            if CMA == float("NaN"):
                CMA = 0
            sum = AMB+BMC+CMA
            delta = sum - 2 * np.pi
            if abs(delta) < eps:
                s = 1
                break
            else:
                s = 0
                if i == len(pointsA)-1:
                    break
                else:
                    i += 1
                    continue
        return s

    @staticmethod
    def intersect_alternative_3(pointsA, pointsB, pointsC, ray, origin):
        # plane equation: A_0x+B_0y+C_0z+D=0
        # A_0,B_0,C_0 = mask.normal - внести нормаль
        plane_vec_1 = pointsA[0]-pointsB[0]
        plane_vec_2 = pointsA[0]-pointsC[0]
        # учитываем порядок в векторном произведении, чтобы получить нормально "вниз"
        A_0, B_0, C_0 = np.cross(plane_vec_2,plane_vec_1)
        # D = -(A_0*mask.coords3[0][0]+B_0*mask.coords3[0][1]+C_0*mask.coords3[0][2])
        D = -(A_0*pointsA[0][0]+B_0*pointsB[0][1]+C_0*pointsC[0][2])
        const_1 = - ray[0]/ray[2] * origin[2]*A_0+A_0*origin[2] - B_0*ray[1]/ray[2]*origin[2]+B_0*origin[1]+D
        M_z = -const_1/(A_0*ray[0]/ray[2]+B_0*ray[1]/ray[2]+C_0)
        M_x = (M_z - origin[2])*ray[0]/ray[2]+origin[0]
        M_y = (M_z - origin[2])*ray[1]/ray[2]+origin[1]
        M = np.array((M_x,M_y,M_z))
        eps = 10 ** (-1) * 8
        i = 0
        s = 0
        while s < 1:
            A = pointsA[i]
            B = pointsB[i]
            C = pointsC[i]
            MA = M - A
            MB = M - B
            MC = M - C
            # угол AMB
            AMB = np.arccos((np.dot(MA, MB)) /
                            (np.linalg.norm(MA)*np.linalg.norm(MB))
                            )
            # угол BMC
            BMC = np.arccos((np.dot(MC, MB)) /
                            (np.linalg.norm(MC) * np.linalg.norm(MB))
                            )
            # угол CMA
            CMA = np.arccos((np.dot(MC, MA)) /
                            (np.linalg.norm(MC) * np.linalg.norm(MA))
                            )
            if AMB == float("NaN"):
                AMB = 0
            if BMC == float("NaN"):
                BMC = 0
            if CMA == float("NaN"):
                CMA = 0
            angles_sum = AMB+BMC+CMA
            delta = angles_sum - 2 * np.pi
            if abs(delta) < eps:
                s = 1
                break
            else:
                s = 0
                if i == len(pointsA)-1:
                    break
                else:
                    i += 1
                    continue
        return s



    @staticmethod
    def intersect_muller(pointsA, pointsB, pointsC, source_origin, raysX, raysY, raysZ):
        """

        :param pointsA:
        :param pointsB:
        :param pointsC:
        :param source_origin:
        :param raysX:
        :param raysY:
        :param raysZ:
        :return: True/False set, where True means that ray has intersected a triangle
        and False  means that ray hasn`t intersected it
        """
        len = np.sqrt(raysX ** 2 + raysY ** 2 + raysZ ** 2)
        rays = np.column_stack((raysX / len, raysY / len, raysZ / len))
        non_intersected = []
        # Hard MPC
        for ray in rays:
            s = RayService.intersect_alternative_3(pointsA, pointsB, pointsC, ray, source_origin)
            #s = RayService.intrsct_alternative_2(pointsA, pointsB, pointsC, ray, source_origin)
            #s = RayService.intrsct_alternative(pointsA, pointsB, pointsC, ray, source_origin)
            #s = RayService.intrsct(pointsA, pointsB, pointsC, ray, source_origin)
            non_intersected.append(s)
        non_intersected = np.array(non_intersected, dtype=np.bool)
        # t = np.dot(v0v2, qvec) * invDet
        return non_intersected

    @staticmethod
    def round_trip_connect(start, end):
        return [(i, i + 1) for i in range(start, end)] + [(end, start)]

    @staticmethod
    def find_range(array, sum):
        """
        This method is used in calculation of the mask
        :param array:
        :param sum:
        :return:
        """
        indexes = np.arange(len(array) // 2)
        for index in indexes:
            d = np.sum(array[(-1)*index:]) + np.sum(array[:index])
            if np.sum(array[(-1)*index:]) + np.sum(array[:index]) > sum:
                return index
        return 0
