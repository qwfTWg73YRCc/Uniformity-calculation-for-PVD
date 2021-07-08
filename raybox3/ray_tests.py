from raybox3 import RayObject
from raybox3 import RayService
from raybox3 import ObjectDimensionError
import numpy as np
import unittest


class RayBoxTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_init(self):
        #One of dimension miss
        x = (1, 0, 0, 1)
        y = (1, 1, 1, 1)
        z = (3, 3, 3, 3)
        self.assertRaises(ObjectDimensionError, RayObject, x, y)
        self.assertRaises(ObjectDimensionError, RayObject, x, y, z[:-1])
        self.assertRaises(ValueError, RayObject, 'Foo', 'Lol', z)

    def test_service(self):
        vec = (1, 1, -1)
        point = (2, 1, -3)
        rot_m = RayService.rotation_matrix(vec, point, np.pi/6)
        right_result = np.array(((0.9106836, 0.33333333, 0.24401694, 0.57735027),
                                 (-0.24401694, 0.9106836, -0.33333333, -0.42264973),
                                 (-0.33333333, 0.24401694, 0.9106836, 0.15470054),
                                 (0, 0, 0, 1)))
        point_to_rotate = (1, 1, 1, 1)
        rotated = RayService.rotate(rot_m, point_to_rotate)
        right_rotated = (2.06538414, -0.0893164, 0.97606774)
        self.assertTrue(np.allclose(rot_m, right_result))
        self.assertTrue(np.allclose(rotated, right_rotated))


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RayBoxTest)
    unittest.TextTestRunner().run(suite)