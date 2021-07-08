from raybox3 import RayObject, RayPlane, RayScene, RayCircleMaskScene, RayPointSource, RayCircleSource, RayPlaneSource
from raybox3 import RayService
import numpy as np

"""
Experiment 2: "Testing the rotation functionality".
We can use Circle source, consisted of 12 cosine sources, located on the circle
with the specified radius (RayCircleSource) or the only one cosine source.
The result is the same. In some range of the distances 
between the source and substrate we get that situation. But sometimes it makes sence
to use exactly the circle source so this functionality exists in the project.
"""


x = np.array((-50, 50, 50, -50)) + 20.25
y = np.array((-50, -50, 50, 50)) + 17.3
z = 100
normal = (0, 0, -1)

source_origin = (30.5, 2.5, 0)
source_normal = (0, 0, 1)

plane = RayPlane(x=x, y=y, normal=normal, depth=z, power=6)
#source = RayCircleSource(origin=source_origin, normal=source_normal, radius=22, filled=False)
source = RayPointSource(origin=source_origin, normal=source_normal)
scene = RayScene()
scene.push_object(plane)
scene.push_source(source)

v = (0, 1, 0)
point = (0, 0, 222)

#plane.rotate(V=v, point=point, angle=np.pi/2)

if __name__ == '__main__':
    scene.calculate_on_rotation(V=v, point=point, steps=72)
    #scene.calculate()
    plane.normalize_colors()
    plane.plot(contours=40)
    plane.plot_on_line(axis=1, line=17.3)
    plane.plot_on_line(axis=0, line=20.25)
    #plane.save_on_line(axis=0, line=17.3, filename='second_oX.txt', normalize=1)
    #plane.save_on_line(axis=1, line=20.25, filename='second_oY.txt', normalize=1)
    # plane.write_to_wolfram(name='second_experiment.txt')
    # Checked: OK

