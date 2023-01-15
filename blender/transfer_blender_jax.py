import os
import sys
import time
import subprocess
import tempfile
import shutil
import bpy
import bmesh

# add site-packages from vnenv to blender interpreter
sys.path.append(os.path.join(os.getenv('APPDATA'), "Python", "Python310", "site-packages"))
# sys.path.append(r"C:\dev\3rd-party\jax-dev\venv\Lib\site-packages")
import numpy
import scipy

# avoid GPU when its not available...
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".XX"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax.numpy as jnp

HERE = r"C:\dev\3rd-party\maya-transfer-blend-shape\scripts\transfer_blend_shape"#os.path.dirname(__file__)
PATH_TO_PY_EXECUTABLE_WITH_SCIPY = r"C:\dev\3rd-party\jax-dev\venv\Scripts\python.exe"

# utils
def as_chunks(l, num):
    """
    :param list l:
    :param int num: Size of split
    :return: Split list
    :rtype: list
    """
    chunks = []
    for i in range(0, len(l), num):
        chunks.append(l[i:i + num])
    return chunks


def subprocess_dot(a, b):
    dot_script = os.path.join(HERE, "dot.py")
    with tempfile.TemporaryDirectory() as tempdir:
        np_dir = os.path.join(tempdir, "np")
        script = os.path.join(np_dir, 'script.py')

        os.makedirs(np_dir, exist_ok=True)
        shutil.copy2(dot_script, script)
        numpy.save(os.path.join(np_dir, "a.npy"), a)
        numpy.save(os.path.join(np_dir, "b.npy"), b)
        print(f'running dot subprocess with {script}')

        proc = subprocess.Popen([PATH_TO_PY_EXECUTABLE_WITH_SCIPY, script])
        proc.communicate()

        if proc.returncode != 0:
            tb = open(os.path.join(np_dir, "tb.log")).read()
            print(tb.strip())
            raise Exception("failed running dot in subprocess")

        dot = numpy.load(os.path.join(np_dir, "dot.npy"))
        print('Done dot subprocess')

    return dot


def dot_backend(a, b):
    print("dot backend is jax.jnp")
    return jnp.dot(a, b)

class Transfer(object):
    def __init__(
            self,
            source_mesh=None,
            target_mesh=None,
    ):
        self._source_mesh = None
        self._target_mesh = None
        self._threshold = 0.001
        self._iterations = 3

        self.source_mesh = bpy.data.objects[source_mesh]
        self.target_mesh = bpy.data.objects[target_mesh]

        self.source_points = self.get_mesh_points(self.source_mesh.data)
        self.target_points = self.get_mesh_points(self.target_mesh.data)
        self.source_triangles = self.get_triangles(self.source_mesh.data)

    def get_mesh_points(self, mesh) -> numpy.array:
        return numpy.array([v.co for v in mesh.vertices])

    def filter_vertices(self, points):
        """
        :param numpy.Array points:
        :return: Static/Dynamic vertices
        :rtype: numpy.Array, numpy.Array
        """
        lengths = scipy.linalg.norm(self.source_points - points, axis=1)
        return numpy.nonzero(lengths <= self._threshold)[0], numpy.nonzero(lengths > self._threshold)[0]

    def get_triangles(self, mesh):
        # get mesh triangles
        # https://odederell3d.blog/2019/11/23/blender-python-accessing-mesh-triangles/
        mesh.calc_loop_triangles()
        loop_triangles = mesh.loop_triangles
        triangles = []
        for t in loop_triangles:
            triangles.extend(t.vertices)

        return triangles

    def calculate_target_matrix(self):
        """
        :return: Target matrix
        :rtype: numpy.Array
        """
        triangles = self.source_triangles
        target_points = self.target_points

        matrix = numpy.zeros((len(triangles), target_points.shape[0]))
        for i, (i0, i1, i2) in enumerate(as_chunks(triangles, 3)):
            e0 = target_points[i1] - target_points[i0]
            e1 = target_points[i2] - target_points[i0]
            va = numpy.array([e0, e1]).transpose()

            q, r = numpy.linalg.qr(va)
            inv_rqt = numpy.dot(numpy.linalg.inv(r), q.transpose())

            for j in range(3):
                matrix[i * 3 + j][i0] = - inv_rqt[0][j] - inv_rqt[1][j]
                matrix[i * 3 + j][i1] = inv_rqt[0][j]
                matrix[i * 3 + j][i2] = inv_rqt[1][j]

        return matrix

    def calculate_deformation_gradient(self, points):
        """
        :param numpy.Array points:
        :return: Deformation gradient
        :rtype: numpy.Array
        """
        triangles = self.source_triangles
        source_points = self.source_points

        matrix = numpy.zeros((len(triangles), 3))
        for i, (i0, i1, i2) in enumerate(as_chunks(triangles, 3)):
            va = self.calculate_edge_matrix(source_points[i0], source_points[i1], source_points[i2])
            vb = self.calculate_edge_matrix(points[i0], points[i1], points[i2])

            q, r = numpy.linalg.qr(va)
            inv_rqt = numpy.dot(numpy.linalg.inv(r), q.transpose())

            sa = numpy.dot(vb, inv_rqt)
            sat = sa.transpose()
            matrix[i * 3: i * 3 + 3] = sat

        return matrix

    @staticmethod
    def calculate_edge_matrix(point1, point2, point3):
        """
        :param numpy.Array point1:
        :param numpy.Array point2:
        :param numpy.Array point3:
        :return: Edge matrix
        :rtype: numpy.Array
        """
        e0 = point2 - point1
        e1 = point3 - point1
        e2 = numpy.cross(e0, e1)
        return numpy.array([e0, e1, e2]).transpose()

    def calculate_area(self, points):
        """
        :param numpy.Array points:
        :return: Triangle areas
        :rtype: numpy.Array
        """
        vertex_area = numpy.zeros(shape=(len(points),))
        source_triangles = self.source_triangles
        triangle_points = numpy.take(points, source_triangles, axis=0)
        triangle_points = triangle_points.reshape((len(triangle_points) // 3, 3, 3))

        length = triangle_points - triangle_points[:, [1, 2, 0], :]
        length = scipy.linalg.norm(length, axis=2)

        s = numpy.sum(length, axis=1) / 2.0
        areas = numpy.sqrt(s * (s - length[:, 0]) * (s - length[:, 1]) * (s - length[:, 2]))

        for indices, area in zip(as_chunks(source_triangles, 3), areas):
            for index in indices:
                vertex_area[index] += area

        return vertex_area

    def get_source_area(self):
        return self.calculate_area(self.source_points)

    def get_target_connectivity(self):
        """
        :return: Target connectivity
        :rtype: list[list[int]]
        :raise RuntimeError: When target is not defined.
        """
        if self.target_mesh is None:
            raise RuntimeError("Target mesh has not been defined, unable to query connectivity.")

        from_mesh = self.target_mesh.data
        bm = bmesh.new()  # create an empty BMesh
        bm.from_mesh(from_mesh)
        connectivity = []
        for vert in bm.verts:
            vl = []
            for l in vert.link_edges:
                vl.append(l.other_vert(vert).index)
            connectivity.append(vl)
        bm.free()

        return connectivity

    def calculate_laplacian_matrix(self, weights, ignore):
        """
        Create a laplacian smoothing matrix based on the weights, for the
        smoothing the number of vertices and vertex connectivity is used
        together with the provided weights, the weights are clamped to a
        maximum of 1. Any ignore indices will have their weights set to 0.

        :param numpy.Array weights:
        :param numpy.Array ignore:
        :return: Laplacian smoothing matrix
        :rtype: scipy.sparse.csr.csr_matrix
        """
        num = self.target_points.shape[0]
        connectivity = self.get_target_connectivity()

        weights[ignore] = 0
        data, rows, columns = [], [], []

        for i, weight in enumerate(weights):
            weight = min([weights[i], 1])
            indices = connectivity[i]
            z = len(indices)
            data += ([i] * (z + 1))
            rows += indices + [i]
            columns += ([-weight / float(z)] * z) + [weight]

        return scipy.sparse.coo_matrix((columns, (data, rows)), shape=(num, num)).tocsr()

    def calculate_laplacian_weights(self, points, ignore):
        """
        Calculate the laplacian weights depending on the change in per vertex
        area between the source and target points. The calculated weights are
        smoothed a number of times defined by the iterations, this will even
        out the smooth.

        :param numpy.Array points:
        :param numpy.Array ignore:
        :return: Laplacian weights
        :rtype: numpy.Array
        """
        source_area = self.get_source_area()
        target_area = self.calculate_area(points)
        weights = numpy.dstack((source_area, target_area))
        weights = numpy.max(weights.transpose(), axis=0) / numpy.min(weights.transpose(), axis=0) - 1
        smoothing_matrix = self.calculate_laplacian_matrix(numpy.ones(len(points)), ignore)

        for _ in range(self._iterations):
            diff = numpy.array(smoothing_matrix.dot(weights))
            weights = weights - diff

        return weights.reshape(len(points))

    def execute(self, source_name:str = 'BASE_SOURCE'):
        print(f"started... {source_name}")
        t = time.time()

        from_mesh = bpy.data.objects[source_name]
        from_mesh_points = self.get_mesh_points(from_mesh.data)
        static_vertices, deformed_vertices = self.filter_vertices(from_mesh_points)
        target_matrix = self.calculate_target_matrix()

        # calculate deformation gradient, the static vertices are used to
        # anchor the static vertices in place.
        static_matrix = target_matrix[:, static_vertices]
        static_points = self.target_points[static_vertices, :]
        static_gradient = dot_backend(static_matrix, static_points)
        deformation_gradient = self.calculate_deformation_gradient(from_mesh_points) - static_gradient

        # isolate dynamic vertices and solve their position. As it is quicker
        # to set all points rather than individual ones the entire target
        # point list is constructed.
        deformed_matrix = target_matrix[:, deformed_vertices]
        deformed_matrix_transpose = deformed_matrix.transpose()

        dot = dot_backend(deformed_matrix_transpose, deformed_matrix)
        lu, piv = scipy.linalg.lu_factor(dot)
        uts = dot_backend(deformed_matrix_transpose, deformation_gradient)
        deformed_points = scipy.linalg.lu_solve((lu, piv), uts)
        #
        target_points = self.target_points.copy()
        target_points[deformed_vertices, :] = deformed_points
        #
        # # calculate the laplacian smoothing weights/matrix using the
        # # per-vertex area difference, this will ensure area's with most
        # # highest difference receive the most smoothing, these are applied
        # # to the calculated points
        smoothing_weights = self.calculate_laplacian_weights(from_mesh_points, static_vertices)
        smoothing_matrix = self.calculate_laplacian_matrix(smoothing_weights, static_vertices)
        #
        for _ in range(self._iterations):
            dot = smoothing_matrix.dot(target_points)
            diff = numpy.array(dot)
            target_points = target_points - diff

        target_mesh = self.target_mesh.data
        result_mesh = bpy.data.meshes.new(f"{source_name}_MESH")
        bm = bmesh.new()
        bm.from_mesh(target_mesh)

        for i, v in enumerate(bm.verts):
            v.co = target_points[i]

        # Finish up, write the bmesh back to the mesh
        bm.to_mesh(result_mesh)
        result_object = bpy.data.objects.new(f"{source_name}_TGT", result_mesh)
        bpy.data.scenes[0].collection.objects.link(result_object)
        bm.free()

        print("Transferred '{}' in {:.3f} seconds.".format(source_name, time.time() - t))

t = Transfer('BASE', 'TARGET')
sources_col = bpy.data.collections['sources']
for s in sources_col.objects:
    t.execute(s.name)


#https://github.com/cloudhan/jax-windows-builder#readme
#https://jax.readthedocs.io/en/latest/notebooks/quickstart.html
'''

installing jax on blender
downloaing a wheel from here
https://whls.blob.core.windows.net/unstable/index.html

download https://whls.blob.core.windows.net/unstable/cuda111/jaxlib-0.3.25+cuda11.cudnn82-cp310-cp310-win_amd64.whl

/blender install path/python/bin/python.exe pip install https://whls.blob.core.windows.net/unstable/cuda111/jaxlib-0.3.25+cuda11.cudnn82-cp310-cp310-win_amd64.whl
/blender install path/python/bin/python.exe pip install pip install jax[cuda111] -f https://whls.blob.core.windows.net/unstable/index.html --use-deprecated legacy-resolver

then blender can import jax (after setting the site path properly)

'''