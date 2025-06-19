import collections

import pylab as plt
import numpy as np
import scipy.linalg

import io_off_model

def calc_v_adjacency_matrix(mesh, verbose=False):
  # 计算顶点邻接矩阵
  if 'v_adjacency_matrix' in mesh.keys():
    return
  if verbose:
    print('calc_v_adjacency_matrix')
  n_vertices = mesh['vertices'].shape[0]
  mesh['v_adjacency_matrix'] = np.zeros((n_vertices, n_vertices), dtype=np.bool_)
  for face in mesh['faces']:
    for i, j in zip([0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]):
      mesh['v_adjacency_matrix'][face[i], face[j]] = True
#函数作用：首先，检查输入的 mesh 字典中是否已经存在名为 'v_adjacency_matrix' 的键。如果存在，说明已经计算过顶点邻接矩阵，直接返回，避免重复计算。
# 如果 'v_adjacency_matrix' 不在 mesh 中，那么开始计算。首先，获取网格中顶点的数量 n_vertices。
#然后，创建一个大小为 (n_vertices, n_vertices) 的零矩阵 v_adjacency_matrix，数据类型为布尔类型，用于表示邻接关系。
#遍历网格中的每个面（mesh['faces']），对于每个面的三个顶点，将对应的顶点在 v_adjacency_matrix 中的位置设为 True，表示这两个顶点是邻接的。
#函数效果：最终，mesh['v_adjacency_matrix'] 成为一个布尔矩阵，其中的元素 mesh['v_adjacency_matrix'][i, j] 为 True 表示顶点 i 和 j
# 通过一条边相邻，为 False 表示它们之间没有直接的边相连


def calc_vf_adjacency_matrix(mesh, verbose=False):
  # Usage:# 计算顶点-面邻接矩阵
  #   vf_adjacency_matrix[vertex_index, face_index],
  #   if True, vertex_index is in mesh['faces'][face_index]
  if 'vf_adjacency_matrix' in mesh.keys():
    return
  if verbose:
    print('calc_vf_adjacency_matrix')
  n_vertices = mesh['vertices'].shape[0]
  n_faces = mesh['faces'].shape[0]
  mesh['vf_adjacency_matrix'] = np.zeros((n_vertices, n_faces), dtype=np.bool_)  #在这个矩阵中，每一行对应一个顶点，每一列对应一个面。如果顶点与面相邻，则相应的矩阵元素为 True，否则为 False。
  for f_ind in range(mesh['faces'].shape[0]):
    face = mesh['faces'][f_ind]
    mesh['vf_adjacency_matrix'][face, f_ind] = True

def calc_face_normals(mesh, verbose=False, must_recalc=False):
  # 计算面法向量
  if 'face_normals' in mesh.keys() and must_recalc == False:
    return
  if verbose:
    print('calc_face_normals')
  v0 = mesh['vertices'][mesh['faces'][:, 0]]  #v0为一个形状为 (m, 3) 的数组，其中 m 是面数量，每个数组表示一个面的第一个顶点坐标。
  v1 = mesh['vertices'][mesh['faces'][:, 1]]
  v2 = mesh['vertices'][mesh['faces'][:, 2]]
  tmp = np.cross(v0 - v1, v1 - v2) #叉乘公式得到法向量
  norms = np.linalg.norm(tmp, axis=1, keepdims=True)
  norms[norms == 0] = 1
  mesh['face_normals'] = tmp / norms  #法向量归一化得到单位法向量，这种矩阵运算使得 tmp 直接是一个形状为 (m, 3) 的数组，其中每一行是一个面的法向量。最终，mesh['face_normals'] 成为一个矩阵，其中每一行包含一个面的法向量。这个法向量对于进行表面特征提取、渲染、光照计算等任务非常有用。
  return mesh['face_normals']

def calc_face_plane_parameters(mesh, verbose=False, must_recalc=False):
  # 计算每个面平面的法向量和距离原点的有向距离
  if 'face_plane_parameters' in mesh.keys() and must_recalc == False:
    return
  if verbose:
    print('calc_face_plane_parameters')
  calc_face_normals(mesh, must_recalc=must_recalc)
  normals = mesh['face_normals']
  p0s = mesh['vertices'][mesh['faces'][:, 0]]
  ds = np.sum(-normals * p0s, axis=1)[:, None] #获取法向量数组 normals，以及每个面的第一个顶点在顶点数组中的坐标 p0s，以及每个面到原点的有向距离。
  #axis用来为超过一维的数组定义属性。二维数据拥有两个轴：第0轴沿着行的方向垂直向下，第1轴沿着列的方向水平延申。1表示横轴，方向从左到右；0表示纵轴，方向从上到下。当axis=1时，数组的变化是横向的，体现出列的增加或者减少。反之，当axis=0时，数组的变化是纵向的，体现出行的增加或减少。
  mesh['face_plane_parameters'] = np.hstack((normals, ds))
  #将法向量和距离水平连接，形成平面参数矩阵 mesh['face_plane_parameters']

  # Check
  if 0:
    # All a^2 + b^2 + c^2 ~= 1 :
    norma_ = np.linalg.norm(mesh['face_plane_parameters'][:, :3], axis=1)
    print('Are all normas ~1? ', np.all(np.abs(norma_ - 1) < 1e-6))
    # All points of faces lay in the planes :
    p1s = mesh['vertices'][mesh['faces'][:, 1]] # p1 is taken
    p1s_with1ns = np.hstack((p1s, np.ones((p1s.shape[0], 1))))
    d = np.sum(mesh['face_plane_parameters'] * p1s_with1ns, axis=1)
    print('Are all 2nd points of all faces lay in the plane defined?', np.all(np.abs(d) < 1e-6))
    #检查法向量的模是否接近于1，并验证所有面的第二个顶点是否在其定义的平面上。这些检查可能用于确保计算的几何特性满足某些预期条件。


def calc_triangles_area(mesh, verbose=False):
  # 计算三角面片的面积
  if 'faces_area' in mesh.keys():
    return
  if verbose:
    print('calc_triangles_area')
  all_triangles = mesh['vertices'][mesh['faces']]               # get all triangles, matrix of : [n-faces , 3 , 3]
  diff_each_2 = np.diff(all_triangles, axis=1)                  # get two edges for each triangle
  cross = np.cross(diff_each_2[:, 0], diff_each_2[:, 1])        # the magnitude result of the cross product equals to the "makbilit" area
  mesh['faces_area'] = (np.sum(cross ** 2, axis=1) ** .5) / 2   # get the magnitue for each face and devide it by 2
  #叉乘结果平方再开根号，去除正负号，S=(|ABXAC|)*1/2
  return mesh['faces_area']

def calc_vertices_area(mesh, verbose=False):
  # 计算顶点的面积
  if verbose:
    print('calc_vertices_area')
  if 'faces_area' not in mesh.keys():
    calc_triangles_area(mesh)
  if 'vf_adjacency_matrix' not in mesh.keys():
    calc_vf_adjacency_matrix(mesh)
    mesh['vertices_area'] = np.zeros((mesh['vertices'].shape[0]))
  for i in range(mesh['vertices_area'].shape[0]):
    areas = mesh['faces_area'][mesh['vf_adjacency_matrix'][i]]
    mesh['vertices_area'][i] = np.sum(areas) / 3
  return mesh['vertices_area']
#对于每个顶点，通过顶点所属的邻接面片的面积来计算顶点的面积。遍历每个顶点，获取其邻接面片的面积，然后将这些面积之和除以3得到该顶点的面积。

def add_edges_to_mesh(mesh):
  # 为网格添加边信息
  if 'edges' in mesh.keys():
    return
  edges = {}
  edges2face = {}
  for f_index, f in enumerate(mesh['faces']):
    for e in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
      e_ = (min(e), max(e))
      if e_ not in edges.keys():
        edges2face[e_] = []
        edges[e_] = 0
      edges[e_] += 1
      edges2face[e_].append(f_index)
  mesh['edges'] = np.array(list(edges.keys()))
  n_faces_for_edge = np.array(list(edges.values()))
  mesh['n_boundary_edges'] = np.sum(n_faces_for_edge == 1)  #非封闭边界的数量
  mesh['is_watertight'] = np.all(n_faces_for_edge == 2)  #网格是否封闭
  non_maniford_edges_idxs = np.where(n_faces_for_edge != 2)[0]
  non_maniford_edges = mesh['edges'][non_maniford_edges_idxs]  #所有非连通的边
  mesh['non_maniford_vertices'] = np.unique(non_maniford_edges)  #所有涉及非连通边的顶点
  mesh['faces_graph'] = {f:[] for f in range(mesh['n_faces'])}  #构建一个面片图，即 mesh['faces_graph']，其中每个面片对应一个节点，图中的边表示面片之间的邻接关系。
  for adj_faces in edges2face.values():
    for f in adj_faces:
      mesh['faces_graph'][f] += [f_ for f_ in adj_faces if f_ is not f]
  #mesh字典中包含了关于网格边的详细信息，包括每条边的出现次数、是否构成封闭边界、非连通边的顶点等。这些信息对于网格拓扑结构的分析和处理非常有用。
  faces_graph = mesh['faces_graph']
  n_faces = len(mesh['faces'])
  face_adjacency_matrix = np.zeros((n_faces, n_faces), dtype=int)
  for face_id1 in range(n_faces):
    for neighbor_face_id in faces_graph[face_id1]:
      face_adjacency_matrix[face_id1, neighbor_face_id] = 1
  mesh['face_adjacency_matrix'] = face_adjacency_matrix


#计算面邻接矩阵，
def calc_face_adjacency_matrix(mesh):
  add_edges_to_mesh(mesh)
  faces_graph = mesh['faces_graph']
  n_faces = len(mesh['faces'])
  face_adjacency_matrix = np.zeros((n_faces, n_faces), dtype=int)
  for face_id1 in range(n_faces):
    for neighbor_face_id in faces_graph[face_id1]:
      face_adjacency_matrix[face_id1, neighbor_face_id] = 1
  mesh['face_adjacency_matrix'] = face_adjacency_matrix
  return face_adjacency_matrix


def topological_measures(mesh, verbose=False):
  # 计算拓扑学度量
  V = mesh['vertices'].shape[0]
  F = mesh['faces'].shape[0]
  if 'edges' not in mesh.keys():
    add_edges_to_mesh(mesh)
  E = mesh['edges'].shape[0]  #顶点数 V，面数 F，边数 E。
  chi = V + F - E   #Euler数（欧拉数）
  genus = 1 - chi / 2  #亏格（genus），亏格是一个拓扑不变量，它描述了网格表面的孔的数量。

  if verbose:
    print('Number of faces / vertices / edges : ', F, V, E)
    print('Chi : ', chi)
    print('Genus : ', genus)
    print('Number of boundary edges : ', mesh['n_boundary_edges'])

  return genus, mesh['n_boundary_edges']

def calc_valences(mesh):
  # 计算顶点的度数，即与每个顶点相邻的边的数量。
  if 'valences' in mesh.keys():
    return

  mesh['valences'] = np.bincount(mesh['faces'].flatten())
  #使用 NumPy 的 bincount 函数计算每个顶点的度数。mesh['faces'].flatten() 将面片矩阵转换为一维数组，其中包含了所有顶点的索引。
  # np.bincount 函数统计每个顶点在这个数组中出现的次数，即度数

def calc_v2f_one_ring_matrix(mesh):
  # 计算顶点到面的一环矩阵，这个矩阵用于描述每个顶点与其相邻面之间的关系，即一个顶点邻接的所有面。
  if 'v2f_one_ring_matrix' in mesh.keys():
    return
  mesh['v2f_one_ring_matrix'] = np.zeros((mesh['faces'].shape[0], mesh['vertices'].shape[0])) #索引矩阵其行数为面的数量，列数为顶点的数量。
  for v in range(mesh['vertices'].shape[0]):
    idxs = np.where(mesh['faces'] == v)[0]
    mesh['v2f_one_ring_matrix'][idxs, v] = True  #对于每个顶点，通过遍历所有面片，找到包含该顶点的面片索引，然后将对应位置的值设为 True，表示该顶点邻接于这个面

def calc_interpolation_matrices(mesh, verbose=False):
  # 计算插值矩阵
  if 'interp_matrix_v2f' in mesh.keys():
    return
  if verbose:
    print('calc_interpolation_matrices')
  # Make sure area is calculated and also the adjacency matrix  # 确保已计算面积和邻接矩阵
  calc_triangles_area(mesh)  #三角面片面积
  calc_vertices_area(mesh)   #顶点面积
  calc_v2f_one_ring_matrix(mesh)  #顶点到面的一环矩阵

  # Resape to matrices (for "brodcasting") and multiply to get the results
  # 重塑矩阵（用于“广播”）并相乘以获得结果
  Af = mesh['faces_area'].reshape(1, -1)
  Av = mesh['vertices_area'].reshape(-1, 1)
  invAv = (1 / (3 * Av))
  #从面到顶点的插值矩阵
  mesh['interp_matrix_f2v'] = mesh['v2f_one_ring_matrix'].T * (Af * invAv / 3)
  #从顶点到面的插值矩阵
  mesh['interp_matrix_v2f'] = (1 / Af.T) * mesh['interp_matrix_f2v'].T * Av.T
  #mesh 字典中包含了从面到顶点和从顶点到面的插值矩阵，这些矩阵可以用于在网格上进行插值操作，例如从面到顶点的插值和从顶点到面的插值。


def calc_face_centers(mesh):
  # 计算面的中心点
  n_faces = len(mesh['faces'])
  face_centers = np.zeros((n_faces, n_faces), dtype=int)
  if 'face_centers' in mesh.keys():
    return
  mesh['face_centers'] = mesh['vertices'][mesh['faces']].mean(axis=1)  #axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
  face_centers = mesh['face_centers']
  return face_centers
  #根据面片的顶点索引获取相应的顶点坐标，axis=1 表示在每行上取平均值，返回m*1矩阵。
  #当axis参数为0时，表示沿着纵轴进行计算。
  #当axis参数为1时，表示沿着横轴进行计算。

def bfsdfs(graph, root, max_size=np.inf, bfs_flag=True):
  # 广度优先或深度优先搜索，具体取决于传递给函数的参数 bfs_flag 的值
  res = []
  seen, queue = set([root]), collections.deque([root])
  while queue:
    if bfs_flag:
      vertex = queue.popleft()   # Change to pop for DFS# 改为广度优先搜索
    else:
      vertex = queue.pop()  # Change to pop for DFS# 改为深度优先搜索
    res.append(vertex)
    if len(res) >= max_size:
      break
    for node in graph[vertex]:
      if node not in seen:
        seen.add(node)
        queue.append(node)
  return res  #返回的 res 列表包含搜索到的节点的顺序

def calc_dist_from_plane(mesh, f0_mean, a, f):
  # 计算点到平面的距离
  a = a.reshape((1, 3))  #a为平面法向量
  #使用 np.mean 计算平面顶点的均值，先通过 mesh['faces'][f] 获取平面上的所有顶点的索引，然后使用 mesh['vertices'] 获取这些顶点的坐标。
  x = np.mean(mesh['vertices'][mesh['faces'][f]], axis=0).reshape((3, 1)) - f0_mean #用np.mean计算顶点均值，即平面中心点f0_mean，计算点 x 到平面上的中心点 f0_mean 的向量差。
  dist = np.dot(a, x) / np.linalg.norm(a)
  #使用点到平面的距离公式：distance=a*x/||a||
  return abs(dist[0][0])


def cut_mesh(mesh, f0, a, max_length=np.inf):
  # 切割网格模型
  # mesh: 包含网格信息的字典。
  # f0: 切割的起始面的索引。
  # a: 平面的法向量。
  # max_length: 切割的最大面片数。
  if 'faces_graph' not in mesh.keys():
    add_edges_to_mesh(mesh)

  f0_mean = np.mean(mesh['vertices'][mesh['faces'][f0]], axis=0).reshape((3, 1))
  faces = [f0]  #初始化 faces 列表
  not_allowed = [f0]   #用于记录已经被访问的面

  while len(faces) < max_length:
    f_to_add = -1
    if f0 in mesh['faces_graph'][faces[-1]] and len(faces) > 3: # if we've returned to the same face we've started, its time to finish..
      break   #如果已经回到起始面 f0 且 faces 的长度大于3，则结束循环。
    min_d = np.inf
    for f in mesh['faces_graph'][faces[-1]]:
      d = calc_dist_from_plane(mesh, f0_mean, a, f)
      if d < min_d and f not in not_allowed:
        min_d = d
        f_to_add = f
    if f_to_add == -1:
      break
    print(f_to_add)
    not_allowed += mesh['faces_graph'][faces[-1]]   #如果找到了合适的面，将其加入 faces 中，并将其相邻的面加入 not_allowed
    faces.append(f_to_add)
    #该函数通过在网格模型上进行迭代，沿着给定平面切割，返回切割后的面片索引列表。

  return faces  #返回切割后的面片列表 faces

if __name__ == '__main__':
  from visualization import visualize_mesh

  mesh_fn = r"D:\Pycharm\Py_Projects\SurfaceSimplification-master\meshes\airplane_0359.off"
  #mesh_fn = 'hw2_data/phands.off' # hw2_data/
                                # torus_fat_r2 / cat / sphere_s0 / vase / phands / disk
                                # .off
  mesh = io_off_model.read_off(mesh_fn, verbose=True)

  if 0: # HW2, Ex. 2, part I
    calc_vf_adjacency_matrix(mesh)
  if 0: # HW2, Ex. 4, part I
    t_area = calc_triangles_area(mesh)
    print('Surface area: ', t_area.sum())
    v_area = calc_vertices_area(mesh)
    print('Total vertices area: ', v_area.sum())
  if 0: # HW2, Ex.6, part I
    calc_interpolation_matrices(mesh)
    # Generate some function over the mesh faces
    faces_function = np.zeros((mesh['n_faces']))
    face = 0
    val = 100
    for _ in range(100):
      faces_function[face] = val
      found = False
      for cand_face, cand_vs in enumerate(mesh['faces']):
        if faces_function[cand_face] != 0:
          continue
        if np.union1d(cand_vs, mesh['faces'][face]).size == 4: # 2 vertices are the same
          val += 1
          face = cand_face
          break
    vertices_function = np.dot(mesh['interp_matrix_f2v'], faces_function)

    # Check one way and back
    f_ = np.dot(mesh['interp_matrix_v2f'], vertices_function)
    v_ = np.dot(mesh['interp_matrix_f2v'], f_)
    plt.figure()
    plt.plot(f_ - faces_function)
    plt.figure()
    plt.plot(v_ - vertices_function)
    plt.show()

    # Visualization
    visualize_mesh(mesh, faces_function=faces_function, show_tringles=True)
    visualize_mesh(mesh, vertices_function=vertices_function, show_tringles=True)
  if 0: # HW2, Ex.7, part I
    topological_measures(mesh, verbose=True)
  if 0: # HW2, Ex3, part II
    calc_interpolation_matrices(mesh)
    print('Interpolation matrices size:')
    print('  f2v: ', mesh['interp_matrix_f2v'].shape)
    print('  v2f: ', mesh['interp_matrix_v2f'].shape)
    x1 = scipy.linalg.null_space(mesh['interp_matrix_f2v'])
    print('Null space for f2v: ', x1.shape)
    x2 = scipy.linalg.null_space(mesh['interp_matrix_v2f'])
    print('Null space for v2f: ', x2.shape)

  if 1: # HW2, self question, #1
    add_edges_to_mesh(mesh)
    faces_order = bfsdfs(mesh['faces_graph'], 0, bfs_flag=True)
    faces_function = np.zeros((mesh['n_faces']))
    for i, f in enumerate(faces_order):
      faces_function[f] = i + mesh['n_faces']
    visualize_mesh(mesh, faces_function=faces_function, show_tringles=False)

  if 0: # HW2, self question, #2
    f0 = 0
    faces = cut_mesh(mesh, f0, np.array((0, 1, 1)))
    print('Number of faces used to cut the mesh:', len(faces))
    print(faces)
    faces_function = np.zeros((mesh['n_faces']))
    if 1:
      faces_function[faces] = 1
      faces_function[f0] = 2
    else:
      for i, f in enumerate(faces):
        faces_function[f] = i

    if 1:
      visualize_mesh(mesh, faces_function=faces_function, show_tringles=False)
    else:
      calc_interpolation_matrices(mesh)
      vertices_function = np.dot(mesh['interp_matrix_f2v'], faces_function)
      visualize_mesh(mesh, vertices_function=vertices_function, show_tringles=False)

