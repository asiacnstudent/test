import sys
import threading
import time
import os
import copy
import heapq
import open3d as o3d

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import minimize_scalar
from scipy.spatial import distance
from tqdm import tqdm
import numpy as np
import pylab as plt
import trimesh

import io_off_model
import mesh_calc

# Some flags / constants to define the simplification
CALC_OPTIMUM_NEW_POINT = False # True / False
ENABLE_NON_EDGE_CONTRACTION = False # True / False #控制是否合并非共享边的顶点的一个条件
CLOSE_DIST_TH = 0.1  #控制是否合并太靠近的两个顶点的阈值
SAME_V_TH_FOR_PREPROCESS = 0.001  #预处理时小于SAME_V_TH_FOR_PREPROCESS值的顶点删除
PRINT_COST = False  #是否打印顶点对的cost
SELF_CHECKING = False # True / False
np.set_printoptions(linewidth=200) #打印数组时设置行宽为 200 个字符
ALLOW_CONTRACT_NON_MANIFOLD_VERTICES = False  #是否使用边界或非流形的顶点(有孔洞或者不连通)，目前找的bunny模型是流形的
MINIMUM_NUMBER_OF_FACES = 40 #模型最少的面片数
VISUAL = True


class Logger(object):
  def __init__(self, filename='default.log', stream=sys.stdout):
    self.terminal = stream
    # 确保输出目录存在
    if not os.path.exists('output_data'):
      os.makedirs('output_data')

    # 处理日志文件名
    self.log_filename = self.get_unique_filename(filename)
    self.log = open(self.log_filename, 'w')

  def get_unique_filename(self, filename):
    base, ext = os.path.splitext(filename)
    count = 1
    new_filename = os.path.join('output_data', filename)

    # 检查同名文件并添加数字后缀
    while os.path.exists(new_filename):
      new_filename = os.path.join('output_data', f"{base}_{count}{ext}")
      count += 1

    return new_filename

  def write(self, message):
    self.log.write(message)
    self.terminal.write(message)

  def flush(self):
    pass

  def close(self):
    self.log.close()

program_name = os.path.splitext(os.path.basename(__file__))[0]
sys.stdout = Logger(f'{program_name}.log', sys.stdout)
sys.stderr = Logger(f'{program_name}_Iteration.log', sys.stderr)

def calc_Q_for_vertex(mesh, v_idx):    # 计算顶点的 Q 矩阵
  # Calculate K & Q according to eq. (2)
  Q = np.zeros((4, 4))
  for f_idx in np.where(mesh['vf_adjacency_matrix'][v_idx])[0]:
    plane_params = mesh['face_plane_parameters'][f_idx][:, None]
    Kp = plane_params * plane_params.T
    Q += Kp
  return Q

def calc_Q_for_each_vertex(mesh):     # 计算每个顶点的 Q 矩阵
  # Prepare some mesh paramenters and run on all vertices to call Q calculation   # 准备一些网格参数并遍历所有顶点以调用 Q 计算
  mesh['all_v_in_same_plane'] = np.abs(np.diff(mesh['face_plane_parameters'], axis=0)).sum() == 0
  mesh['Qs'] = []
  for v_idx in range(mesh['n_vertices']):
    Q = calc_Q_for_vertex(mesh, v_idx)
    mesh['Qs'].append(Q)

def add_pair(mesh, v1, v2, edge_connection):    # 将一对顶点添加到堆中
  # Do not use vertices on bound or non-manifold ones    # 不要使用边界或非流形的顶点
  if not ALLOW_CONTRACT_NON_MANIFOLD_VERTICES:
    if not mesh['is_watertight'] and (v1 in mesh['non_maniford_vertices'] or v2 in mesh['non_maniford_vertices']):
      return

  # Add pair of indices to the heap, keys by the cost    # 将索引对按成本推入堆中
  Q = mesh['Qs'][v1] + mesh['Qs'][v2]
  new_v1_ = calc_new_vertex_position(mesh, v1, v2, Q)
  if mesh['all_v_in_same_plane']:
    cost = np.linalg.norm(mesh['vertices'][v1] - mesh['vertices'][v2])
  else:
    new_v1 = np.vstack((new_v1_[:, None], np.ones((1, 1))))
    cost = np.dot(np.dot(new_v1.T, Q), new_v1)[0, 0]
  if PRINT_COST:
    print('For pair: ', v1, ',', v2, ' ; the cost is: ', cost)
  heapq.heappush(mesh['pair_heap'], (cost, v1, v2, edge_connection, new_v1_))

def select_vertex_pairs(mesh):  # 选择顶点对
  print('Calculating pairs cost and add to heap')
  for v1 in tqdm(range(mesh['n_vertices'])):
    for v2 in range(v1 + 1, mesh['n_vertices']):
      edge_connection = mesh['v_adjacency_matrix'][v1, v2]
      vertices_are_very_close = ENABLE_NON_EDGE_CONTRACTION and np.linalg.norm(mesh['vertices'][v2] - mesh['vertices'][v1]) < CLOSE_DIST_TH
      if edge_connection or vertices_are_very_close:
        add_pair(mesh, v1, v2, edge_connection)

def calc_new_vertex_position(mesh, v1, v2, Q):   # 计算新顶点位置
  # Calculating the new vetrex position, given 2 vertices (paragraph 4.):
  # 1. If A (to be defined below) can be inverted, use it
  # 2. If this matrix is not invertible, attempt to find the optimal vertex along the segment V1 and V2
  # 3. The new vertex will be at the midpoint
  # 计算新的顶点位置，给定两个顶点（第 4 段）：
  # 1. 如果 A（将在下面定义）可逆，使用它
  # 2. 如果这个矩阵不可逆，则尝试沿着 V1 和 V2 段找到最佳顶点
  # 3. 2失败则新的顶点将在中点处
  A = Q.copy()
  A[3] = [0, 0, 0, 1]                                 # Defined by eq. (1)
  if np.linalg.matrix_rank(A) == 4:
    A_can_be_ineverted = True #检查 A 的秩是否为 4 来判断矩阵是否可逆。如果 A 的秩为 4，那么 A 是满秩矩阵，因此是可逆的
  else:
    A_can_be_ineverted = False
  # 如果 A 可逆，计算 A 的逆矩阵 A_inv，然后用 A_inv 乘以列向量 [0, 0, 0, 1]，得到的新顶点 new_v1 是该乘积的前三个分量。
  if A_can_be_ineverted: #A可逆
    if CALC_OPTIMUM_NEW_POINT:
      A_inv = np.linalg.inv(A)  #求逆
      new_v1 = np.dot(A_inv, np.array([[0, 0, 0, 1]]).T)[:3]
      new_v1 = np.squeeze(new_v1)
    else:
      new_v1 = (mesh['vertices'][v1] + mesh['vertices'][v2]) / 2
  else: #A不可逆
    new_v1 = look_for_minimum_cost_on_connected_line(mesh, v1, v2, Q)
  return new_v1

def look_for_minimum_cost_on_connected_line(mesh, v1, v2, Q): #Q矩阵不可逆时使用，在连接线上寻找最小成本
  def error_function(t):
    point = (1 - t) * mesh['vertices'][v1] + t * mesh['vertices'][v2]
    point_homogeneous = np.append(point, 1)  # 将点转换为齐次坐标
    return np.dot(point_homogeneous, np.dot(Q, point_homogeneous))
  result = minimize_scalar(error_function, bounds=(0, 1), method='bounded')
  if result.success:
    t_opt = result.x
    new_vertex = (1 - t_opt) * mesh['vertices'][v1] + t_opt * mesh['vertices'][v2]
  else:
    # 如果优化失败，使用中点作为新的顶点位置
    new_vertex = (mesh['vertices'][v1] + mesh['vertices'][v2]) / 2
  return new_vertex

def contract_best_pair(mesh):
  # Get the best pair of indices from heap, and contract them to a single vertex
  # 合并最佳的一对

  # get pair from heap  # 从堆中获取一对
  if len(mesh['pair_heap']) == 0:
    return
  cost, v1, v2, is_edge, new_v1 = heapq.heappop(mesh['pair_heap'])

  # update v1 - position
  mesh['vertices'][v1] = new_v1

  # remove v2:
  mesh['vertices'][v2] = [-1, -1, -1]                # 从网格中“移除”顶点（最终将在函数：clean_mesh_from_removed_items 中移除）
  mesh['v_adjacency_matrix'][v2, :] = False
  mesh['v_adjacency_matrix'][:, v2] = False
  if is_edge:
    all_v2_faces = np.where(mesh['vf_adjacency_matrix'][v2])[0]
    mesh['vf_adjacency_matrix'][v2, :] = False
    for f in all_v2_faces:
      if v1 in mesh['faces'][f]:                      # 如果面包含 v2 也与 v1 共享顶点：
        mesh['faces'][f] = [-1, -1, -1]               #  "remove" face from mesh.
        mesh['vf_adjacency_matrix'][:, f] = False
      else:                                           # else:
        v2_idx = np.where(mesh['faces'][f] == v2)[0]  #  replace v2 with v1
        new_v1_nbrs = mesh['faces'][f][mesh['faces'][f] != v2]
        mesh['faces'][f, v2_idx] = v1
        mesh['vf_adjacency_matrix'][v1, f] = True
        mesh['v_adjacency_matrix'][v1, new_v1_nbrs] = True
        mesh['v_adjacency_matrix'][new_v1_nbrs, v1] = True
  else:
    mesh['faces'][mesh['faces'] == v2] = v1
    idxs = np.where(np.sum(mesh['faces'] == v1, axis=1) > 1)[0]
    mesh['faces'][idxs, :] = -1

  mesh['n_faces'] = (mesh['faces'][:, 0] != -1).sum()

  if SELF_CHECKING:
    check_mesh(mesh)

  # remove all v1, v2 pairs from heap (forbidden_vertices can be than removed)
  # 从堆中移除所有 v1, v2 对（forbidden_vertices 然后可以移除）
  for pair in mesh['pair_heap'][:]:
    if pair[1] in [v1, v2] or pair[2] in [v1, v2]:
      mesh['pair_heap'].remove(pair)

  # Update Q of vertex v1
  if CALC_OPTIMUM_NEW_POINT:
    update_planes_parameters_near_vertex(mesh, v1)
    calc_Q_for_vertex(mesh, v1)

  # add new pairs of the new vertex
  v2 = None
  for v2_ in range(mesh['n_vertices']):
    if v1 == v2:
      continue
    edge_connection = mesh['v_adjacency_matrix'][v1, v2_]
    vertices_are_very_close = ENABLE_NON_EDGE_CONTRACTION and np.linalg.norm(mesh['vertices'][v2] - mesh['vertices'][v1]) < CLOSE_DIST_TH
    if edge_connection or vertices_are_very_close:
      add_pair(mesh, v1, v2_, edge_connection)

def update_planes_parameters_near_vertex(mesh, v):
  # Get faces near v and recalculate their plane parameters
  # 获取靠近 v 的面并重新计算它们的平面参数
  mesh_calc.calc_face_plane_parameters(mesh, must_recalc=True)

def check_mesh(mesh):  # 检查网格的一致性
  # Check that there is no duplicated face   # 检查是否存在重复的面
  f_idx_no_erased = np.where(mesh['faces'][:, 1] != -1)[0]
  real_faces = mesh['faces'][f_idx_no_erased]
  if np.unique(real_faces, axis=0).shape != real_faces.shape:
    raise Exception('Duplicated face')
  # Check that adjacent matrices coherent to faces list   # 检查邻接矩阵是否与面列表一致
  tmp_v_adjacency_matrix = mesh['v_adjacency_matrix'].copy()
  tmp_vf_adjacency_matrix = mesh['vf_adjacency_matrix'].copy()
  for f_idx, f in enumerate(mesh['faces']):
    if f[0] == -1:
      continue
    for v1_, v2_ in [(f[0], f[1]), (f[0], f[2]), (f[1], f[2])]:
      tmp_v_adjacency_matrix[v1_, v2_] = False
      tmp_v_adjacency_matrix[v2_, v1_] = False
      if mesh['v_adjacency_matrix'][v1_, v2_] == False or mesh['v_adjacency_matrix'][v2_, v1_] == False:
        raise Exception('Bad v_adjacency_matrix')
    for v_ in f:
      tmp_vf_adjacency_matrix[v_, f_idx] = False
      if mesh['vf_adjacency_matrix'][v_, f_idx] == False:
        raise Exception('Bad vf_adjacency_matrix')
  if np.any(tmp_vf_adjacency_matrix):
    raise Exception('vf_adjacency_matrix has wrong True elements')
  if np.any(tmp_v_adjacency_matrix):
    raise Exception('v_adjacency_matrix has wrong True elements')

  # Check if a face have 2 same vertex indices  # 检查面是否有 2 个相同的顶点索引
  idxs = np.where(mesh['faces'][:, 0] != -1)[0]
  to_check = mesh['faces'][idxs]
  if np.any(np.diff(np.sort(to_check, axis=1), axis=1) == 0):
    raise Exception('Bug: face found with 2 idintical vertex indices!')


def clean_mesh_from_removed_items(mesh):   # 从删除的元素中清理网格
  # Remove Faces
  faces2delete = np.where(np.all(mesh['faces'] == -1, axis=1))[0]
  mesh['faces'] = np.delete(mesh['faces'], faces2delete, 0)

  # Remove vertices and fix face indices   # 删除顶点并修复面索引
  is_to_remove = (mesh['vertices'][:, 0] == -1) + np.isnan(mesh['vertices'][:, 0])
  v_to_remove = np.where(is_to_remove)[0]
  v_to_keep   = np.where(is_to_remove == 0)[0]
  mesh['vertices'] = mesh['vertices'][v_to_keep, :]
  for v_idx in v_to_remove[::-1]:
    f_to_update = np.where(mesh['faces'] > v_idx)
    mesh['faces'][f_to_update] -= 1
  # 更新顶点和面数量
  mesh['n_vertices'] = len(mesh['vertices'])
  mesh['n_faces'] = len(mesh['faces'])

def mesh_preprocess(mesh):  # 网格预处理
  # Unite all "same" vertices - ones that are very close
  # 合并所有“相同”的顶点 - 非常接近的顶点
  for v_idx, v in enumerate(mesh['vertices']):
    d = np.linalg.norm(mesh['vertices'] - v, axis=1)
    idxs0 = np.where(d < SAME_V_TH_FOR_PREPROCESS)[0][1:]
    for v_idx_to_update in idxs0:  #将小于SAME_V_TH_FOR_PREPROCESS值的顶点删除
      mesh['vertices'][v_idx_to_update] = [np.nan, np.nan, np.nan]
      idxs1 = np.where(mesh['faces'] == v_idx_to_update)
      mesh['faces'][idxs1] = v_idx

  # Remove duplicated faces
  # 删除重复的面  #因为通过预处理会产生相同的顶点
  for f in mesh['faces']:
    if f[0] == -1:
      continue
    dup_faces = np.where(np.all(mesh['faces'] == f, axis=1))[0][1:]
    mesh['faces'][dup_faces, :] = -1

  # Check if model is watertight   # 检查模型是否是封闭的
  mesh_calc.add_edges_to_mesh(mesh)
  print(mesh['name'], 'is water-tight:', mesh['is_watertight'])

  # Prepare mesh   # 准备网格
  mesh_calc.calc_v_adjacency_matrix(mesh)
  mesh_calc.calc_vf_adjacency_matrix(mesh)
  mesh_calc.calc_face_plane_parameters(mesh)

  # Make sure the mesh is good now
  # 确保网格现在是好的
  if SELF_CHECKING:
    check_mesh(mesh)

def simplify_mesh(mesh_orig, n_vertices_to_merge):
  # 复制输入的网格，避免在原始网格上进行修改
  mesh = copy.deepcopy(mesh_orig)
  tb = time.time()
  mesh_preprocess(mesh)  #网格预处理
  plot_off(mesh)
  print('Init time:', time.time() - tb)

  mesh['pair_heap'] = [] # 初始化一个空的堆用于存储顶点对
  calc_Q_for_each_vertex(mesh)
  # Select pairs and add them to a heap    # 选择顶点对并将它们添加到堆中
  select_vertex_pairs(mesh)
  # Take and contract pairs   # 取出并合并顶点对，直到达到指定的顶点合并次数或者网格面数小于等于设定的最小面数

  tb = time.time()
  for _ in tqdm(range(n_vertices_to_merge)):
    contract_best_pair(mesh)
    if mesh['n_faces'] <= MINIMUM_NUMBER_OF_FACES:
      break
  clean_mesh_from_removed_items(mesh)
  print('Iteration time:', time.time() - tb)
  return mesh

def get_mesh(idx=0):
  global CLOSE_DIST_TH

  if idx == -1:
    mesh = io_off_model.get_simple_mesh('for_mesh_simplification_1')
    mesh['name'] = 'simple_2d_mesh_1'
    n_vertices_to_merge = 1
  elif idx == -2:
    mesh = io_off_model.get_simple_mesh('for_mesh_simplification_2')
    mesh['name'] = 'simple_2d_mesh_2'
    n_vertices_to_merge = 1
    CLOSE_DIST_TH = 0.5
  else:
    mesh_fns = [['meshes/bunny2.off',        5000],
                ['meshes/cat.off',           2500],
                ['meshes/dinosaur.off',       500],
                ['meshes/horse_simplified_40000.off',       4000],
                ['meshes/Arma_simplified_10000.off',  5000],
                ['meshes/person_0067.off',    1200],
                ['meshes/person_0004.off',    1000],
                ['meshes/ankylosaurus.off',   2000],
                ['meshes/person_0004.off',    2000],
                ['meshes/person_0067.off',    2000],
                ['meshes/phands.off',         2000],
                ['meshes/fandisk.off', 2000],
                ['meshes/plants.off',         5000]
                ]
    n_vertices_to_merge = mesh_fns[idx][1]
    mesh = io_off_model.read_off(mesh_fns[idx][0], verbose=True)
    mesh['name'] = os.path.split(mesh_fns[idx][0])[-1]
  return mesh, n_vertices_to_merge

def run_one(mesh_id=0, n_vertices_to_merge=None):
  # 获取指定编号的网格和对应的顶点合并次数
  mesh, n_vertices_to_merge_ = get_mesh(mesh_id)    #调用get_mesh(mesh_id)
  if n_vertices_to_merge is None:   # 如果未提供顶点合并次数，则使用默认值JU
    n_vertices_to_merge = n_vertices_to_merge_
  mesh_simplified = simplify_mesh(mesh, n_vertices_to_merge)    #调用simplify_mesh
  print('Number of vertices in: ', mesh['n_vertices'])
  print('Number of faces in: ', mesh['n_faces'])
  print('Number of vertices after simplification: ', mesh_simplified['n_vertices'])
  print('Number of faces after simplification: ', mesh_simplified['n_faces'])
  if not os.path.isdir('output_meshes'):   # 创建输出文件夹（如果不存在）并保存简化后的网格
    os.makedirs('output_meshes')
  fn = 'output_meshes/' + mesh['name'].split('.')[0] + '_origin'+ '_simplified_' + str(n_vertices_to_merge) + '.off'
  io_off_model.write_off_mesh(fn, mesh_simplified)    # 将简化后的网格写入off文件
  #fn = 'output_meshes/' + mesh['name'].split('.')[0] + '_simplified.obj'
  #io_off_model.write_off_mesh(fn, mesh_simplified)
  fn = 'output_meshes/' + mesh['name'].split('.')[0] + '.obj'  # 将原始的网格off文件转为obj文件
  io_off_model.write_mesh(fn, mesh)
  fn = 'output_meshes/' + mesh['name'].split('.')[0] + '_origin'+'_simplified_' + str(n_vertices_to_merge) + '.obj'
  io_off_model.write_mesh(fn, mesh_simplified)
  #Hausdorff距离计算
  mesh_original = o3d.io.read_triangle_mesh(f'D:/Pycharm/Py_Projects/SurfaceSimplification-master/meshes/{mesh["name"].split(".")[0]}.off')
  mesh_simplified = o3d.io.read_triangle_mesh(f'D:/Pycharm/Py_Projects/SurfaceSimplification-master/output_meshes/{mesh["name"].split(".")[0]}_origin_simplified_{n_vertices_to_merge}.off')
  # 提取网格的顶点信息
  vertices_original = np.asarray(mesh_original.vertices)
  vertices_simplified = np.asarray(mesh_simplified.vertices)
  # 计算原始网格到简化网格的定向 Hausdorff 距离
  hausdorff_distance_1 = distance.directed_hausdorff(vertices_original, vertices_simplified)[0]
  # 计算简化网格到原始网格的定向 Hausdorff 距离
  hausdorff_distance_2 = distance.directed_hausdorff(vertices_simplified, vertices_original)[0]
  # 取两个方向的最大值作为 Hausdorff 距离
  hausdorff_distance = max(hausdorff_distance_1, hausdorff_distance_2)
  print(f"Hausdorff Distance: {hausdorff_distance}")
  if VISUAL:  #对简化后结果可视化
    new_mesh = io_off_model.read_off(f'D:/Pycharm/Py_Projects/SurfaceSimplification-master/output_meshes/{mesh["name"].split(".")[0]}_origin_simplified_{n_vertices_to_merge}.off')
    plot_off(new_mesh)

  time.sleep(5)


def plot_off(new_mesh):


  faces = new_mesh['faces']
  vertices = new_mesh['vertices']

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  # 绘制三角面片
  face_vertices = vertices[faces]
  poly3d = [face_vertices[i] for i in range(face_vertices.shape[0])]
  ax.add_collection3d(Poly3DCollection(poly3d, facecolors='cyan', linewidths=0.1, edgecolors='k', alpha=0.7))

  # 设置坐标轴范围
  ax.set_xlim([np.min(vertices[:, 0]), np.max(vertices[:, 0])])
  ax.set_ylim([np.min(vertices[:, 1]), np.max(vertices[:, 1])])
  ax.set_zlim([np.min(vertices[:, 2]), np.max(vertices[:, 2])])
  # 设置统一视角
  ax.view_init(elev=90, azim=-90)
  # 隐藏坐标轴
  ax.set_axis_off()
  plt.show()



def run_all():
  # 分别运行简化和不简化的模型
  simple_models = [-2, -1]
  watertight_models = [4, 5, 6]
  non_watertight_models = [0, 1, 2, 3]
  for mesh_id in non_watertight_models:  # 对非watertight的模型运行简化
    run_one(mesh_id)

if __name__ == '__main__':
  #run_all()
  #run_bunny_many()
  run_one(0)      #  调用run_one函数，参数为5，即运行第5个模型的简化