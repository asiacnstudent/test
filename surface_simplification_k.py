import sys
import time
import os
import copy
import heapq

from scipy.optimize import minimize_scalar
from tqdm import tqdm
import numpy as np
import pylab as plt
import trimesh
from scipy.spatial import Delaunay

import io_off_model
import mesh_calc
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Some flags / constants to define the simplification
CALC_OPTIMUM_NEW_POINT = True # True / False #是否进行顶点对的Q矩阵可逆的判断
ENABLE_NON_EDGE_CONTRACTION = False # True / False #控制是否合并非共享边的顶点的一个条件
CLOSE_DIST_TH = 0.1  #控制是否合并太靠近的两个顶点的阈值
SAME_V_TH_FOR_PREPROCESS = 0.001  #预处理时小于SAME_V_TH_FOR_PREPROCESS值的顶点删除
PRINT_COST = False  #是否打印顶点对的cost
SELF_CHECKING = False # True / False
np.set_printoptions(linewidth=200) #打印数组时设置行宽为 200 个字符
ALLOW_CONTRACT_NON_MANIFOLD_VERTICES = False  #是否使用边界或非流形的顶点(有孔洞或者不连通)，目前找的bunny模型是流形的
MINIMUM_NUMBER_OF_FACES = 40 #模型最少的面片数
VISUAL = True#是否对结果可视化
Boundary = True#是否显示聚类结果的边界面片

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

def run_one(mesh_id=0, n_vertices_to_merge=None):
  # 获取指定编号的网格和对应的顶点合并次数
  mesh, n_vertices_to_merge_ = get_mesh(mesh_id)    #调用get_mesh(mesh_id)
  tb = time.time()
  if n_vertices_to_merge is None:   # 如果未提供顶点合并次数，则使用默认值JU
    n_vertices_to_merge = n_vertices_to_merge_
  mesh_simplified = simplify_mesh(mesh, n_vertices_to_merge)    #调用simplify_mesh
  # print('Number of vertices in: ', mesh['n_vertices'])
  # print('Number of faces in: ', mesh['n_faces'])
  # print('Number of vertices after simplification: ', mesh_simplified['n_vertices'])
  # print('Number of faces after simplification: ', mesh_simplified['n_faces'])
  # print('runTime: ', time.time() - tb)   #计算这两个时间戳之间的差值，以获取程序的运行时间。
  # if not os.path.isdir('output_meshes'):   # 创建输出文件夹（如果不存在）并保存简化后的网格
  #   os.makedirs('output_meshes')
  # fn = 'output_meshes/' + mesh['name'].split('.')[0] + '_simplified_' + str(n_vertices_to_merge) + '.off'
  # io_off_model.write_off_mesh(fn, mesh_simplified)    # 将简化后的网格写入off文件
  # #fn = 'output_meshes/' + mesh['name'].split('.')[0] + '_simplified.obj'
  # #io_off_model.write_off_mesh(fn, mesh_simplified)
  # fn = 'output_meshes/' + mesh['name'].split('.')[0] + '.obj'  # 将原始的网格off文件转为obj文件
  # io_off_model.write_mesh(fn, mesh)
  # fn = 'output_meshes/' + mesh['name'].split('.')[0] + '_simplified_' + str(n_vertices_to_merge) + '.obj'
  # io_off_model.write_mesh(fn, mesh_simplified)
  # if VISUAL:  #对简化后结果可视化
  #   new_mesh = io_off_model.read_off(f'D:/Pycharm/Py_Projects/Mesh Simplification via Clustering-Guided Adaptive Edge Collapsing/output_meshes/bunny2_simplified_{n_vertices_to_merge}.off')
  #   plot_off(new_mesh,mesh_simplified)
  time.sleep(5)

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
    mesh_fns = [['meshes/bunny2.off',         5000],
                ['meshes/dinosaur.off',          500],    # 合并顶点对数
                ['meshes/person_0067.off',    1200],
                ['meshes/airplane_0359.off',  200],
                ['meshes/person_0004.off',    1000],

                ['meshes/cat_simplified_6000.off',            6000],
                ['meshes/ankylosaurus_simplified_6500.off',   4000],
                ['meshes/person_0004.off',    2000],
                ['meshes/person_0067.off',    2000],
                ['meshes/phands.off',         2000],
                ['meshes/dragon.off',         10000],
                ['meshes/Arma.off',           10000]
                ]
    n_vertices_to_merge = mesh_fns[idx][1]
    mesh = io_off_model.read_off(mesh_fns[idx][0], verbose=True)
    mesh['name'] = os.path.split(mesh_fns[idx][0])[-1]
  return mesh, n_vertices_to_merge

#网格简化
def simplify_mesh(mesh_orig, n_vertices_to_merge):
  mesh = copy.deepcopy(mesh_orig)    # 复制输入的网格，避免在原始网格上进行修改
  tb = time.time()
  mesh_preprocess(mesh)  #调用网格预处理，合并低于阈值的顶点
  mesh['pair_heap'] = [] # 初始化一个空的堆用于存储顶点对
  compute_gaussian_curvature(mesh)  # 计算顶点高斯曲率
  plot_mesh_clusters(mesh)  # 画出聚类后的网格
  return mesh

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

def mesh_preprocess(mesh):  # 网格预处理
  # Unite all "same" vertices - ones that are very close
  # 合并所有“相同”的顶点 - 非常接近的顶点  #SAME_V_TH_FOR_PREPROCESS = 0.001
  for v_idx, v in enumerate(mesh['vertices']):
  #mesh['vertices'] 是一个形状为 (n, 3) 的数组，其中 n 是顶点数量，每个顶点由3个坐标组成；mesh['faces'] 是一个形状为 (m, 3) 的数组，其中 m 是面数量，每个面由3个顶点索引组成。
    d = np.linalg.norm(mesh['vertices'] - v, axis=1)
    idxs0 = np.where(d < SAME_V_TH_FOR_PREPROCESS)[0][1:]
    for v_idx_to_update in idxs0:
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
  mesh_calc.add_edges_to_mesh(mesh)     # 为网格添加边信息
  print(mesh['name'], 'is water-tight:', mesh['is_watertight'])

  # Prepare mesh   # 准备网格：顶点邻接表，顶点-面邻接矩阵，法向量和到面到原点距离的增广矩阵
  mesh_calc.calc_v_adjacency_matrix(mesh)
  mesh_calc.calc_vf_adjacency_matrix(mesh)
  mesh_calc.calc_face_plane_parameters(mesh)
  # 确保网格现在是好的
  if SELF_CHECKING:
    check_mesh(mesh)

def compute_voronoi_area(vertex, other_vertex_1, other_vertex_2, is_obtuse):
  area = np.linalg.norm(np.cross(other_vertex_1 - vertex, other_vertex_2 - vertex)) / 2
  if is_obtuse:
    return area / 2
  else:
    edge_midpoint_1 = (vertex + other_vertex_1) / 2
    edge_midpoint_2 = (vertex + other_vertex_2) / 2
    return np.linalg.norm(np.cross(edge_midpoint_1 - vertex, edge_midpoint_2 - vertex)) / 2

def angle_between_vectors_dot_product(u, v): #计算夹角的弧度值
  dot_product = np.dot(u, v)
  norms_product = np.linalg.norm(u) * np.linalg.norm(v)
  cos_angle = np.clip(dot_product / norms_product, -1.0, 1.0)
  angle = np.arccos(cos_angle)
  return angle

def compute_gaussian_curvature(mesh):   #计算顶点高斯曲率
  mesh['gaussian_curvature'] = np.zeros(mesh['n_vertices'])

  for v_idx in range(mesh['n_vertices']):
    neighbor_faces = np.where(mesh['vf_adjacency_matrix'][v_idx])[0]
    angle_sum = 0
    voronoi_area = 0

    for f_idx in neighbor_faces:
      face = mesh['faces'][f_idx]
      if face[0] == v_idx:
        other_vertices = face[1:]
      elif face[1] == v_idx:
        other_vertices = [face[0], face[2]]
      else:
        other_vertices = face[:2]

      edge_vector_1 = mesh['vertices'][other_vertices[0]] - mesh['vertices'][v_idx]
      edge_vector_2 = mesh['vertices'][other_vertices[1]] - mesh['vertices'][v_idx]
      angle = angle_between_vectors_dot_product(edge_vector_1, edge_vector_2)
      angle_sum += angle
      is_obtuse = angle > np.pi / 2
      voronoi_area += compute_voronoi_area(mesh['vertices'][v_idx], mesh['vertices'][other_vertices[0]],
                                           mesh['vertices'][other_vertices[1]], is_obtuse)

    if voronoi_area == 0:
      mesh['gaussian_curvature'][v_idx] = 0
    else:
      mesh['gaussian_curvature'][v_idx] = (2 * np.pi - angle_sum) / voronoi_area

  return mesh['gaussian_curvature']

def normalize_curvature(cluster_curvature, min_curvature, max_curvature):  #高斯曲率归一化[-1,1]
  # 线性归一化到 [-1, 1] 范围内
  normalized_curvature = -1 + 2 * (cluster_curvature - min_curvature) / (max_curvature - min_curvature)
  return normalized_curvature

def plot_mesh_clusters(mesh): # 谱聚类及结果可视化
  vertices = mesh['vertices']
  num_vertices = len(mesh['vertices'])
  faces = mesh['faces']
  gaussian_curvature = mesh['gaussian_curvature']
  global_min_curvature = gaussian_curvature.min()
  global_max_curvature = gaussian_curvature.max()

  normalized_gaussian_curvature = normalize_curvature(gaussian_curvature, global_min_curvature, global_max_curvature)
  mesh['normalized_gaussian_curvature'] = normalized_gaussian_curvature

  # 谱聚类部分
  curvature = mesh['normalized_gaussian_curvature']
  v_adjacency_matrix = mesh['v_adjacency_matrix']

  # 构建权重矩阵
  sigma = 0.5  # 合适的 sigma 值
  neighbor_radius = 0.2  # 邻域半径
  # 计算顶点之间的欧氏距离
  vertex_distances = np.linalg.norm(vertices[:, np.newaxis, :] - vertices[np.newaxis, :, :], axis=-1)

  # 基于高斯曲率和邻域距离构建权重矩阵
  W = np.exp(-np.square(np.subtract.outer(curvature, curvature)) / sigma ** 2) * (
            vertex_distances < neighbor_radius) * v_adjacency_matrix

  # 非线性变换
  alpha = 0.5  # 控制非线性变换的强度
  W = W ** alpha

  # 度矩阵
  D = np.diag(W.sum(axis=1))

  # 拉普拉斯矩阵
  L = D - W

  # 计算拉普拉斯矩阵的特征向量
  num_clusters = 5  # 固定为某个值
  eigvals, eigvecs = eigsh(L, k=num_clusters, which='SM')

  # 特征向量聚类
  U = eigvecs[:, :num_clusters]

  # 记录簇数量和对应的 SSE
  sse_list = []
  cluster_range = range(5, 26)  # 从5到25的簇数

  for n_clusters in cluster_range:
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(U)
    sse_list.append(kmeans.inertia_)  # inertia_ 是 KMeans 的 SSE 值

  # 绘制簇数与 SSE 的图
  plt.plot(cluster_range, sse_list, 'bx-')
  plt.xlabel('Number of clusters (K)')
  plt.ylabel('SSE')
  plt.title('Elbow Method for Optimal K')
  plt.show()
  return

if __name__ == '__main__':
  #run_all()
  #run_bunny_many(=
  run_one(0)      #  调用run_one函数，参数为n，即运行第n+1个模型的简化