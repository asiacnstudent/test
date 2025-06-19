import sys
import time
import os
import copy
import heapq
import open3d as o3d

from scipy.optimize import minimize_scalar
from tqdm import tqdm
import numpy as np
import pylab as plt
import trimesh
from scipy.spatial import Delaunay, distance

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
CALC_OPTIMUM_NEW_POINT = True # True / False
ENABLE_NON_EDGE_CONTRACTION = False # True / False #控制是否合并非共享边的顶点的一个条件
CLOSE_DIST_TH = 0.1  #控制是否合并太靠近的两个顶点的阈值
SAME_V_TH_FOR_PREPROCESS = 0.001  #预处理时小于SAME_V_TH_FOR_PREPROCESS值的顶点删除
PRINT_COST = False  #是否打印顶点对的cost
SELF_CHECKING = False # True / False
np.set_printoptions(linewidth=200) #打印数组时设置行宽为 200 个字符
ALLOW_CONTRACT_NON_MANIFOLD_VERTICES = False  #是否使用边界或非流形的顶点(有孔洞或者不连通)，目前找的bunny模型是流形的
MINIMUM_NUMBER_OF_FACES = 40 #模型最少的面片数
VISUAL = True#是否对结果可视化
Boundary = False#是否显示聚类结果的边界面片


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

def plot_off(new_mesh, mesh_simplified):
  faces = new_mesh['faces']
  vertices = new_mesh['vertices']
  face_labels = mesh_simplified['faces_clusters']  # 使用 faces_clusters

  # 获取边界标志
  vertex_boundary_flags = mesh_simplified['vertex_boundary_flags']
  face_boundary_flags = mesh_simplified['face_boundary_flags']

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  # 定义颜色映射
  colors = plt.cm.get_cmap('viridis', num_clusters)
  boundary_color = 'red'  # 边界面片的特殊颜色

  # 绘制每个聚类的面片
  for i in range(num_clusters):
    cluster_faces = faces[face_labels  == i]
    face_vertices = vertices[cluster_faces]
    poly3d = [face_vertices[j] for j in range(face_vertices.shape[0])]
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors=colors(i), linewidths=0.1, edgecolors='k', alpha=0.7))

  #边界面片
  if Boundary:
    boundary_faces = faces[face_boundary_flags]
    boundary_face_vertices = vertices[boundary_faces]
    boundary_poly3d = [boundary_face_vertices[j] for j in range(boundary_face_vertices.shape[0])]
    ax.add_collection3d(
      Poly3DCollection(boundary_poly3d, facecolors=boundary_color, linewidths=0.1, edgecolors='k', alpha=0.7))
    # 隐藏坐标轴
  ax.set_axis_off()
  plt.show()

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
  fn = 'output_meshes/' + mesh['name'].split('.')[0] + '_simplified_' + str(n_vertices_to_merge) + '.off'
  io_off_model.write_off_mesh(fn, mesh_simplified)    # 将简化后的网格写入off文件
  #fn = 'output_meshes/' + mesh['name'].split('.')[0] + '_simplified.obj'
  #io_off_model.write_off_mesh(fn, mesh_simplified)
  fn = 'output_meshes/' + mesh['name'].split('.')[0] + '.obj'  # 将原始的网格off文件转为obj文件
  io_off_model.write_mesh(fn, mesh)
  fn = 'output_meshes/' + mesh['name'].split('.')[0] + '_simplified_' + str(n_vertices_to_merge) + '.obj'
  io_off_model.write_mesh(fn, mesh_simplified)
  if VISUAL:  #对简化后结果可视化
    new_mesh = io_off_model.read_off(f'D:/Pycharm/Py_Projects/SurfaceSimplification-master/output_meshes/{mesh["name"].split(".")[0]}_simplified_{n_vertices_to_merge}.off')
    plot_off(new_mesh,mesh_simplified)
  #Hausdorff距离计算
  mesh_original = o3d.io.read_triangle_mesh(f'D:/Pycharm/Py_Projects/SurfaceSimplification-master/meshes/{mesh["name"].split(".")[0]}.off')
  mesh_simplified = o3d.io.read_triangle_mesh(f'D:/Pycharm/Py_Projects/SurfaceSimplification-master/output_meshes/{mesh["name"].split(".")[0]}_simplified_{n_vertices_to_merge}.off')
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
    mesh_fns = [['meshes/dinosaur.off',       1500],    # 合并顶点对数
                ['meshes/person_0067.off',    1200],
                ['meshes/airplane_0359.off',  200],
                ['meshes/person_0004.off',    1000],
                ['meshes/bunny2.off',         5000],
                ['meshes/cat_simplified_6000.off',            6000],
                ['meshes/ankylosaurus_simplified_6500.off',   4000],
                ['meshes/person_0004.off',    2000],
                ['meshes/person_0067.off',    2000],
                ['meshes/phands.off',         2000]
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
  calc_Q_for_each_vertex(mesh) # 计算所有顶点的Q矩阵
  print('Initial time:', time.time() - tb)
  compute_gaussian_curvature(mesh)  # 计算顶点高斯曲率
  plot_mesh_clusters(mesh,n_vertices_to_merge)  # 画出聚类后的网格
  #分类入堆，在一个类入堆后删减目标任务面片，清空堆后再入下个类
  vertices_clusters = mesh['vertices_clusters']
  clusters_ratio = mesh['clusters_ratio']
  ##删减数量确定
  cluster_pair_counts = mesh['tasks_per_cluster']
  tb = time.time()
  for cluster_idx in range(num_clusters):   #类间循环
    cluster_vertices = np.where(vertices_clusters == cluster_idx)[0]
    select_vertex_pairs(mesh,cluster_vertices)  # 选择每个类别的所有顶点对并将它们添加到堆中
    for _ in tqdm(range(cluster_pair_counts[cluster_idx])):
      contract_best_pair(mesh)
      if mesh['n_faces'] <= MINIMUM_NUMBER_OF_FACES:
        break
    clear_heap(mesh)  #清空堆
  #合并顶点对，直到达到指定的顶点合并次数或者网格面数小于等于设定的最小面数
  print('Iteration time:', time.time() - tb)
  print("Mesh simplification completed.")
  # Remove old unused faces   # 移除旧的未使用的面
  clean_mesh_from_removed_items(mesh)
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

num_clusters = 12  #聚类中心数量，全局变量
def plot_mesh_clusters(mesh,n_vertices_to_merge):  # 可视化聚类结果
  vertices = mesh['vertices']
  num_vertices = len(mesh['vertices'])
  faces = mesh['faces']
  gaussian_curvature = mesh['gaussian_curvature']
  global_min_curvature = gaussian_curvature.min()
  global_max_curvature = gaussian_curvature.max()

  normalized_gaussian_curvature = normalize_curvature(gaussian_curvature, global_min_curvature, global_max_curvature)
  mesh['normalized_gaussian_curvature'] = normalized_gaussian_curvature

  #谱聚类
  curvature = mesh['normalized_gaussian_curvature']
  v_adjacency_matrix = mesh['v_adjacency_matrix']
  # 构建权重矩阵
  sigma = 0.5  # 选择一个合适的sigma值
  neighbor_radius = 0.2  # 邻域半径，控制权重矩阵的局部性
  # 计算顶点之间的欧氏距离
  vertex_distances = np.linalg.norm(vertices[:, np.newaxis, :] - vertices[np.newaxis, :, :], axis=-1)
  # 基于高斯曲率和邻域距离构建权重矩阵
  W = np.exp(-np.square(np.subtract.outer(curvature, curvature)) / sigma ** 2) * (
            vertex_distances < neighbor_radius) * v_adjacency_matrix
  # 使用非线性变换增强高斯曲率变化大的区域
  alpha = 0.5  # 控制非线性变换的强度
  W = W ** alpha
  # 计算度矩阵
  D = np.diag(W.sum(axis=1))
  # 计算拉普拉斯矩阵
  L = D - W
  # 计算拉普拉斯矩阵的特征值和特征向量
    # 假设我们希望分成n个簇
  eigvals, eigvecs = eigsh(L, k=num_clusters, which='SM')
  # 选择前 num_clusters 个特征向量
  U = eigvecs[:, :num_clusters]
  # 使用k-means对特征向量进行聚类
  kmeans = KMeans(n_clusters=num_clusters)
  labels = kmeans.fit_predict(U)# 打印聚类结果
  # 将顶点聚类结果存入 mesh['vertices_clusters']
  mesh['vertices_clusters'] = labels
  # 计算面片的高斯曲率（顶点高斯曲率的平均值）
  face_curvature = np.mean(normalized_gaussian_curvature[faces], axis=1)
  # 计算每个面片的标签（通过计算每个面片的顶点标签的众数）
  face_labels = np.array([np.bincount(labels[face]).argmax() for face in faces])
  # 将面片聚类结果存入 mesh['faces_clusters']
  mesh['faces_clusters'] = face_labels
  #*******
  # 统计每个类别的顶点总数和面片总数,占比
  vertex_counts = np.bincount(labels, minlength=num_clusters)
  face_counts = np.bincount(face_labels, minlength=num_clusters)
  for i in range(num_clusters):
    print(f'Cluster {i + 1}: Vertices count = {vertex_counts[i]}, Faces count = {face_counts[i]}')
  clusters_ratio = np.array(vertex_counts) / num_vertices
  mesh['clusters_ratio'] = clusters_ratio

  # 初始化一个列表保存每个类别的分配任务数量
  tasks_per_cluster = []
  for i in range(num_clusters):
    # 获取当前类别的曲率和顶点数量
    cluster_curvatures = face_curvature[face_labels == i]
    num_clusters_vertices = vertex_counts[i]

    # 计算高斯曲率绝对值的均值
    mean_abs_curvature = np.abs(cluster_curvatures).mean()

    # 计算曲率的权重（高斯曲率绝对值越小，权重越大）
    curvature_weight = 1 / mean_abs_curvature

    # 计算顶点数量的权重
    vertex_weight = num_clusters_vertices / num_vertices

    # 计算总权重
    total_weight = curvature_weight * vertex_weight

    # 保存权重，用于后续分配任务
    tasks_per_cluster.append(total_weight)
  # 归一化权重并计算每个类别的任务分配数量
  total_weight_sum = sum(tasks_per_cluster)
  for i in range(num_clusters):
    # 计算每个类别的分配任务数量
    tasks_per_cluster[i] = int((tasks_per_cluster[i] / total_weight_sum) * n_vertices_to_merge)
    print(f'Cluster {i + 1}: Assigned merge tasks = {tasks_per_cluster[i]}')

  mesh['tasks_per_cluster'] = tasks_per_cluster
  # 打印每个类的归一化高斯曲率范围
  for i in range(num_clusters):
    # 获取当前簇的曲率
    cluster_curvatures = face_curvature[face_labels == i]

    # 计算曲率的绝对值范围
    min_curvature = np.abs(cluster_curvatures).min()
    max_curvature = np.abs(cluster_curvatures).max()

    # 输出结果
    print(f'Cluster {i + 1}: Absolute Curvature range [{min_curvature}, {max_curvature}]')

  # 标记不同类别的边界顶点和面片
  vertex_boundary_flags = np.zeros(vertices.shape[0], dtype=bool)
  face_boundary_flags = np.zeros(faces.shape[0], dtype=bool)
  for i, face in enumerate(faces):
    for j in range(3):
      for k in range(j + 1, 3):
        if labels[face[j]] != labels[face[k]]:
          vertex_boundary_flags[face[j]] = True
          vertex_boundary_flags[face[k]] = True
          face_boundary_flags[i] = True
  mesh['vertex_boundary_flags'] = vertex_boundary_flags
  mesh['face_boundary_flags'] = face_boundary_flags

  # 可视化三角面片
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  colors = plt.cm.get_cmap('viridis', num_clusters)
  for i in range(num_clusters):
    cluster_faces = faces[face_labels  == i]
    face_vertices = vertices[cluster_faces]
    poly3d = [face_vertices[j] for j in range(face_vertices.shape[0])]
    ax.add_collection3d(Poly3DCollection(poly3d, facecolors=colors(i), linewidths=0.1, edgecolors='k', alpha=0.7))

  # 标记边界面片
  if Boundary:
    boundary_color = 'red'  # 边界面片的特殊颜色
    boundary_faces = faces[face_boundary_flags]
    boundary_face_vertices = vertices[boundary_faces]
    boundary_poly3d = [boundary_face_vertices[j] for j in range(boundary_face_vertices.shape[0])]
    ax.add_collection3d(
      Poly3DCollection(boundary_poly3d, facecolors=boundary_color, linewidths=0.1, edgecolors='k', alpha=0.7))
  # 隐藏坐标轴
  ax.set_axis_off()
  plt.show()

  return mesh['vertex_boundary_flags'],mesh['face_boundary_flags'],mesh['vertices_clusters'],mesh['faces_clusters'],mesh['clusters_ratio'],mesh['tasks_per_cluster']

def calc_Q_for_each_vertex(mesh):     # 计算每个顶点的 Q 矩阵
  # Prepare some mesh paramenters and run on all vertices to call Q calculation   # 准备一些网格参数并遍历所有顶点以调用 Q 计算
  mesh['all_v_in_same_plane'] = np.abs(np.diff(mesh['face_plane_parameters'], axis=0)).sum() == 0
  mesh['Qs'] = []
  for v_idx in range(mesh['n_vertices']):
    Q = calc_Q_for_vertex(mesh, v_idx)
    mesh['Qs'].append(Q)

def calc_Q_for_vertex(mesh, v_idx):    # 计算顶点的 Q 矩阵   #注意此处Q矩阵计算过程，目前没有归一化即 a^2 +b^2 +c^2 = 1
  # Calculate K & Q according to eq. (2)
  Q = np.zeros((4, 4))
  for f_idx in np.where(mesh['vf_adjacency_matrix'][v_idx])[0]:
  #在这个矩阵中，每一行代表一个顶点，每一列代表一个面片。矩阵中的元素指示了对应顶点和面片之间的关系：
  #如果顶点与面片相邻，则对应元素为1。
  #如果顶点与面片不相邻，则对应元素为0。
    plane_params = mesh['face_plane_parameters'][f_idx][:, None]
    Kp = plane_params * plane_params.T
    Q += Kp
  return Q

def select_vertex_pairs(mesh,cluster_vertices):  # 选择每个类的顶点对,将不同类别的边界顶点除外
  for i in range(len(cluster_vertices)):   #类中循环
    for j in range(i + 1, len(cluster_vertices)):
      v1 = cluster_vertices[i]
      v2 = cluster_vertices[j]
      if mesh['vertex_boundary_flags'][v1] or mesh['vertex_boundary_flags'][v2]:
        continue
      edge_connection = mesh['v_adjacency_matrix'][v1, v2]
      vertices_are_very_close = ENABLE_NON_EDGE_CONTRACTION and np.linalg.norm(mesh['vertices'][v2] - mesh['vertices'][v1]) < CLOSE_DIST_TH
      if edge_connection or vertices_are_very_close:
        add_pair(mesh, v1, v2, edge_connection)

def add_pair(mesh, v1, v2, edge_connection):    # 将一对顶点添加到堆中
  # Do not use vertices on bound or non-manifold ones    # 不要使用边界或非流形的顶点(有孔洞或者不连通)，目前找的模型都是流形的
  if not ALLOW_CONTRACT_NON_MANIFOLD_VERTICES:
    if not mesh['is_watertight'] and (v1 in mesh['non_maniford_vertices'] or v2 in mesh['non_maniford_vertices']):
      return #直接返回，不添加到堆中，若ALLOW_CONTRACT_NON_MANIFOLD_VERTICES为false，mesh['is_watertight']为0
  # Add pair of indices to the heap, keys by the cost    # 将索引对按成本推入堆中
  Q = mesh['Qs'][v1] + mesh['Qs'][v2]
  new_v1_ = calc_new_vertex_position(mesh, v1, v2, Q)
  if mesh['all_v_in_same_plane']:
    cost = np.linalg.norm(mesh['vertices'][v1] - mesh['vertices'][v2]) #
  else:
    new_v1 = np.vstack((new_v1_[:, None], np.ones((1, 1))))
    cost = np.dot(np.dot(new_v1.T, Q), new_v1)[0, 0]
  if PRINT_COST:
    print('For pair: ', v1, ',', v2, ' ; the cost is: ', cost)
  heapq.heappush(mesh['pair_heap'], (cost, v1, v2, edge_connection, new_v1_))

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

def clear_heap(mesh):
  mesh['pair_heap'] = []

def update_planes_parameters_near_vertex(mesh, v):
  # Get faces near v and recalculate their plane parameters
  # 获取靠近 v 的面并重新计算它们的平面参数
  mesh_calc.calc_face_plane_parameters(mesh, must_recalc=True)

def look_for_minimum_cost_on_connected_line():        # TODO
  return None    #在连接线上寻找最小成本

def calc_new_vertex_position(mesh, v1, v2, Q):   # 计算新顶点位置
  # Calculating the new vetrex position, given 2 vertices (paragraph 4.):
  # 1. If A (to be defined below) can be inverted, use it
  # 2. If this matrix is not invertible, attempt to find the optimal vertex along the segment V1 and V2
  # 3. The new vertex will be at the midpoint
  # 计算新的顶点位置，给定两个顶点（第 4 段）：
  # 1. 如果 A（将在下面定义）可逆，使用它
  # 2. 如果这个矩阵不可逆，则尝试沿着 V1 和 V2 段找到最佳顶点
  # 3. 新的顶点将在中点处
  A = Q.copy()
  A[3] = [0, 0, 0, 1]                                 # Defined by eq. (1)
  if CALC_OPTIMUM_NEW_POINT:
    A_can_be_ineverted = np.linalg.matrix_rank(A) == 4  # TODO: bug fix!
  else:
    A_can_be_ineverted = False
  if A_can_be_ineverted:     #将顶点对的Q矩阵最后一行赋为（0,0,0,1）
    A_inv = np.linalg.inv(A)
    new_v1 = np.dot(A_inv, np.array([[0, 0, 0, 1]]).T)[:3]
    new_v1 = np.squeeze(new_v1)
  else:
    if CALC_OPTIMUM_NEW_POINT:
      new_v1 = look_for_minimum_cost_on_connected_line()
    else:
      new_v1 = None
    if new_v1 is None:     #矩阵不可逆取顶点对中点
      new_v1 = (mesh['vertices'][v1] + mesh['vertices'][v2]) / 2

  return new_v1

def clean_mesh_from_removed_items(mesh):   # 从删除的元素中清理网格
  # Remove Faces
  faces2delete = np.where(np.all(mesh['faces'] == -1, axis=1))[0]
  mesh['faces'] = np.delete(mesh['faces'], faces2delete, 0)
  mesh['faces_clusters'] = np.delete(mesh['faces_clusters'], faces2delete, 0)

  # Remove vertices and fix face indices   # 删除顶点并修复面索引
  is_to_remove = (mesh['vertices'][:, 0] == -1) + np.isnan(mesh['vertices'][:, 0])
  v_to_remove = np.where(is_to_remove)[0]
  v_to_keep   = np.where(is_to_remove == 0)[0]
  mesh['vertices'] = mesh['vertices'][v_to_keep, :]
  mesh['vertices_clusters'] = mesh['vertices_clusters'][v_to_keep]

  # 保存当前的 vertex_boundary_flags 并更新
  vertex_boundary_flags = mesh['vertex_boundary_flags']
  vertex_boundary_flags = vertex_boundary_flags[v_to_keep]
  # 保存当前的 face_boundary_flags 并更新
  face_boundary_flags = mesh['face_boundary_flags']
  face_boundary_flags = np.delete(face_boundary_flags, faces2delete, 0)
  for v_idx in v_to_remove[::-1]:
    f_to_update = np.where(mesh['faces'] > v_idx)
    mesh['faces'][f_to_update] -= 1
  # 更新顶点和面数量
  mesh['n_vertices'] = len(mesh['vertices'])
  mesh['n_faces'] = len(mesh['faces'])
  # 更新 mesh 中的边界标志
  mesh['vertex_boundary_flags'] = vertex_boundary_flags
  mesh['face_boundary_flags'] = face_boundary_flags


if __name__ == '__main__':
  #run_all()
  #run_bunny_many()
  run_one(0)      #  调用run_one函数，参数为n，即运行第n+1个模型的简化