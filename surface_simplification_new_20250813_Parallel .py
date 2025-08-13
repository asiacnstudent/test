import time
import os
import copy
import heapq
import open3d as o3d
import sys

from scipy.optimize import minimize_scalar
from tqdm import tqdm
import numpy as np
import pylab as plt
#import trimesh
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
from sklearn.cluster import SpectralClustering
from scipy.sparse.csgraph import laplacian
from sklearn.metrics.pairwise import pairwise_distances


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
Boundary = True#是否显示聚类结果的边界面片
USE_TRIANGLE_QUALITY = False  # True/False 是否在 cost 计算中考虑三角形质量因子


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
  colors = plt.cm.get_cmap('viridis', n_clusters)
  boundary_color = 'red'  # 边界面片的特殊颜色

  # 绘制每个聚类的面片
  for i in range(n_clusters):
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
  # 设置坐标轴范围
  ax.set_xlim([np.min(vertices[:, 0]), np.max(vertices[:, 0])])
  ax.set_ylim([np.min(vertices[:, 1]), np.max(vertices[:, 1])])
  ax.set_zlim([np.min(vertices[:, 2]), np.max(vertices[:, 2])])
  # 隐藏坐标轴
  ax.set_axis_off()
  # 设置统一视角
  ax.view_init(elev=90, azim=-90)
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
    new_mesh = io_off_model.read_off(f'output_meshes/{mesh["name"].split(".")[0]}_simplified_{n_vertices_to_merge}.off')
    plot_off(new_mesh,mesh_simplified)
  #Hausdorff距离计算
  mesh_original = o3d.io.read_triangle_mesh(f'meshes/{mesh["name"].split(".")[0]}.off')
  mesh_simplified = o3d.io.read_triangle_mesh(f'output_meshes/{mesh["name"].split(".")[0]}_simplified_{n_vertices_to_merge}.off')
  # 提取网格的顶点信息
  vertices_original = np.asarray(mesh_original.vertices)
  vertices_simplified = np.asarray(mesh_simplified.vertices)
  # 计算原始网格到简化网格的定向 Hausdorff 距离
  hausdorff_distance_1 = distance.directed_hausdorff(vertices_original, vertices_simplified)[0]
  # 计算简化网格到原始网格的定向 Hausdorff 距离
  hausdorff_distance_2 = distance.directed_hausdorff(vertices_simplified, vertices_original)[0]
  # 取两个方向的最大值作为 Hausdorff 距离
  hausdorff_distance = max(hausdorff_distance_1, hausdorff_distance_2)
  print(f"Hausdorff Distance Front: {hausdorff_distance_1} ，Hausdorff Distance Back: {hausdorff_distance_2}")
  print(f"Hausdorff Distance: {hausdorff_distance}")
  export_experiment_log_to_csv("task_allocation_log.csv")
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
    mesh_fns = [['meshes/bunny2.off',        2000],
                ['meshes/cat.off',           5000],
                ['meshes/dinosaur.off',       500],
                ['meshes/horse_simplified_40000_simplified_4000.off',       2000],
                ['meshes/Arma_simplified_10000.off',  5000],
                ['meshes/person_0067.off',    1200],
                ['meshes/person_0004.off',    1000],
                ['meshes/ankylosaurus.off',   2000],
                ['meshes/phands.off',         2000]
                ]
    n_vertices_to_merge = mesh_fns[idx][1]
    mesh = io_off_model.read_off(mesh_fns[idx][0], verbose=True)
    mesh['name'] = os.path.split(mesh_fns[idx][0])[-1]
  return mesh, n_vertices_to_merge

#网格简化
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import threading
import time
# ...existing code...

def simplify_mesh(mesh_orig, n_vertices_to_merge):
    mesh = copy.deepcopy(mesh_orig)  # 深拷贝避免修改原网格
    tb = time.time()
    mesh_preprocess(mesh)
    mesh['pair_heap'] = []  # 初始化堆
    calc_Q_for_each_vertex(mesh)  # 计算每个顶点的 Q 矩阵
    compute_gaussian_curvature(mesh)  # 高斯曲率用于聚类
    perform_clustering_and_task_assignment(mesh, n_vertices_to_merge)  # 聚类+任务分配
    print('Initial time:', time.time() - tb)

    cluster_stats = mesh['cluster_stats']
    vertex_clusters = mesh['vertices_clusters']
    n_clusters = len(cluster_stats)

    # --- 并行聚类内部简化 ---
    import threading
    from concurrent.futures import ThreadPoolExecutor
    progress_vals = [0 for _ in range(n_clusters)]
    progress_locks = [threading.Lock() for _ in range(n_clusters)]
    task_counts = [stat['assigned_tasks'] for stat in cluster_stats.values()]

    def print_progress():
        bar_strs = []
        for i in range(n_clusters):
            with progress_locks[i]:
                done = progress_vals[i]
                total = task_counts[i]
            percent = int(100 * done / total) if total > 0 else 100
            bar = f"Cluster {i}: [{done:3d}/{total:3d}] {percent:3d}%"
            bar_strs.append(bar)
        print('\r' + ' | '.join(bar_strs), end='', flush=True)

    def cluster_simplify_worker(cluster_idx, stat, idx):
        if stat['frozen']:
            return 0
        cluster_vertices = stat['vertex_indices']
        task_count = stat['assigned_tasks']
        select_vertex_pairs(mesh, cluster_vertices)

        completed = 0
        for _ in range(task_count):
            contract_best_pair(mesh)
            completed += 1
            with progress_locks[idx]:
                progress_vals[idx] = completed
            if mesh['n_faces'] <= MINIMUM_NUMBER_OF_FACES:
                break
        clear_heap(mesh)
        return completed

    tb = time.time()
    with ThreadPoolExecutor(max_workers=n_clusters) as executor:
        futures = []
        for i, (cluster_idx, stat) in enumerate(cluster_stats.items()):
            futures.append(executor.submit(cluster_simplify_worker, cluster_idx, stat, i))
        # 主线程定时刷新进度
        while True:
            all_done = all(f.done() for f in futures)
            print_progress()
            if all_done:
                break
            time.sleep(1)  # 刷新频率可调
    print()  # 换行，避免进度条覆盖后续输出

    # 统计聚类内部实际完成的合并数
    total_simplified = sum(stat['actual_simplified'] for stat in cluster_stats.values())
    leftover = n_vertices_to_merge - total_simplified
    # if leftover > 0:
    #     print(f"\n{leftover} tasks remaining, will process boundary vertex pairs...")
    #     n = mesh['n_vertices']
    #     clusters = mesh['vertices_clusters']
    #     boundary_vertex_indices = np.where(mesh['vertex_boundary_flags'])[0].tolist()
    #     select_vertex_pairs(mesh, boundary_vertex_indices, allow_boundary=True)
    #     merged = 0
    #     for _ in range(leftover):
    #         contract_best_pair(mesh)
    #         merged += 1
    #         if mesh['n_faces'] <= MINIMUM_NUMBER_OF_FACES:
    #             break
    #     print(f"Boundary vertex pair simplification finished, merged {merged} pairs.")

    clean_mesh_from_removed_items(mesh)
    print('Iteration time:', time.time() - tb)
    print("Mesh simplification completed.")
    print_final_cluster_summary(mesh)
    return mesh
# ...existing




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





n_clusters = 8  #聚类中心数量，全局变量
# --- 高斯曲率谱聚类 + 自适应任务分配 + 动态任务重分配核心逻辑 ---

def visualize_mesh_clusters(mesh):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    vertices = mesh['vertices']
    faces = mesh['faces']
    face_labels = mesh['faces_clusters']
    face_boundary_flags = mesh['face_boundary_flags']
    num_clusters = len(np.unique(face_labels))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.get_cmap('viridis', num_clusters)
    for i in range(num_clusters):
        cluster_faces = faces[face_labels == i]
        face_vertices = vertices[cluster_faces]
        poly3d = [face_vertices[j] for j in range(face_vertices.shape[0])]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors=colors(i), linewidths=0.1, edgecolors='k', alpha=0.7))

    # 标记边界面片
    boundary_faces = faces[face_boundary_flags]
    boundary_face_vertices = vertices[boundary_faces]
    boundary_poly3d = [boundary_face_vertices[j] for j in range(boundary_face_vertices.shape[0])]
    ax.add_collection3d(
        Poly3DCollection(boundary_poly3d, facecolors='red', linewidths=0.1, edgecolors='k', alpha=0.7))

    ax.set_xlim([np.min(vertices[:, 0]), np.max(vertices[:, 0])])
    ax.set_ylim([np.min(vertices[:, 1]), np.max(vertices[:, 1])])
    ax.set_zlim([np.min(vertices[:, 2]), np.max(vertices[:, 2])])
    ax.set_axis_off()
    # 设置统一视角
    ax.view_init(elev=90, azim=-90)
    plt.show()

# --- 高斯曲率谱聚类 + 自适应任务分配 + 动态任务重分配核心逻辑 ---



# 实验记录器：用于记录任务分配与误差等统计信息
EXPERIMENT_LOG = []

# --- 高斯曲率谱聚类 + 全局归一化低曲率统计 + 混合任务分配（考虑低曲率占比 + 顶点总数 + 全局曲率差异）---



def normalize_curvature(cluster_curvature, min_curvature, max_curvature):
    """将高斯曲率归一化到 [0, 1] 区间"""
    return (np.abs(cluster_curvature) - min_curvature) / (max_curvature - min_curvature + 1e-8)


def perform_clustering_and_task_assignment(mesh, n_vertices_to_merge, n_clusters=8):
    from sklearn.cluster import KMeans
    from scipy.sparse.linalg import eigsh
    import numpy as np

    vertices = mesh['vertices']
    faces = mesh['faces']
    gaussian_curvature = mesh['gaussian_curvature']

    # 全局归一化高斯曲率
    K_abs = np.abs(gaussian_curvature)
    K_min, K_max = K_abs.min(), K_abs.max()
    K_norm = normalize_curvature(gaussian_curvature, K_min, K_max)
    mesh['normalized_gaussian_curvature'] = K_norm

    # 构建谱聚类用的权重矩阵
    v_adjacency_matrix = mesh['v_adjacency_matrix']
    sigma = 0.5
    neighbor_radius = 0.2
    vertex_distances = np.linalg.norm(vertices[:, None, :] - vertices[None, :, :], axis=-1)
    W = np.exp(-np.square(np.subtract.outer(K_norm, K_norm)) / sigma ** 2) * (
        vertex_distances < neighbor_radius) * v_adjacency_matrix
    W = W ** 0.5

    D = np.diag(W.sum(axis=1))
    L = D - W
    eigvals, eigvecs = eigsh(L, k=n_clusters, which='SM')
    U = eigvecs[:, :n_clusters]
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(U)
    mesh['vertices_clusters'] = labels
    face_labels = np.array([np.bincount(labels[face]).argmax() for face in faces])
    mesh['faces_clusters'] = face_labels

    # 边界标志
    vertex_boundary = np.zeros(len(vertices), dtype=bool)
    face_boundary = np.zeros(len(faces), dtype=bool)
    for f_id, face in enumerate(faces):
        if -1 in face:
            continue
        cids = labels[face]
        if len(set(cids)) > 1:
            face_boundary[f_id] = True
            for v in face:
                vertex_boundary[v] = True
    mesh['vertex_boundary_flags'] = vertex_boundary
    mesh['face_boundary_flags'] = face_boundary

    # 全局低曲率门槛
    low_curv_threshold = np.percentile(K_norm, 25)

    cluster_stats = {}
    low_curv_counts = {}
    vertex_counts = {}

    # 先初始化 cluster 信息
    for cid in range(n_clusters):
        v_indices = np.where(labels == cid)[0]
        f_indices = np.where(face_labels == cid)[0]
        K_cluster = K_norm[v_indices]
        low_mask = K_cluster < low_curv_threshold

        cluster_stats[cid] = {
            'vertex_indices': v_indices,
            'face_indices': f_indices,
            'vertex_count': len(v_indices),
            'face_count': len(f_indices),
            'mean_abs_K': np.mean(K_abs[v_indices]) + 1e-6,
            'K_range': K_abs[v_indices].ptp(),
            'low_curv_count': int(np.sum(low_mask)),
            'assigned_tasks': 0,
            'actual_simplified': 0,
            'cumulative_error': 0.0,
            'frozen': False,
            'error_threshold': 0.0
        }
        low_curv_counts[cid] = cluster_stats[cid]['low_curv_count']
        vertex_counts[cid] = cluster_stats[cid]['vertex_count']

    total_low = sum(low_curv_counts.values())
    total_v = sum(vertex_counts.values())
    alpha = 0.7  # 低曲率占比权重

    # ---------- 修复点：为每个 cluster 生成候选对并计算平均 cost ----------
    cluster_avg_qem_costs = {}
    mesh['pair_heap'] = []  # 确保空堆
    for cid, stat in cluster_stats.items():
        select_vertex_pairs(mesh, stat['vertex_indices'])  # 只考虑该 cluster 内部
        costs = [cost for (cost, v1, v2, _, _) in mesh['pair_heap']
                 if labels[v1] == cid and labels[v2] == cid]
        cluster_avg_qem_costs[cid] = np.mean(costs) if costs else 1e-4
    mesh['pair_heap'] = []  # 清空，防止旧数据干扰后续
    # -------------------------------------------------------------------

    for cid, stat in cluster_stats.items():
        w_low = low_curv_counts[cid] / total_low if total_low > 0 else 0
        w_v = vertex_counts[cid] / total_v if total_v > 0 else 0
        w_i = alpha * w_low + (1 - alpha) * w_v
        S_i = int(w_i * n_vertices_to_merge)
        stat['assigned_tasks'] = S_i
        scaling_factor = 10.0  # 你后面可以再调
        stat['error_threshold'] = S_i * cluster_avg_qem_costs[cid] * scaling_factor

        EXPERIMENT_LOG.append({
            'cluster_id': cid,
            'mean_abs_K': stat['mean_abs_K'],
            'K_range': stat['K_range'],
            'vertex_count': stat['vertex_count'],
            'low_curvature_count': stat['low_curv_count'],
            'assigned_tasks': S_i
        })

    mesh['cluster_stats'] = cluster_stats
    mesh['actual_merge_count'] = 0

    print("--- Cluster Task Summary ---")
    for cid, stat in cluster_stats.items():
        print(f"Cluster {cid:2d} | Color ID: {cid:2d} | Vertices: {stat['vertex_count']:4d} | Faces: {stat['face_count']:4d} "
              f"| Assigned tasks: {stat['assigned_tasks']:4d} | Mean |K|: {stat['mean_abs_K']:.6f} "
              f"| error_threshold: {stat['error_threshold']:.6f}")

    if VISUAL:
        visualize_mesh_clusters(mesh)



def export_experiment_log_to_csv(path="experiment_log.csv"):
    import csv
    if not EXPERIMENT_LOG:
        print("No experiment log recorded.")
        return
    with open(path, 'w', newline='') as csvfile:
        fieldnames = list(EXPERIMENT_LOG[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in EXPERIMENT_LOG:
            writer.writerow(row)
    print(f"Experiment log saved to {path}")



def print_final_cluster_summary(mesh):
    print("\n--- Final Cluster Simplification Summary ---")
    for cid, stat in mesh['cluster_stats'].items():
        print(f"Cluster {cid:2d} | Assigned: {stat['assigned_tasks']:4d} | Simplified: {stat['actual_simplified']:4d} | Frozen: {stat['frozen']} | Cumulative Error: {stat['cumulative_error']:.6f}")
    print(f"\nTotal actual merged pairs: {mesh['actual_merge_count']}")





# --- 是否达到阈值进行冻结判断，嵌入每次 contract_pair() 中 ---
def update_cluster_after_merge(mesh, v1, v2, cost):
    cid = mesh['vertices_clusters'][v1]
    stat = mesh['cluster_stats'][cid]
    if stat['frozen']:
        return
    if stat['actual_simplified'] >= stat['assigned_tasks']:
        return

    stat['actual_simplified'] += 1
    mesh['actual_merge_count'] += 1  # 关键：全局统计
    stat['cumulative_error'] += cost

    if (stat['actual_simplified'] >= stat['assigned_tasks'] or
        stat['cumulative_error'] >= stat['error_threshold']):
        stat['frozen'] = True







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

def select_vertex_pairs(mesh, cluster_vertices, allow_boundary=False):
    for i in range(len(cluster_vertices)):
        for j in range(i + 1, len(cluster_vertices)):
            v1 = cluster_vertices[i]
            v2 = cluster_vertices[j]
            if not allow_boundary:
                if mesh['vertex_boundary_flags'][v1] or mesh['vertex_boundary_flags'][v2]:
                    continue  # 在聚类内部跳过边界顶点
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
    # 计算三角形质量因子 Q_t（取涉及 v1, v2 的三角形最差的 Q_t）
  # 如果开关打开，叠加三角形质量因子
  if USE_TRIANGLE_QUALITY:
      incident_faces = np.where((mesh['faces'] == v1) | (mesh['faces'] == v2))[0]
      min_quality = 1  # Q_t 最大值为 1，初始值设为最大
      for f in incident_faces:
        if -1 in mesh['faces'][f]:  # 跳过已删除的面
          continue
        Q_t = compute_triangle_quality(mesh['faces'][f], mesh['vertices'])
        min_quality = min(min_quality, Q_t)  # 选最小值，优先折叠最坏质量的三角形
      # 计算最终 cost
      alpha = 0.8  # QEM 误差权重
      beta = 0.2  # 质量因子权重（调整后可提高狭长三角形的折叠优先级）
      cost = alpha * cost + beta * (1 - min_quality)   #加入三角形质量因子的代价计算公式

  if PRINT_COST:
    print('For pair: ', v1, ',', v2, ' ; the cost is: ', cost)
  heapq.heappush(mesh['pair_heap'], (cost, v1, v2, edge_connection, tuple(new_v1_)))

def compute_triangle_quality(triangle, vertices):
  """
  计算三角形的质量因子 Q_t
  :param triangle: (v1, v2, v3) 三角形顶点索引
  :param vertices: 顶点坐标数组
  :return: 质量因子 Q_t
  """
  v1, v2, v3 = vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]

  # 计算三角形面积 A
  edge1 = v2 - v1
  edge2 = v3 - v1
  A = 0.5 * np.linalg.norm(np.cross(edge1, edge2))

  # 计算三角形边长
  l1 = np.linalg.norm(v2 - v1)
  l2 = np.linalg.norm(v3 - v2)
  l3 = np.linalg.norm(v1 - v3)

  # 避免除零错误
  perimeter_square = l1 ** 2 + l2 ** 2 + l3 ** 2
  if perimeter_square == 0:
    return 0  # 退化三角形（所有点重合）

  # 计算 Q_t
  Q_t = (4 * np.sqrt(3) * A) / perimeter_square
  return Q_t

# --- 集成进折叠流程 ---

def contract_best_pair(mesh):
    import heapq
    if len(mesh['pair_heap']) == 0:
        return

    cost, v1, v2, is_edge, new_v1 = heapq.heappop(mesh['pair_heap'])

    cid_v1 = mesh['vertices_clusters'][v1]
    cid_v2 = mesh['vertices_clusters'][v2]
    if mesh['cluster_stats'][cid_v1]['frozen'] or mesh['cluster_stats'][cid_v2]['frozen']:
        return

    mesh['vertices'][v1] = new_v1
    mesh['vertices'][v2] = [-1, -1, -1]
    mesh['v_adjacency_matrix'][v2, :] = False
    mesh['v_adjacency_matrix'][:, v2] = False

    if is_edge:
        all_v2_faces = np.where(mesh['vf_adjacency_matrix'][v2])[0]
        mesh['vf_adjacency_matrix'][v2, :] = False
        for f in all_v2_faces:
            if v1 in mesh['faces'][f]:
                mesh['faces'][f] = [-1, -1, -1]
                mesh['vf_adjacency_matrix'][:, f] = False
            else:
                v2_idx = np.where(mesh['faces'][f] == v2)[0]
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

    if mesh.get('SELF_CHECKING', False):
        check_mesh(mesh)

    mesh['pair_heap'] = [p for p in mesh['pair_heap'] if v1 not in [p[1], p[2]] and v2 not in [p[1], p[2]]]

    if mesh.get('CALC_OPTIMUM_NEW_POINT', True):
        update_planes_parameters_near_vertex(mesh, v1)
        calc_Q_for_vertex(mesh, v1)

    update_cluster_after_merge(mesh, v1, v2, cost)

    for v2_ in range(mesh['n_vertices']):
        if v1 == v2_:
            continue
        if mesh['vertices'][v2_][0] == -1:
            continue
        edge_connection = mesh['v_adjacency_matrix'][v1, v2_]
        vertices_are_very_close = mesh.get('ENABLE_NON_EDGE_CONTRACTION', False) and np.linalg.norm(mesh['vertices'][v2_] - mesh['vertices'][v1]) < mesh.get('CLOSE_DIST_TH', 1e-3)
        if edge_connection or vertices_are_very_close:
            add_pair(mesh, v1, v2_, edge_connection)


def clear_heap(mesh):
  mesh['pair_heap'] = []

def update_planes_parameters_near_vertex(mesh, v):
  # Get faces near v and recalculate their plane parameters
  # 获取靠近 v 的面并重新计算它们的平面参数
  mesh_calc.calc_face_plane_parameters(mesh, must_recalc=True)

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