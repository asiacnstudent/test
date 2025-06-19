def read_off(filename):
    vertices = []
    faces = []

    with open(filename, 'r') as file:
        # 跳过文件头
        file.readline()  # OFF
        counts = file.readline().strip().split()
        num_vertices, num_faces, _ = map(int, counts)  # 读取顶点和面数量

        for _ in range(num_vertices):
            vertex = file.readline().strip().split()
            vertices.append(vertex)  # 顶点坐标

        for _ in range(num_faces):
            face = file.readline().strip().split()
            faces.append([int(idx) for idx in face[1:]])  # 读取面，跳过第一个数字

    return vertices, faces


def write_obj(vertices, faces, output_filename):
    with open(output_filename, 'w') as file:
        for vertex in vertices:
            file.write("v " + " ".join(vertex) + "\n")  # 写入顶点
        for face in faces:
            file.write("f " + " ".join(str(idx + 1) for idx in face) + "\n")  # 写入面，索引从1开始


def convert_off_to_obj(input_filename, output_filename):
    vertices, faces = read_off(input_filename)
    write_obj(vertices, faces, output_filename)


# 示例用法
input_filename = '/Mesh Simplification via Clustering-Guided Adaptive Edge Collapsing/meshes/horse_simplified_40000.off'
output_filename = '/Mesh Simplification via Clustering-Guided Adaptive Edge Collapsing/meshes/horse_simplified_40000.obj'  # 输出的 OBJ 文件名
convert_off_to_obj(input_filename, output_filename)
