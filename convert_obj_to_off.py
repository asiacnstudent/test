def read_obj(filename):
    vertices = []
    faces = []

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':  # 顶点
                vertices.append(parts[1:4])  # 取 x, y, z
            elif parts[0] == 'f':  # 面
                face = [int(idx.split('/')[0]) - 1 for idx in parts[1:]]  # 处理索引
                faces.append(face)

    return vertices, faces


def write_off(vertices, faces, output_filename):
    with open(output_filename, 'w') as file:
        file.write("OFF\n")
        file.write(f"{len(vertices)} {len(faces)} 0\n")  # 0 为边的数量
        for vertex in vertices:
            file.write(" ".join(vertex) + "\n")
        for face in faces:
            file.write("3 " + " ".join(map(str, face)) + "\n")


def convert_obj_to_off(input_filename, output_filename):
    vertices, faces = read_obj(input_filename)
    write_off(vertices, faces, output_filename)


# 示例用法
input_filename = '/Mesh Simplification via Clustering-Guided Adaptive Edge Collapsing/meshes/fandisk.obj'
output_filename = '/Mesh Simplification via Clustering-Guided Adaptive Edge Collapsing/meshes/fandisk.off'  # 输出的 OFF 文件名
convert_obj_to_off(input_filename, output_filename)
