#make cylinder to visualize B-Spline curve
import numpy as np
import json
import os
from sklearn.decomposition import PCA
from scipy.interpolate import splprep, splev
import open3d as o3d


def load_obj_vertices(path):
    """Load vertex only (v x y z ...) from OBJ."""
    vertices = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z])
    return np.array(vertices)


def load_obj_mesh(path):
    """Load full obj as Open3D mesh."""
    mesh = o3d.io.read_triangle_mesh(path)
    mesh.compute_vertex_normals()
    return mesh


# ----------- 파일 입력 -----------
obj_path = input("OBJ 파일 경로: ").strip()
json_path = input("JSON 파일 경로: ").strip()

if not os.path.exists(obj_path):
    raise FileNotFoundError(obj_path)
if not os.path.exists(json_path):
    raise FileNotFoundError(json_path)

# ----------- OBJ 로딩 -----------
vertices = load_obj_vertices(obj_path)
mesh = load_obj_mesh(obj_path)
print("OBJ vertex 개수:", len(vertices))

# ----------- JSON 로딩 -----------
with open(json_path, "r") as f:
    data = json.load(f)

instances = np.array(data["instances"])
print("instances 개수:", len(instances))

# ----------- instance별 centroid 계산 -----------
centroids = {}
unique_ids = sorted(set(instances) - {0})

for inst_id in unique_ids:
    idx = np.where(instances == inst_id)[0]
    pts = vertices[idx]
    centroids[inst_id] = pts.mean(axis=0)

print("치아 centroid 개수:", len(centroids))

# ----------- PCA 2D projection + spline fitting -----------
start_id = min(centroids, key=lambda k: centroids[k][0])

current = start_id
visited = {current}
ordered_ids = [current]

while len(visited) < len(centroids):
    next_id = min(
        (k for k in centroids if k not in visited),
        key=lambda j: np.linalg.norm(centroids[current] - centroids[j])
    )
    ordered_ids.append(next_id)
    visited.add(next_id)
    current = next_id

C = np.vstack([centroids[i] for i in ordered_ids])

pca = PCA(n_components=2)
C_2d = pca.fit_transform(C)

x, y = C_2d[:, 0], C_2d[:, 1]
tck, _ = splprep([x, y], s=5)
u_new = np.linspace(0, 1, 200)
x_new, y_new = splev(u_new, tck)

curve_2d = np.vstack([x_new, y_new]).T
curve_3d = pca.inverse_transform(curve_2d)

# ----------- OBJ 파일로 curve 저장 -----------
curve_obj_path = "arch_curve.obj"
with open(curve_obj_path, 'w') as f:
    for v in curve_3d:
        f.write(f"v {v[0]} {v[1]} {v[2]}\n")
    for i in range(len(curve_3d)-1):
        f.write(f"l {i+1} {i+2}\n")

print("아치 곡선 OBJ 저장 완료 →", curve_obj_path)

# ----------- centroid & curve JSON 저장 -----------
with open("centroids.json", "w") as f:
    json.dump({str(k): v.tolist() for k, v in centroids.items()}, f, indent=4)

with open("curve.json", "w") as f:
    json.dump(curve_3d.tolist(), f, indent=4)

print("centroid & curve JSON 저장 완료.")


# ----------- Cylinder 생성 함수 -----------
def create_cylinder_between(p1, p2, radius=0.4, resolution=20, color=[0,1,0]):  #---------- fitting 곡선 두께 조절: radius
    p1 = np.array(p1)
    p2 = np.array(p2)
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length == 0:
        return None

    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=resolution)
    cylinder.compute_vertex_normals()
    cylinder.paint_uniform_color(color)

    direction_norm = direction / length
    z_axis = np.array([0, 0, 1])

    v = np.cross(z_axis, direction_norm)
    c = np.dot(z_axis, direction_norm)
    s = np.linalg.norm(v)

    if s != 0:
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    else:
        R = np.eye(3)

    cylinder.rotate(R, center=np.array([0, 0, 0]))
    cylinder.translate(p1)

    return cylinder


# -------------------------------
# ---- 시각화 준비 --------------
# -------------------------------

geometries = []

# Mesh → PointCloud로 변환하여 반투명 효과 구현
pcd = mesh.sample_points_uniformly(number_of_points=50000)
pcd.paint_uniform_color([0.8, 0.8, 0.8])
geometries.append(pcd)

# centroid spheres
for c in centroids.values():
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5) # ------- centroid 크기 조절
    sphere.paint_uniform_color([1, 0, 0])
    sphere.translate(c)
    geometries.append(sphere)

# thick arch curve cylinders
curve_cylinders = []
for i in range(len(curve_3d)-1):
    cyl = create_cylinder_between(curve_3d[i], curve_3d[i+1])
    if cyl:
        curve_cylinders.append(cyl)

geometries.extend(curve_cylinders)


# -------------------------------
# ---- Viewer 실행 --------------
# -------------------------------

print("Open3D PointCloud + Thick Curve 시각화 시작...")

vis = o3d.visualization.Visualizer()
vis.create_window("Dental Arch Fitting (Point Mesh)", width=1200, height=900)

for g in geometries:
    vis.add_geometry(g)

# point cloud size 설정
opt = vis.get_render_option()
opt.point_size = 1.2
opt.mesh_show_back_face = True

vis.run()
vis.destroy_window()
