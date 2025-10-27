import numpy as np
from scipy.spatial.transform import Rotation


def save_basis_vectors_ply(trajectory, output_file, offset=2):
    vertices = []
    triangles = []
    vertex_idx = 0
    thickness = 0.05  # Reduced thickness for better visualization

    positions = trajectory["positions"]
    rot_matrices = Rotation.from_quat(trajectory["quaternions"]).as_matrix()

    for pos, rot_mat in zip(positions, rot_matrices):
        x_end = pos + offset * rot_mat[0, :]
        y_end = pos + offset * rot_mat[1, :]
        z_end = pos + offset * rot_mat[2, :]

        # Create cylinder vertices and faces for each axis
        for i, (end, color) in enumerate(
            [(x_end, [255, 0, 0]), (y_end, [0, 255, 0]), (z_end, [0, 0, 255])]
        ):
            direction = end - pos
            direction = direction / np.linalg.norm(direction)

            # Create circle points around axis
            circle_points = []
            num_segments = 8
            for j in range(num_segments):
                angle = j * 2 * np.pi / num_segments
                circle_points.append([np.cos(angle), np.sin(angle)])

            # Create cylinder vertices
            for point in circle_points:
                perp = np.cross(direction, [0, 0, 1])
                if np.linalg.norm(perp) < 1e-5:
                    perp = np.cross(direction, [0, 1, 0])
                perp = perp / np.linalg.norm(perp)
                binormal = np.cross(direction, perp)
                vertex = pos + thickness * (point[0] * perp + point[1] * binormal)
                vertices.append([*vertex, *color])
                vertex = end + thickness * (point[0] * perp + point[1] * binormal)
                vertices.append([*vertex, *color])

            # Create triangles
            for j in range(num_segments):
                j0 = vertex_idx + j * 2
                j1 = vertex_idx + ((j + 1) % num_segments) * 2
                triangles.extend([[3, j0, j0 + 1, j1], [3, j1, j0 + 1, j1 + 1]])

            vertex_idx += num_segments * 2

    with open(output_file, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(triangles)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]} {int(v[3])} {int(v[4])} {int(v[5])}\n")

        for t in triangles:
            f.write(f"{t[0]} {t[1]} {t[2]} {t[3]}\n")

