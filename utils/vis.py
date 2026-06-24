
import plotly.graph_objs as go
import numpy as np
import matplotlib.pyplot as plt



def points_to_voxel_mesh(points, voxel_size=0.005):
    corners = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                        [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]]) * voxel_size
    vertices_cube = points[:, None] + corners
    vertices = vertices_cube.reshape([-1, 3])
    
    idx = np.arange(len(points)).reshape(-1, 1) * 8 + np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    # create 12 triangles for each cube
    faces_cube = np.array([[0, 3, 1, 0, 2, 3], [4, 7, 6, 4, 5, 7], [0, 5, 4, 0, 1, 5], [2, 7, 3, 2, 6, 7], [0, 6, 2, 0, 4, 6], [1, 7, 5, 1, 3, 7]])
    # reverse the idx order to flip the normal for viser
    # faces_cube = np.array([[0, 1, 3, 0, 3, 2], [4, 6, 7, 4, 7, 5], [0, 4, 5, 0, 5, 1], [2, 3, 7, 2, 7, 6], [0, 2, 6, 0, 6, 4], [1, 5, 7, 1, 7, 3]])
    faces = idx[:, faces_cube].reshape(-1, 3)
    # 12 edges for each voxel (cube)
    edges_cube = np.array([
        [0, 1], [1, 3], [3, 2], [2, 0],  # bottom face
        [4, 5], [5, 7], [7, 6], [6, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ])
    edges = idx[:, edges_cube].reshape(-1, 2)

    return vertices, faces, edges

def draw_mesh(vertices, faces, edges=None, name='mesh'):
    mesh = go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], 
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            flatshading=True,
            intensity=color_code_point_cloud(vertices)['height_normalized'],
            # vertexcolor=(color_code_point_cloud(vertices)*255).astype(np.uint8),
            name=name,
            showlegend=True if name else None,
            showscale=False,
            )
    if edges is not None:
        # Vectorized edge line construction
        starts = vertices[edges[:, 0]]
        ends = vertices[edges[:, 1]]
        # Interleave start and end points, insert NaNs for breaks
        edges_xyz = np.empty((edges.shape[0] * 3, 3), dtype=vertices.dtype)
        edges_xyz[0::3] = starts
        edges_xyz[1::3] = ends
        edges_xyz[2::3] = np.nan  # Separator for plotly
        edges_x = edges_xyz[:, 0]
        edges_y = edges_xyz[:, 1]
        edges_z = edges_xyz[:, 2]

        # Plot edges
        edges = go.Scatter3d(
            x=edges_x,
            y=edges_y,
            z=edges_z,
            mode='lines',
            line=dict(color='black', width=2),
            name=f'{name}_edges',
            showlegend=True if name else None,
        )
        return mesh, edges
    else:
        return mesh
    
# region PointsView
def color_code_point_cloud(points):
    """
    对点云进行颜色编码
    
    参数:
    points: N x 3 的NumPy数组，每行表示一个点的(x, y, z)坐标
    
    返回:
    colors: N x 3 的NumPy数组，表示每个点的RGB颜色
    """
    # 提取高度信息（z轴）
    heights = points[:, 2]
    h_min = np.percentile(heights, 5)  # 最小高度
    h_max = np.percentile(heights, 95)  # 最大高度
    # 标准化高度到[0, 1]范围
    height_normalized = (heights - h_min) / (h_max - h_min)
    height_normalized = np.clip(height_normalized, 0, 1)  # 确保在[0, 1]范围内
    
    # 创建颜色映射
    # 使用热图颜色映射：低高度为蓝色，中等高度为绿色，高高度为红色
    colors = plt.cm.jet(height_normalized)[:, :3]
    
    return {'colors': colors, 'height_normalized': height_normalized}
