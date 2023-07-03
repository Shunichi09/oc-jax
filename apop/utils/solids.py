import os
import pathlib
from typing import Iterable, Optional, Tuple, Union, cast


import numpy as np
import open3d as o3d


from svlr_sim.utils.randoms import rand_min_max
from svlr_sim.utils.transformations import (
    create_transformation_matrix,
    rotvec_from_two_vectors,
    rotvec_to_quat,
    transform_position,
)


def create_point_mesh(
    point: np.ndarray, radius: float = 0.1, color: Optional[np.ndarray] = None
) -> o3d.geometry.TriangleMesh:
    """create point mesh


    Args:
        point (np.ndarray): point
        radius (float): size of point, default is 0.1
        color (Optional[np.ndarray]): color, default is None


    Returns:
        o3d.geometry.TriangleMesh: point mesh
    """
    if not len(point.flatten()) == 3:
        raise ValueError

    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    point_flat = point.flatten()
    sphere_mesh.translate(point_flat.reshape((3, 1)).astype(np.float64))

    if color is not None:
        color = np.asarray(color).astype(np.float64)
        sphere_mesh.paint_uniform_color(color.reshape((3, 1)))

    return sphere_mesh


def create_coordinate_mesh(
    transformation_matrix: np.ndarray, size: float = 0.01
) -> o3d.geometry.TriangleMesh:
    """create coordinate mesh


    Args:
        transformation_matrix (np.ndarray): transformation_matrix
        size (float): size of mesh


    Returns:
        o3d.geometry.TriangleMesh: coordinate mesh
    """
    coordinate_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    coordinate_mesh.transform(transformation_matrix)
    return coordinate_mesh


def create_box_mesh(
    transformation_matrix: np.ndarray, width=1.0, height=1.0, depth=1.0
) -> o3d.geometry.TriangleMesh:
    """create box mesh, left bottom is base


    Args:
        transformation_matrix (np.ndarray): transformation_matrix
        width (float): width size of box mesh
        height (float): height size of box mesh
        depth (float): depth size of box mesh


    Returns:
        o3d.geometry.TriangleMesh: box mesh
    """
    box_mesh = o3d.geometry.TriangleMesh.create_box(width, height, depth)
    box_mesh.transform(transformation_matrix)  # left bottom is base
    return box_mesh


def create_normal_vector_mesh(
    point: np.ndarray, normal: np.ndarray, size: float = 0.05, axis: str = "z"
) -> o3d.geometry.TriangleMesh:
    """create normal coordinate mesh, axis direction (x=red, y=green, z=blue) means normal vector


    Args:
        point (np.ndarray): point
        normal (np.ndarray): normal vector
        size (float): size of coordinate mesh, default is 0.05
        axis (str): axis of normal direction, default is z


    Returns:
        o3d.geometry.TriangleMesh: normal mesh, axis direction (x=red, y=green, z=blue) means normal vector
    """
    if axis == "z":
        rotvec = rotvec_from_two_vectors(np.array([0.0, 0.0, 1.0]), normal)
    else:
        raise NotImplementedError

    transformation_mat = create_transformation_matrix(
        point, rotvec_to_quat(rotvec), rotation_type="quaternion"
    )
    normal_mesh = create_coordinate_mesh(transformation_mat, size)
    return normal_mesh


def mesh_to_point_cloud(
    mesh: o3d.geometry.TriangleMesh,
    num_points: int = 1000,
    method: str = "poisson",
    return_with_ndarray: bool = True,
    return_with_normal: bool = False,
) -> Union[
    Tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray],
    Tuple[o3d.geometry.PointCloud, np.ndarray],
    o3d.geometry.PointCloud,
]:
    """mesh to point cloud data


    Args:
        mesh (o3d.geometry.TriangleMesh): mesh
        num_points (int): number of points, default is 1000
        method (str): sample method, default is poisson
        return_with_ndarray (bool): if true, return np.ndarray points, default is True
        return_with_normal (bool): if true, return np.ndarray normal vectors, default is False,


    Returns:
        Tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray],
        Tuple[o3d.geometry.PointCloud, np.ndarray],
        o3d.geometry.PointCloud: point cloud, numpy points, numpy normal vectors
    """
    if method == "poisson":
        pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
    elif method == "uniform":
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)

    if return_with_normal:
        pcd.estimate_normals()

    if return_with_ndarray and return_with_normal:
        return pcd, np.asarray(pcd.points), np.asarray(pcd.normals)
    elif return_with_ndarray and (not return_with_normal):
        return pcd, np.asarray(pcd.points)
    elif (not return_with_ndarray) and return_with_normal:
        return pcd, np.asarray(pcd.normals)
    else:
        return pcd


def create_point_cloud_from_depth_image(
    depth: np.ndarray, scale: float, K: np.ndarray, organized: bool = False
) -> np.ndarray:
    """Generate point cloud using depth image and camera intrinsic


    Args:
        depth (np.ndarray): depth image
        scale (float): scale of depth
        K (np.array): camera_intrinsic matrix
                         [fx  0 cx]
                     K = [ 0 fy cy]
                         [ 0  0  1]
        organized (bool): whether to keep the cloud in image shape (H,W,3)


    Returns:
        cloud (np.ndarray): generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    assert (3, 3) == K.shape
    height, width = depth.shape
    xmap = np.arange(width)
    ymap = np.arange(height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / scale
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    points_x = (xmap - cx) * points_z / fx
    points_y = (ymap - cy) * points_z / fy
    cloud: np.ndarray = np.stack(
        [points_x, points_y, points_z], axis=-1
    )  # FIXME: Is this the correct way to cast ?
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud


def create_o3d_point_cloud_from_np_ndarray(
    position_array: np.ndarray, rgb_array: Optional[np.ndarray] = None
) -> o3d.geometry.PointCloud:
    """create open3d point cloud from np_ndarray


    Args:
        position_array (np.ndarray): position array
        rgb_array (np.ndarray): rgb array, optional


    Returns:
        o3d.geometry.PointCloud: point cloud of open3d
    """
    o3d_point_cloud = o3d.geometry.PointCloud()
    o3d_point_cloud.points = o3d.utility.Vector3dVector(position_array)

    if rgb_array is not None:
        if np.max(rgb_array) > 1.0:
            raise ValueError("rgb array should be range [0 1]")
        o3d_point_cloud.colors = o3d.utility.Vector3dVector(rgb_array)

    return o3d_point_cloud


def transform_mesh(
    mesh: o3d.geometry.TriangleMesh, poses: Iterable[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[o3d.geometry.TriangleMesh, ...]:
    """transform mesh by given poses


    Args:
        mesh (o3d.geometry.TriangleMesh)
        poses (Iterable[Tuple[np.ndarray, np.ndarray]]): pose, rotation should be quaternion


    Returns:
        Tuple[o3d.geometry.TriangleMesh, ...]: transformed mesh
    """
    transformed_meshes = []
    for position, rotation in poses:
        transformed_mesh = o3d.geometry.TriangleMesh(mesh)
        transformation_matrix = create_transformation_matrix(
            position, rotation, rotation_type="quaternion"
        )
        transformed_mesh.transform(transformation_matrix)
        transformed_meshes.append(transformed_mesh)
    return tuple(transformed_meshes)


def fusion_mesh(
    meshes: Iterable[o3d.geometry.TriangleMesh],
    rgb: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> o3d.geometry.TriangleMesh:
    """fusion the given meshes


    Args:
        meshed (Iterable[o3d.geometry.TriangleMesh]): meshes
        rgb (np.ndarray): color, if the mesh has color, this function does not paint with this parameters


    Returns:
        o3d.geometry.TriangleMesh: fusioned mesh
    """
    vertices = []
    triangles = []
    colors = []
    vertex_counts = 0

    for i, mesh in enumerate(meshes):
        vertex = np.array(mesh.vertices)
        triangle = np.array(mesh.triangles) + vertex_counts

        if mesh.has_vertex_colors():
            color = np.array(mesh.vertex_colors)
        else:
            color = np.tile(rgb, (vertex.shape[0], 1))

        vertices.append(vertex)
        triangles.append(triangle)
        colors.append(color)
        vertex_counts += vertex.shape[0]

    vertices = np.concatenate(vertices, axis=0)
    triangles = np.concatenate(triangles, axis=0)
    colors = np.concatenate(colors, axis=0)

    fusioned_mesh = o3d.geometry.TriangleMesh()
    fusioned_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    fusioned_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    fusioned_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return fusioned_mesh


def add_noise_to_normal(
    normal: np.ndarray, max_angle: float, min_angle: float
) -> np.ndarray:
    """add noise to normal by creating a cone, this function will add noise to z-axis


    Args:
        normal (np.ndarray): normal
        max_angle (float): max angle
        min_angle (float): min angle


    Returns:
        np.ndarray: normal with noise
    """
    if min_angle > max_angle:
        raise ValueError
    # normal size to 1
    normal = normal / np.linalg.norm(normal)
    cone_angle = rand_min_max(max_val=max_angle, min_val=min_angle)
    circle_size = np.tan(cone_angle) * 1.0  # normal size is 1

    base_angle = rand_min_max(max_val=2.0 * np.pi, min_val=0.0)
    x = circle_size * np.cos(base_angle)
    y = circle_size * np.sin(base_angle)
    z = -1.0  # move 0
    new_normal_at_origin = np.array([x, y, z])
    # transform
    rotvec = rotvec_from_two_vectors(np.array([0.0, 0.0, -1.0]), normal)
    new_normal = transform_position(
        new_normal_at_origin,
        create_transformation_matrix(
            rotation=rotvec_to_quat(rotvec), rotation_type="quaternion"
        ),
    )
    return new_normal


def vector_projection_to_plane(
    plane_a: float,
    plane_b: float,
    plane_c: float,
    vector: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """project vector to plane


    Args:
        plane_a (float): ax + by + cy + d = 0
        plane_b (float): ax + by + cy + d = 0
        plane_c (float): ax + by + cy + d = 0
        vector (np.ndarray): numpy array
        normalize (bool): if true, normalize the return vector


    Returns:
        np.ndarray: projected_vector


    Notes:
        https://en.wikipedia.org/wiki/Vector_projection
        See: Vector rejection
    """
    plane_norm_vec = np.array([plane_a, plane_b, plane_c])
    plane_norm_vec /= np.linalg.norm(plane_norm_vec)
    # vec - (vec * plane_norm) / (plane_norm**2) * plane_norm
    subs = (vector * plane_norm_vec) / np.sum(plane_norm_vec**2) * plane_norm_vec
    map_vec = vector - subs
    if normalize:
        return cast(np.ndarray, map_vec / np.linalg.norm(map_vec))
    else:
        return cast(np.ndarray, map_vec)


def point_from_depth(depth_data, camera_intrinsic, camera_extrinsic, bbox, scale=1.0):
    point_cloud = create_point_cloud_from_depth_image(
        depth_data, scale=scale, K=camera_intrinsic, organized=True
    )  # (h*w*3)
    min_width, min_height, width, height = bbox
    max_width = min_width + width
    max_height = min_height + height
    height_index = np.array(
        [min_height, max_height, min_height, max_height], dtype=np.int64
    )
    width_index = np.array([min_width, min_width, max_width, max_width], dtype=np.int64)
    center_point = np.mean(point_cloud[height_index, width_index, :])
    point = transform_position(center_point, np.linalg.inv(camera_extrinsic))
    return point


def farthest_point_sample(
    point_cloud: np.ndarray, num_samples: int, init_idx: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """farthest point sample


    Args:
        point (np.ndarray): pointcloud data, shape (num_points, 3)
        num_samples (np.ndarray): number of samples
        init_idx (Optional[int]): = initial index


    Return:
        Tuple[np.ndarray, np.ndarray]: sampled pointcloud and index
    """
    num_point, dim = point_cloud.shape
    if dim != 3:
        raise ValueError

    sample_idx = np.zeros((num_samples,), dtype=np.int32)
    distances = np.ones((num_point,)) * 1e7
    if init_idx is None:
        farthest_idx = np.random.randint(0, num_point)
    else:
        farthest_idx = init_idx

    for i in range(num_samples):
        sample_idx[i] = farthest_idx
        farthest_point = point_cloud[farthest_idx, :]
        distance = np.sum((point_cloud - farthest_point) ** 2, -1)
        mask = distance < distances
        distances[mask] = distance[mask]
        farthest_idx = np.argmax(distances, -1)

    sampled_point_cloud = point_cloud[sample_idx.astype(np.int32)]
    return sampled_point_cloud, sample_idx.astype(np.int32)


def compute_index_cropped_point_cloud(
    o3d_pc: o3d.geometry.PointCloud,
    min_vector: np.ndarray,
    max_vector: np.ndarray,
) -> np.ndarray:
    if not all(min_vector <= max_vector):
        raise ValueError
    points = np.array(o3d_pc.points, dtype=np.float32)
    is_in = np.logical_and(min_vector <= points, points <= max_vector)
    in_idx = np.where(np.all(is_in, axis=1))[0]
    return in_idx


def load_mesh(stl_path: Union[str, pathlib.Path]) -> o3d.geometry.TriangleMesh:
    """load mesh


    Args:
        stl_path (str): stl file path


    Returns:
        o3d.geometry.TriangleMesh: loaded mesh
    """
    if not os.path.exists(stl_path):
        raise ValueError("{} can not find".format(stl_path))
    # see: http://www.open3d.org/docs/release/python_api/open3d.visualization.rendering.TriangleMeshModel.html
    stl_data = o3d.io.read_triangle_model(str(stl_path))
    mesh = stl_data.meshes[0].mesh
    # mesh.meshes this is list of MeshInfo
    assert len(stl_data.meshes) == 1, "Contain more than two meshes !!"
    return mesh
