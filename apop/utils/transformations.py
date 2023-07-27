from typing import List, Tuple, Union, cast

import numpy as np
from scipy.spatial.transform import Rotation


def create_transformation_matrix(
    translation: Union[List[float], np.ndarray] = np.zeros(3),
    rotation: Union[List[float], np.ndarray] = np.eye(3),
    rotation_type: str = "matrix",
) -> np.ndarray:
    """create transformation matrix

    Args:
        translation (Union[List[float], np.ndarray]): translation vector, default is np.zeros(3)
        rotation (Union[List[float], np.ndarray]): rotation, matrix or quaternion are supported, default is np.eye(3)
        rotation_type (str): rotation type, default is matrix
    Returns:
        np.ndarray: transformation mat
    """
    translation = np.array(translation)
    rotation = np.array(rotation)

    transformation_mat = np.eye(4)
    if rotation_type == "matrix":
        transformation_mat[:3, :3] = rotation.copy()
    elif rotation_type == "quaternion":
        transformation_mat[:3, :3] = quat_to_matrix(rotation)
    else:
        raise ValueError

    transformation_mat[:-1, -1] = translation.copy()
    return transformation_mat


def extract_position(transformation_matrix: np.ndarray) -> np.ndarray:
    if len(transformation_matrix.shape) == 3:
        return transformation_matrix[:, :3, -1]
    elif len(transformation_matrix.shape) == 2:
        return transformation_matrix[:3, -1]
    else:
        raise ValueError


def extract_rotation(
    transformation_matrix: np.ndarray, rotation_type: str = "quaternion"
) -> np.ndarray:
    if len(transformation_matrix.shape) == 3:
        rotation_matrix = transformation_matrix[:, :3, :3]
    elif len(transformation_matrix.shape) == 2:
        rotation_matrix = transformation_matrix[:3, :3]
    else:
        raise ValueError

    if rotation_type == "quaternion":
        return matrix_to_quat(rotation_matrix)
    elif rotation_type == "matrix":
        return rotation_matrix.copy()
    else:
        raise ValueError


def _mujoco_quat_to_scipy_quat(nnabla_grasp_quat: np.ndarray) -> np.ndarray:
    """(w, x, y, z) to (x, y, z, w)"""
    if len(nnabla_grasp_quat.shape) == 2:
        scipy_quat = nnabla_grasp_quat[:, np.array([1, 2, 3, 0])]
        assert scipy_quat.shape[1] == 4
    elif len(nnabla_grasp_quat.shape) == 1:
        scipy_quat = nnabla_grasp_quat[np.array([1, 2, 3, 0])]
        assert scipy_quat.shape[0] == 4
    else:
        raise ValueError

    return scipy_quat


def _scipy_quat_to_mujoco_quat(scipy_quat: np.ndarray) -> np.ndarray:
    """(x, y, z, w) to (w, x, y, z)"""
    if len(scipy_quat.shape) == 2:
        nnabla_grasp_quat = scipy_quat[:, np.array([3, 0, 1, 2])]
        assert nnabla_grasp_quat.shape[1] == 4
    elif len(scipy_quat.shape) == 1:
        nnabla_grasp_quat = scipy_quat[np.array([3, 0, 1, 2])]
        assert nnabla_grasp_quat.shape[0] == 4
    else:
        raise ValueError

    return nnabla_grasp_quat


def quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """transform quaternion to rotation matrix

    Args:
        quaternion (np.ndarray): quaternion, format is (w, x, y, z), shape is (batch_size, 4) or (4,)
    Returns:
        np.ndarray: rotation matrix, shape is (batch_size, 3, 3) or (3, 3)
    """
    # (w x y z) to (x y z w)
    scipy_quaternion = _mujoco_quat_to_scipy_quat(quaternion)
    scipy_rotation = Rotation.from_quat(scipy_quaternion)
    return np.array(scipy_rotation.as_matrix())


def rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """transform rotation vector to quaternion

    Args:
        rotvec (np.ndarray): rotation vector, shape is (batch_size, 3) or (3,)
    Returns:
        np.ndarray: quaternion, shape is (batch_size, 4) or (4, ), format is (w, x, y, z)
    """
    scipy_rotation = Rotation.from_rotvec(rotvec)
    scipy_quaternion = scipy_rotation.as_quat()
    nnabla_grasp_quaternion = _scipy_quat_to_mujoco_quat(scipy_quaternion)
    return np.array(nnabla_grasp_quaternion)


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    """transform rotation vector to matrix

    Args:
        rotvec (np.ndarray): rotation vector, shape is (batch_size, 3) or (3,)
    Returns:
        np.ndarray: matrix, shape is (batch_size, 3, 3) or (3, 3)
    """
    scipy_rotation = Rotation.from_rotvec(rotvec)
    return np.array(scipy_rotation.as_matrix())


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """transform rotation matrix to quaternion

    Args:
        matrix (np.ndarray): rotation matrix, shape is (batch_size, 3, 3) or (3, 3)
    Returns:
        np.ndarray: quaternion, format is (w, x, y, z), shape is (batch_size, 4) or (4,)
    """
    # (w x y z) to (x y z w)
    scipy_rotation = Rotation.from_matrix(matrix)
    scipy_quaternion = scipy_rotation.as_quat()
    nnabla_grasp_quaternion = _scipy_quat_to_mujoco_quat(scipy_quaternion)
    return np.array(nnabla_grasp_quaternion)


def euler_to_quat(
    angles: Union[float, Tuple[float, ...], Tuple[Tuple[float, ...], ...], np.ndarray],
    order: str = "xyz",
) -> np.ndarray:
    """transform euler to quaternion

    Args:
        angles (Union[float, Tuple[float, ...], Tuple[Tuple[float, ...], ...], np.ndarray]): angles
        order (str): order
    Returns:
        np.ndarray: quaternion, format is (w, x, y, z), shape is (batch_size, 4) or (4,)
    """
    scipy_rotation = Rotation.from_euler(order, angles, degrees=False)
    scipy_quaternion = scipy_rotation.as_quat()
    nnabla_grasp_quaternion = _scipy_quat_to_mujoco_quat(scipy_quaternion)
    return np.array(nnabla_grasp_quaternion)


def euler_to_matrix(
    angles: Union[float, Tuple[float, ...], Tuple[Tuple[float, ...], ...]],
    order: str = "xyz",
) -> np.ndarray:
    """transform euler to matrix

    Args:
        angles (Union[float, Tuple[float, ...], Tuple[Tuple[float, ...], ...]]): angles
        order (str): order
    Returns:
        np.ndarray: rotation matrix, shape is (batch_size, 3, 3) or (3, 3)
    """
    scipy_rotation = Rotation.from_euler(order, angles, degrees=False)
    return np.array(scipy_rotation.as_matrix())


def matrix_to_euler(matrix: np.ndarray, order="xyz") -> np.ndarray:
    """transform matrix to euler

    Args:
        matrix (np.ndarray): rotation matrix, shape is (batch_size, 3, 3) or (3, 3)
        order (str): order
    Returns:
        np.ndarray: euler
    """
    scipy_rotation = Rotation.from_matrix(matrix)
    return cast(np.ndarray, scipy_rotation.as_euler(order))


def compute_approach_vector(quaternion: np.ndarray, axis="z") -> np.ndarray:
    if axis == "x":
        return quat_to_matrix(quaternion)[:, 0]
    elif axis == "y":
        return quat_to_matrix(quaternion)[:, 1]
    elif axis == "z":
        return quat_to_matrix(quaternion)[:, 2]
    else:
        raise ValueError


def combine_quat(quat1, quat2, normalize=True):
    """combine (= multiply) 2 quaternion (w, x, y, z)

    Args:
        quat1 (numpy.ndarray): quaternion, shape(4, )
        quat2 (numpy.ndarray): quaternion, shape(4, )
        normalize (bool): If true, return normalize quaternion
    Returns:
        numpy.ndarray: combined quaternion, shape(4, )
    """
    quat1_mat = np.array(
        [
            [quat1[0], -quat1[1], -quat1[2], -quat1[3]],
            [quat1[1], quat1[0], -quat1[3], quat1[2]],
            [quat1[2], quat1[3], quat1[0], -quat1[1]],
            [quat1[3], -quat1[2], quat1[1], quat1[0]],
        ]
    )
    combined_quat = np.dot(quat1_mat, quat2[:, np.newaxis]).flatten()

    if normalize:
        return combined_quat / np.linalg.norm(combined_quat)

    return combined_quat


def rotvec_from_two_vectors(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    """compute rotvec from vector1 to vector2. order is not inversible

    Args:
        vector1 (np.ndarray): vector
        vector2 (np.ndarray): vector
    Returns:
        np.ndarray: rotvec
    """
    normal_vector = np.cross(vector1, vector2)
    angle = angle_from_two_vectors(vector1, vector2)
    norm = np.linalg.norm(normal_vector)
    denom = norm if norm != 0 else 1.0
    return cast(
        np.ndarray, normal_vector / denom * angle
    )  # FIXME: Is this a correct way to cast ??


def angle_from_two_vectors(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """compute rotvec from vector1 to vector2. order is not inversible

    Args:
        vector1 (np.ndarray): vector
        vector2 (np.ndarray): vector
    Returns:
        float: angle
    """
    cos_theta = np.dot(vector1, vector2) / (
        np.linalg.norm(vector1) * np.linalg.norm(vector2)
    )
    angle = np.arccos(cos_theta)
    return float(angle)


def transform_position(
    vector: np.ndarray, transformation_matrix: np.ndarray
) -> np.ndarray:
    """transform position vector by multipling transformation matrix

    Args:
        vector (np.ndarray): position vector
        transformation_matrix (np.ndarray): transformation matrix
    Returns:
        np.ndarray: transformed position vector
    """
    tmp_vector = np.ones((4, 1))
    tmp_vector[:3, 0] = vector
    transformed_vector = np.matmul(transformation_matrix, tmp_vector)
    return cast(
        np.ndarray, transformed_vector.flatten()[:3]
    )  # FIXME: Is this a correct way to cast ??


if __name__ == "__main__":
    print(euler_to_quat([np.pi * 0.5, 0.0, -np.pi * 0.5]))
    print(euler_to_quat([np.pi * 0.5, -np.pi * 0.5, 0.0]))
