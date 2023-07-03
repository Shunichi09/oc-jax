import itertools
import re
from typing import Dict, List, Tuple


import mujoco
import numpy as np


from apop.utils.randoms import rand_min_max
from apop.utils.transformations import (
    combine_quat,
    create_transformation_matrix,
    euler_to_quat,
    extract_position,
    extract_rotation,
    matrix_to_quat,
)


def zbuffer_to_depth(mj_model, depth_buffer):
    # https://github.com/deepmind/dm_control/blob/main/dm_control/mujoco/engine.py#L893
    # Get the distances to the near and far clipping planes.
    extent = mj_model.stat.extent
    near = mj_model.vis.map.znear * extent
    far = mj_model.vis.map.zfar * extent
    # Convert from [0 1] to depth in meters, see links below:
    # http://stackoverflow.com/a/6657284/1461210
    # https://www.khronos.org/opengl/wiki/Depth_Buffer_Precision
    depth = near / (1 - depth_buffer * (1 - near / far))
    return depth


def camera_extrinsic(mj_data, camera_name):
    camera_position = mj_data.cam(camera_name).xpos.ravel().copy()
    camera_rotation = matrix_to_quat(
        mj_data.cam(camera_name).xmat.ravel().copy().reshape(3, 3)
    )
    # NOTE:
    # Mujoco's camera pose is not correct, we should add base transformation to fit the normal camera definition
    # x_-180_quat
    base_quat = np.array([np.cos(-np.pi * 0.5), np.sin(-np.pi * 0.5), 0.0, 0.0])
    camera_rotation = combine_quat(camera_rotation, base_quat)
    return np.linalg.inv(
        create_transformation_matrix(
            camera_position, camera_rotation, rotation_type="quaternion"
        )
    )


def camera_intrinsic(mj_model, camera_name) -> np.ndarray:
    """get camera intrinsic
    Returns:
        K (np.ndarray): camera_intrinsic matrix, K = [[fx  0 cx], [ 0 fy cy], [ 0  0  1]]
    """
    width = mj_model.vis.global_.offwidth
    height = mj_model.vis.global_.offheight

    camera_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    fovy = mj_model.cam_fovy[camera_id]
    focal_scaling = (1.0 / np.tan(np.deg2rad(fovy) / 2)) * height / 2.0
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([[focal_scaling, 0.0, cx], [0.0, focal_scaling, cy], [0.0, 0.0, 1.0]])
    return K


def segmentation_object_id_map(mj_model, segmentation_array):
    def remove_suffixes(name: str):
        name = re.sub(r"_geom|_collision|_visual", "", name)
        name = re.sub(r"_[0-9]+$", "", name)  # remove _XX (X is number)
        return re.sub(r"_$", "", name)  # remove last underbar

    mujoco_type_image = segmentation_array[:, :, 0]
    mujoco_id_image = segmentation_array[:, :, 1]

    geoms = mujoco_type_image == mujoco.mjtObj.mjOBJ_GEOM
    geom_ids = np.unique(mujoco_id_image[geoms])

    unified_id_map = {}
    id2object_map = {}
    for geom_id in geom_ids:
        object_name = str(
            mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        )
        object_name = remove_suffixes(object_name)
        geom_id = int(geom_id)
        if object_name not in unified_id_map.keys():
            # register only once to unify the id
            unified_id_map[object_name] = geom_id
        id2object_map[geom_id] = object_name
    return unified_id_map, id2object_map


def unify_id_of_segmentation_image(segmentation_image, unified_id_map, id2object_map):
    # segmentation image may have different ids for same object.
    # unify those ids into 1 single id.
    image_height, image_width, channel = segmentation_image.shape

    mujoco_type_image = segmentation_image[:, :, 0]
    mujoco_id_image = segmentation_image[:, :, 1]

    geoms = mujoco_type_image == mujoco.mjtObj.mjOBJ_GEOM
    geom_ids = np.unique(mujoco_id_image[geoms])

    unified_id_image = -np.ones((image_height, image_width, 1))

    for geom_id in geom_ids:
        unified_id_image[mujoco_id_image == geom_id] = unified_id_map[
            id2object_map[geom_id]
        ]

    mujoco_type_image = np.expand_dims(mujoco_type_image, axis=-1)
    return np.concatenate([mujoco_type_image, unified_id_image], axis=-1)


def reset_mocap(mj_model, mj_data):
    """
    This function will call mj_forward.
    DO NOT call this function in environment's step function,
    because your simulation will not run correctly.
    """
    for i in range(mj_model.eq_data.shape[0]):
        # See: https://mujoco.readthedocs.io/en/stable/python.html#enums-and-constants
        if mj_model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
            mj_model.eq_data[i, :] = np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            )
    mujoco.mj_forward(mj_model, mj_data)


def get_model_name_state(mj_data, model_name, suffix_str="_body"):
    return create_transformation_matrix(
        mj_data.body(model_name + suffix_str).xpos,
        mj_data.body(model_name + suffix_str).xquat,
        rotation_type="quaternion",
    )


def get_model_name_states(mj_data, model_names, suffix_str="_body"):
    name_to_state = {}
    for model_name in model_names:
        name_to_state[model_name] = get_model_name_state(
            mj_data, model_name, suffix_str
        )
    return name_to_state


def move_model(mj_data, model_name, transformation_matrix, suffix_str="_joint"):
    target_qpos = np.concatenate(
        [
            extract_position(transformation_matrix),
            extract_rotation(transformation_matrix),
        ]
    )
    mj_data.joint(model_name + suffix_str).qpos[:] = target_qpos.copy()
    mj_data.joint(model_name + suffix_str).qvel[:] = np.zeros(6)


def move_models(mj_data, model_names, transformation_matrixes, suffix_str="_joint"):
    for model_name, transformation_matrix in zip(model_names, transformation_matrixes):
        move_model(mj_data, model_name, transformation_matrix, suffix_str)


def is_contact_between_models(mj_model, mj_data, object_model_names):
    num_contacts = []
    for pair in itertools.combinations(object_model_names, 2):
        n_con, geoms, _ = get_contact_info_between_models(
            mj_model, mj_data, pair[0], pair[1]
        )
        num_contacts.append(n_con)
    return any(np.array(num_contacts) > 0)


def _get_contact(mj_model, mj_data, contact_id: int):
    contact = mj_data.contact[contact_id]
    contact_geom1 = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
    contact_geom2 = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
    return contact, contact_geom1, contact_geom2


def get_contact_info_between_models(
    mj_model, mj_data, geom1_abstract_name: str, geom2_abstract_name: str
) -> Tuple[int, List[Tuple[str, str]], List[np.ndarray]]:
    """get contact info between the abstract geom names
    For example, you can get contact info between banana_XXXX and apple_YYYY
    if you call this function (apple, banana)


    Args:
        geom1_abstract_name (str): geom name
        geom2_abstract_name (str): geom name


    Returns:
        Tuple[int, List[Tuple[str, str]], List[np.ndarray]]:
            number of contacts, list of the pair of contact geom names, list of contact points
    """
    # you can get contact info between banana_XXXX and apple_YYYY
    num_contacts = 0
    contact_geoms = []
    contact_points = []
    for contact_id in range(mj_data.ncon):
        contact, contact_geom1, contact_geom2 = _get_contact(
            mj_model, mj_data, contact_id
        )

        if (
            geom1_abstract_name in contact_geom1
            and geom2_abstract_name in contact_geom2
        ) or (
            geom1_abstract_name in contact_geom2
            and geom2_abstract_name in contact_geom1
        ):
            num_contacts += 1
            contact_geoms.append((contact_geom1, contact_geom2))
            contact_points.append(contact.pos.copy())

    return num_contacts, contact_geoms, contact_points


def get_contact_info_of_model(
    mj_model, mj_data, geom_abstract_name: str
) -> Tuple[int, List[str], List[np.ndarray]]:
    """get contact info of the abstract geom name
    For example, you can get contact info of banana_XXXX if you call this function (banana)


    Args:
        geom_abstract_name (str): geom name


    Returns:
        Tuple[int, List[str], List[np.ndarray]]:
            number of contacts, list of the contact geom name with the given geom name, list of contact points
    """
    # you can get contact info between banana_XXXX and apple_YYYY
    num_contacts = 0
    contact_geoms = []
    contact_points = []
    for contact_id in range(mj_data.ncon):
        contact, contact_geom1, contact_geom2 = _get_contact(
            mj_model, mj_data, contact_id
        )

        if geom_abstract_name in contact_geom1:
            if geom_abstract_name in contact_geom2:
                continue
            num_contacts += 1
            contact_geoms.append(contact_geom2)
            contact_points.append(contact.pos.copy())
        elif geom_abstract_name in contact_geom2:
            if geom_abstract_name in contact_geom1:
                continue
            num_contacts += 1
            contact_geoms.append(contact_geom1)
            contact_points.append(contact.pos.copy())

    return num_contacts, contact_geoms, contact_points


def sample_object_model_states(
    mj_model,
    mj_data,
    object_model_names: List[str],
    max_trials_per_each_object_model: int,
    min_workspace: np.ndarray,
    max_workspace: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    This function will call mj_forward.
    DO NOT call this function in environment's step function,
    because your simulation will not run correctly.
    We also recommend to call mj_reset after calling this function.
    """
    object_model_states = {}
    for i, object_model_name in enumerate(object_model_names):
        default_object_model_state = get_model_name_state(
            mj_data, object_model_name
        ).copy()
        for j in range(max_trials_per_each_object_model):
            position = rand_min_max(min_val=min_workspace, max_val=max_workspace)
            z_angle = rand_min_max(min_val=0.0, max_val=2 * np.pi)
            rotation = euler_to_quat(z_angle, order="z")

            object_model_state = create_transformation_matrix(
                position, rotation, "quaternion"
            )
            move_model(
                mj_data,
                object_model_name,
                object_model_state,
            )

            mujoco.mj_forward(mj_model, mj_data)

            if is_contact_between_models(mj_model, mj_data, object_model_names):
                # reset state
                move_model(
                    mj_data,
                    object_model_name,
                    default_object_model_state,
                )
                mujoco.mj_forward(mj_model, mj_data)
                continue
            else:
                object_model_states[object_model_name] = object_model_state
                break

    return object_model_states
