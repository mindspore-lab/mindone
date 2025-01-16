import numpy as np


def make_float3(arr):
    return np.array(arr, dtype=np.float32)


def make_int3(arr):
    return np.array(arr, dtype=np.int32)


# sim clamp
def clamp(value, low, high):
    return np.clip(value, low, high)


def apply_contraction(point, min_point, max_point, contraction_type):
    # map with res
    normalized = (point - min_point) / (max_point - min_point)
    return normalized


class SingleRaySpec:
    def __init__(self, origin, dir, inv_dir):
        self.origin = make_float3(origin)
        self.dir = make_float3(dir)
        self.inv_dir = make_float3(inv_dir)


class AABBSpec:
    def __init__(self, min, max):
        self.min = make_float3(min)
        self.max = make_float3(max)


def setup_traversal(ray, tmin, tmax, eps, aabb, resolution):
    # Outputs will be returned as a tuple
    res = make_float3(resolution)
    voxel_size = (aabb.max - aabb.min) / res
    ray_start = ray.origin + ray.dir * (tmin + eps)
    ray_end = ray.origin + ray.dir * (tmax - eps)

    # Get voxel index of start and end within grid
    current_index = make_int3(apply_contraction(ray_start, aabb.min, aabb.max, "AABB") * res)
    current_index = clamp(current_index, make_int3(0, 0, 0), resolution - 1)

    final_index = make_int3(apply_contraction(ray_end, aabb.min, aabb.max, "AABB") * res)
    final_index = clamp(final_index, make_int3(0, 0, 0), resolution - 1)

    # Calculate index delta based on ray direction
    index_delta = make_int3(1 if ray.dir.x > 0 else 0, 1 if ray.dir.y > 0 else 0, 1 if ray.dir.z > 0 else 0)
    start_index = current_index + index_delta
    tmax_xyz = ((aabb.min + ((make_float3(start_index) * voxel_size) - ray_start)) * ray.inv_dir) + tmin

    tdist = make_float3(
        tmax if ray.dir.x == 0.0 else tmax_xyz[0],
        tmax if ray.dir.y == 0.0 else tmax_xyz[1],
        tmax if ray.dir.z == 0.0 else tmax_xyz[2],
    )

    step_float = make_float3(
        1.0 if ray.dir.x > 0.0 else -1.0 if ray.dir.x < 0.0 else 0.0,
        1.0 if ray.dir.y > 0.0 else -1.0 if ray.dir.y < 0.0 else 0.0,
        1.0 if ray.dir.z > 0.0 else -1.0 if ray.dir.z < 0.0 else 0.0,
    )
    step_index = make_int3(step_float)

    delta_temp = voxel_size * ray.inv_dir * step_float
    delta = make_float3(
        tmax if ray.dir.x == 0.0 else delta_temp[0],
        tmax if ray.dir.y == 0.0 else delta_temp[1],
        tmax if ray.dir.z == 0.0 else delta_temp[2],
    )

    # Return outputs as a tuple
    return delta, tdist, step_index, current_index, final_index


if __name__ == "__main__":
    # Example usage:
    ray = SingleRaySpec(origin=[0, 0, 0], dir=[1, 1, 1], inv_dir=[1, 1, 1])  # Example values
    aabb = AABBSpec(min=[0, 0, 0], max=[10, 10, 10])  # Example values
    resolution = [10, 10, 10]  # Example resolution
    tmin, tmax, eps = 0.0, 10.0, 0.001  # Example values

    delta, tdist, step_index, current_index, final_index = setup_traversal(ray, tmin, tmax, eps, aabb, resolution)
    print("delta:", delta)
    print("tdist:", tdist)
    print("step_index:", step_index)
    print("current_index:", current_index)
    print("final_index:", final_index)
