import math
import numpy
from vector import Vector2D

DEFAULT_JOINT_OFFSETS_DEG = {
    # Offsets map geometry angles to the servo frame; tune with a measured pose.
    "shoulder_lift": 173.15464810372569,
    "elbow_flex": 217.68087505945822,
    "wrist_flex": 240.8355231631839,
}

def check_triangle_validity(a, b, c): 
    return (a + b >= c) or (a + c >= b) or (b + c >= a) or \
           math.isclose(a + b, c) or math.isclose(a + c, b) or math.isclose(b + c, a)

def get_intersections(position1, radius1, position2, radius2):
    # Calculate distance
    distance_vector = position2 - position1
    distance = distance_vector.length()

    if math.isclose(distance, 0):
        fallback_offset = Vector2D(radius1, 0)
        return position1 + fallback_offset, position1 - fallback_offset

    # Distance = a + b
    a = max((radius1**2 - radius2**2 + distance**2) / (2 * distance), 0)

    # Calculate height
    height = math.sqrt(max(radius1**2 - a**2, 0))
    height_vector = Vector2D(-height * distance_vector.sin(), height * distance_vector.cos())

    # Calculate intersections' position
    point = position1 + distance_vector.normalized() * a
    intersection1 = point + height_vector
    intersection2 = point - height_vector

    return intersection1, intersection2

def find_side(minimal_length, maximal_length, side1, side2):
    for side in numpy.arange(maximal_length, minimal_length, -0.5):
        if check_triangle_validity(side, side1, side2):
            return side
    return 0

def resolve_ik(chain, end_effector, pole=None):
    if pole is None:
        pole = Vector2D(0, 0)

    maximal_distance = sum(chain)
    
    # Init variables
    new_vectors = []
    
    # Calculate current side
    if end_effector.length() > maximal_distance:
        end_effector = end_effector.normalized() * maximal_distance
        
    current_side_vector = end_effector

    for i in range(len(chain) - 1, 0, -1):
        current_side = current_side_vector.length()
        
        # Find possible side
        new_side = find_side(0, sum(chain[:i]), chain[i], current_side)
        
        # Find intersection nearest to the pole
        if i != 1:
            intersections = get_intersections(current_side_vector, chain[i], Vector2D(0, 0), new_side)
        else:
            intersections = get_intersections(current_side_vector, chain[i], Vector2D(0, 0), chain[0])

        intersection = Vector2D(0, 0)
        # Vector2D < operator is overloaded to compare lengths
        if (intersections[0] - pole) < (intersections[1] - pole):
            intersection = intersections[0]
        else:
            intersection = intersections[1]
            
        # Change vector
        new_vectors.insert(0, current_side_vector - intersection)
        current_side_vector = intersection

    new_vectors.insert(0, current_side_vector)

    return new_vectors

def vector_angle_degrees(vector):
    if vector.length() == 0:
        return 0.0

    return math.degrees(vector.get_angle()) % 360.0

def _signed_angle(a, b):
    cross = a.x * b.y - a.y * b.x
    dot = a.x * b.x + a.y * b.y
    angle = math.degrees(math.atan2(cross, dot))
    if math.isclose(abs(angle), 180.0, abs_tol=1e-9):
        return 180.0
    return angle

def _servo_angle(geom_angle, joint_name, offsets):
    offset = offsets.get(joint_name, 0.0)
    return (offset - geom_angle) % 360.0

def to_relative_angles(absolute_angles):
    if not absolute_angles:
        return []

    relative = [absolute_angles[0]]
    for i in range(1, len(absolute_angles)):
        delta = (absolute_angles[i] - absolute_angles[i - 1] + 180.0) % 360.0 - 180.0
        relative.append(delta)
    return relative

def calculate_angles(x, y, z, link_lengths, pole_y=0, pole_z=0, joint_offsets=None):
    """
    Calculate inverse kinematics for an n-link robotic arm.
    
    :param x: Target X coordinate for the end effector.
    :param y: Target Y coordinate for the end effector.
    :param z: Target Z coordinate for the end effector.
    :param link_lengths: List specifying the lengths of each link in the chain.
    :param pole_y: Optional Y coordinate for the pole (used to choose joint orientations).
    :param pole_z: Optional Z coordinate for the pole.
    :return: Dictionary containing the joint angles in degrees mapped to the servo frame (0-360).
    """
    # The extension of the arm in the 3D plane is the hypotenuse of x and y
    extension = math.hypot(x, y)
    end_effector = Vector2D(extension, z)
    pole = Vector2D(pole_y, pole_z)
    
    vectors = resolve_ik(link_lengths, end_effector, pole)

    offsets = DEFAULT_JOINT_OFFSETS_DEG if joint_offsets is None else {
        **DEFAULT_JOINT_OFFSETS_DEG,
        **joint_offsets,
    }

    shoulder_lift = 0.0
    elbow_flex = 0.0
    wrist_flex = 0.0

    if len(vectors) > 0:
        base_axis = Vector2D(1.0, 0.0)
        v1 = vectors[0]
        shoulder_geom = _signed_angle(base_axis, v1)
        shoulder_lift = _servo_angle(shoulder_geom, "shoulder_lift", offsets)

    if len(vectors) > 1:
        v2 = vectors[1]
        elbow_geom = _signed_angle(Vector2D(-v1.x, -v1.y), v2)
        elbow_flex = _servo_angle(elbow_geom, "elbow_flex", offsets)

    if len(vectors) > 2:
        v3 = vectors[2]
        wrist_geom = -_signed_angle(Vector2D(-v2.x, -v2.y), v3)
        wrist_flex = _servo_angle(wrist_geom, "wrist_flex", offsets)

    shoulder_pan = (math.degrees(math.atan2(x, y)) + 90.0) % 360.0

    return {
        "shoulder_pan": shoulder_pan,
        "shoulder_lift": shoulder_lift,
        "elbow_flex": elbow_flex,
        "wrist_flex": wrist_flex,
    }

if __name__ == "__main__":
    # Example usage:
    target_x = 10
    target_y = 10
    target_z = 10

    print("Run this module from the main robot script so link lengths come from one place.")
    print("Example usage requires link_lengths to be passed in by the caller.")
