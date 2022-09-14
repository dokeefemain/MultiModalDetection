import carla
import random
import pandas as pd
import cv2
import numpy as np
import queue

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(loc, K, w2c):
        # Calculate 2D projection of 3D coordinate

        # Format the input coordinate (loc is a carla.Position object)
        point = np.array([loc.x, loc.y, loc.z, 1])
        # transform to camera coordinates
        point_camera = np.dot(w2c, point)

        # New we must change from UE4's coordinate system to an "standard"
        # (x, y ,z) -> (y, -z, x)
        # and we remove the fourth componebonent also
        point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

        # now project 3D->2D using the camera matrix
        point_img = np.dot(K, point_camera)
        # normalize
        point_img[0] /= point_img[2]
        point_img[1] /= point_img[2]

        return point_img[0:2]


client = carla.Client('localhost', 2000)
world = client.get_world()
bp_lib = world.get_blueprint_library()
vehicle_bp =bp_lib.find('vehicle.lincoln.mkz_2020')
spawn_points = world.get_map().get_spawn_points()
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
camera_bp = bp_lib.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1920')
camera_bp.set_attribute('image_size_y', '1080')
camera_bp.set_attribute('fov', '110')
camera_init_trans = carla.Transform(carla.Location(z=2))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
vehicle.set_autopilot(True)
settings = world.get_settings()
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 5
world.apply_settings(settings)

# Get the map spawn points
spawn_points = world.get_map().get_spawn_points()

# Create a queue to store and retrieve the sensor data
image_queue = queue.Queue()
camera.listen(image_queue.put)

world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

# Get the attributes from the camera
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()
fov = camera_bp.get_attribute("fov").as_float()

# Calculate the camera projection matrix to project from 3D -> 2D
K = build_projection_matrix(image_w, image_h, fov)

labs = []
x_mins=[]
y_mins=[]
x_maxs=[]
y_maxs=[]
files = []

for count in range(20):
    world.tick()
    bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
    bounding_box_set.extend(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))
    edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]
    image = image_queue.get()
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    actor_list = world.get_actors()
    # Get camera matrix
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
    name = "out/image_" + str(count) + ".png"
    cv2.imwrite(name, img)

    for bb in bounding_box_set:
        dist = bb.location.distance(vehicle.get_transform().location)
        if dist < 30:
            forward_vec = vehicle.get_transform().get_forward_vector()
            ray = bb.location - vehicle.get_transform().location

            if forward_vec.dot(ray) > 1:
                lab = ""
                location_x, location_y = bb.location.x, bb.location.y
                print(location_x, location_y)
                for i in actor_list:
                    tmp_x, tmp_y = i.get_location().x, i.get_location().y
                    if round(location_x) == round(tmp_x) and round(location_y) == round(tmp_y):
                        lab = i.type_id
                        print(lab)
                p1 = get_image_point(bb.location, K, world_2_camera)
                verts = [v for v in bb.get_world_vertices(carla.Transform())]
                x_max = -10000
                x_min = 10000
                y_max = -10000
                y_min = 10000
                for vert in verts:
                    p = get_image_point(vert, K, world_2_camera)
                    print(p)
                    # Find the rightmost vertex
                    if p[0] > x_max:
                        x_max = p[0]
                    # Find the leftmost vertex
                    if p[0] < x_min:
                        x_min = p[0]
                    # Find the highest vertex
                    if p[1] > y_max:
                        y_max = p[1]
                    # Find the lowest  vertex
                    if p[1] < y_min:
                        y_min = p[1]
                labs.append(lab)
                x_mins.append(x_min)
                x_maxs.append(x_max)
                y_mins.append(y_min)
                y_maxs.append(y_max)
                files.append(name)

df = pd.DataFrame()
df["files"] = files
df["x1"] = x_mins
df["x2"] = x_maxs
df["y1"] = y_mins
df["y2"] = y_maxs
df["lab"] = labs
df.to_csv("out/data.csv")

