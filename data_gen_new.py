import carla
import random
import queue
import numpy as np
import cv2
import pandas as pd

def gen_npc(world, num):
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    for i in range(num):
        vehicle_bp = random.choice(bp_lib.filter('vehicle'))
        npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        if npc:
            npc.set_autopilot(True)

def gen_car(world):
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bp = bp_lib.find('vehicle.lincoln.mkz_2020')

    test = True
    while test:
        vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        if vehicle:
            test = False
            vehicle.set_autopilot(True)

    camera_r = bp_lib.find('sensor.camera.rgb')
    # camera_r.set_attribute('image_size_x', '640')
    # camera_r.set_attribute('image_size_y', '480')
    camera_init_trans = carla.Transform(carla.Location(z=2))
    camera_rgb = world.spawn_actor(camera_r, camera_init_trans, attach_to=vehicle)

    camera_d = bp_lib.find('sensor.camera.depth')
    # camera_d.set_attribute('image_size_x', '640')
    # camera_d.set_attribute('image_size_y', '480')
    camera_init_trans = carla.Transform(carla.Location(z=2))
    camera_depth = world.spawn_actor(camera_d, camera_init_trans, attach_to=vehicle)

    camera_s = bp_lib.find('sensor.camera.semantic_segmentation')
    # camera_s.set_attribute('image_size_x', '640')
    # camera_s.set_attribute('image_size_y', '480')
    camera_init_trans = carla.Transform(carla.Location(z=2))
    camera_semantic = world.spawn_actor(camera_s, camera_init_trans, attach_to=vehicle)


    rgb_queue = queue.Queue()
    camera_rgb.listen(rgb_queue.put)
    depth_queue = queue.Queue()
    camera_depth.listen(depth_queue.put)
    semantic_queue = queue.Queue()
    camera_semantic.listen(semantic_queue.put)

    return vehicle, rgb_queue, depth_queue, semantic_queue, camera_rgb, camera_depth, camera_semantic


client = carla.Client('localhost', 2000)
world = client.get_world()
bp_lib = world.get_blueprint_library()
# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 5
world.apply_settings(settings)
gen_npc(world,50)
vehicle, rgb_queue, depth_queue, semantic_queue, camera_rgb, camera_depth, camera_semantic = gen_car(world)
#vehicle.set_autopilot(True)


freq_sign = []
freq_vehicle = []
freq_light = []
file_names = []
freq_all = []
freq_unlabeled = []


for count in range(1000):
    world.tick()
    print(count)
    try:
        tmp = vehicle.get_transform().location
    except:
        print("test")
        camera_rgb.destroy()
        camera_depth.destroy()
        camera_semantic.destroy()
        gen_npc(world, 10)
        world.tick()
        vehicle, rgb_queue, depth_queue, semantic_queue, camera_rgb, camera_depth, camera_semantic = gen_car(world)
        #vehicle.set_autopilot(True)
        world.tick()

    image = rgb_queue.get()
    img_rgb = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

    image = depth_queue.get()
    #image.convert(carla.ColorConverter.LogarithmicDepth)
    img_depth = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    normalized_depth = np.dot(img_depth[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)

    image = semantic_queue.get()
    img_semantic = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    arr_depth = img_semantic[:, :, 2]
    mask = np.isin(arr_depth, [10, 12, 18])
    #The median prob is actually the median of the prob of all class over all pixels. And the weight is the median prob divided by the prob for that class.
    arr_depth[~mask] = 0
    arr_depth[arr_depth == 10] = 1
    arr_depth[arr_depth == 12] = 2
    arr_depth[arr_depth == 18] = 3
    p_all = np.count_nonzero(arr_depth) / 480000
    p_vehicle = np.count_nonzero(arr_depth == 1) / 480000
    p_sign = np.count_nonzero(arr_depth == 2) / 480000
    p_light = np.count_nonzero(arr_depth == 3) / 480000
    p_unlabeled = np.count_nonzero(arr_depth == 0) / 480000

    freq_sign.append(p_sign)
    freq_light.append(p_light)
    freq_vehicle.append(p_vehicle)
    file_names.append("image_"+str(count))
    freq_all.append(p_all)
    freq_unlabeled.append(p_unlabeled)
    cv2.imwrite("data/run2/rgb/image_" + str(count) + ".png", img_rgb)
    np.save("data/run2/depth/image_" + str(count) + ".npy", normalized_depth)
    np.save("data/run2/semantic/image_" + str(count) + ".npy", arr_depth)

df = pd.DataFrame()
df["Name"] = file_names
df["Unlabeled"] = freq_unlabeled
df["Signs"] = freq_sign
df["Lights"] = freq_light
df["Vehicles"] = freq_vehicle
df['All'] = freq_all
df.to_csv("data/run2/data.csv")







