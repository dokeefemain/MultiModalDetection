import carla
import random
import queue
import numpy as np
import cv2



client = carla.Client('localhost', 2000)
world  = client.get_world()
bp_lib = world.get_blueprint_library()

spawn_points = world.get_map().get_spawn_points()
vehicle_bp =bp_lib.find('vehicle.lincoln.mkz_2020')
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

for i in range(10):
    vehicle_bp = random.choice(bp_lib.filter('vehicle'))
    npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
    if npc:
        npc.set_autopilot(True)

# spawn camera
camera_r = bp_lib.find('sensor.camera.rgb')
camera_init_trans = carla.Transform(carla.Location(z=2))
camera_rgb = world.spawn_actor(camera_r, camera_init_trans, attach_to=vehicle)

camera_d = bp_lib.find('sensor.camera.depth')
camera_init_trans = carla.Transform(carla.Location(z=2))
camera_depth = world.spawn_actor(camera_d, camera_init_trans, attach_to=vehicle)

camera_s = bp_lib.find('sensor.camera.semantic_segmentation')
camera_init_trans = carla.Transform(carla.Location(z=2))
camera_semantic = world.spawn_actor(camera_s, camera_init_trans, attach_to=vehicle)

vehicle.set_autopilot(True)

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 5
world.apply_settings(settings)

# Create a queue to store and retrieve the sensor data
rgb_queue = queue.Queue()
camera_rgb.listen(rgb_queue.put)
depth_queue = queue.Queue()
camera_depth.listen(depth_queue.put)
semantic_queue = queue.Queue()
camera_semantic.listen(semantic_queue.put)
freq_sign = []
freq_vehicle = []
freq_light = []
for count in range(822,1000):
    world.tick()
    image = rgb_queue.get()
    img_rgb = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    cv2.imwrite("out/rgb/image_" + str(count) + ".png", img_rgb)

    image = depth_queue.get()
    image.convert(carla.ColorConverter.LogarithmicDepth)
    img_depth = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    cv2.imwrite("out/depth/image_"+str(count)+".png", img_depth)
    signs = 0
    vehicles = 0
    lights = 0

    for i in world.get_actors():
        if i.id != vehicle.id:
            dist = i.get_transform().location.distance(vehicle.get_transform().location)
            if dist < 35:
                forward_vec = vehicle.get_transform().get_forward_vector()
                ray = i.get_transform().location - vehicle.get_transform().location
                if forward_vec.dot(ray) > 1:
                    if i.type_id[:7] == 'vehicle':
                        vehicles += 1
                    elif i.type_id == 'traffic.traffic_light':
                        lights += 1
                    elif i.type_id[:7] == 'traffic':
                        signs += 1

    freq_sign.append(signs)
    freq_light.append(lights)
    freq_vehicle.append(vehicles)

    image = semantic_queue.get()
    img_semantic = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    arr_depth = img_semantic[:,:,2]
    mask=np.isin(arr_depth,[10,12,18])
    arr_depth[~mask] = 0
    print(np.median(freq_vehicle), np.median(freq_sign), np.median(freq_light), count)
    np.save("out/semantic/image_"+str(count)+".npy", arr_depth)
    # image.convert(carla.ColorConverter.CityScapesPalette)
    # img_semantic = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
    # cv2.imwrite("out/semantic/image_"+str(count)+".png", img_semantic)
print(np.median(freq_vehicle), np.median(freq_sign), np.median(freq_light))






