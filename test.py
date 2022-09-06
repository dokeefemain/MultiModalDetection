import carla
import random

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
world = client.get_world()

spectator = world.get_spectator()

# Get the location and rotation of the spectator through its transform
transform = spectator.get_transform()

location = transform.location
rotation = transform.rotation
print(location, rotation)
# Set the spectator with an empty transform
#spectator.set_transform(carla.Transform())
spawn_points = world.get_map().get_spawn_points()
spawn_0 = spawn_points[0]

# Move to spawn 0
#spectator.set_transform(spawn_0)


vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*')
my_vehicle = world.spawn_actor(random.choice(vehicle_blueprints), spawn_0)
# # Create a transform to place the camera on top of the vehicle
# camera_init_trans = carla.Transform(carla.Location(z=1.5))
#
# # We create the camera through a blueprint that defines its properties
# camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
#
# # We spawn the camera and attach it to our ego vehicle
# camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)
# camera.listen(lambda image: image.save_to_disk('out/%06d.png' % image.frame))

# Find the blueprint of the sensor.
blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
# Modify the attributes of the blueprint to set image resolution and field of view.
blueprint.set_attribute('image_size_x', '1920')
blueprint.set_attribute('image_size_y', '1080')
blueprint.set_attribute('fov', '110')
# Set the time in seconds between sensor captures
blueprint.set_attribute('sensor_tick', '1.0')
transform = carla.Transform(carla.Location(x=0.8, z=1.7))
sensor = world.spawn_actor(blueprint, transform, attach_to=my_vehicle)
sensor.listen()