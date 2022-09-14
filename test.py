#world.tick()
image = image_queue.get()
img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
#actor_list = world.get_actors()
# Get camera matrix
world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
labs = []
x_mins=[]
y_mins=[]
x_maxs=[]
y_maxs=[]
files = []

#actor_list = world.get_actors()
actor_list = []
for i in world.get_actors().filter('*vehicle*'):
    actor_list.append(i)
for i in world.get_actors().filter('*traffic*'):
    actor_list.append(i)
print(actor_list)

for curr in actor_list:
    if curr.id != vehicle.id:
        bb = curr.bounding_box
        dist = curr.get_transform().location.distance(vehicle.get_transform().location)
        print(dist)
        if dist < 50:
            print("test")
            forward_vec = vehicle.get_transform().get_forward_vector()
            ray = curr.get_transform().location - vehicle.get_transform().location

            if forward_vec.dot(ray) > 1:
                lab = ""
                lab = curr.type_id
                p1 = get_image_point(bb.location, K,world_2_camera)
                verts = [v for v in bb.get_world_vertices(curr.get_transform())]
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




                cv2.line(img, (int(x_min),int(y_min)), (int(x_max),int(y_min)), (0,0,255, 255), 1)
                cv2.line(img, (int(x_min),int(y_max)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
                cv2.line(img, (int(x_min),int(y_min)), (int(x_min),int(y_max)), (0,0,255, 255), 1)
                cv2.line(img, (int(x_max),int(y_min)), (int(x_max),int(y_max)), (0,0,255, 255), 1)
cv2.imwrite("out/test.png", img)
