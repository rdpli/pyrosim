import math


def send_to_simulator(sim, weight_matrix, devo_matrix, height=0.3, eps=0.05, source=10, r=1, g=1, b=1):
    """
    A quadruped has a sphere torso with one leg on each side.

    Each leg consists of thigh and shin cylinders which develop (i.e. grow/shrink in length).

    Leg segments are connected by hip and knee joints.

    The shins (it's foot) then has a touch sensor.

    """
    main_body = sim.send_sphere(x=0, y=0, z=height+eps, radius=height/2., r=r, g=g, b=b)

    # id arrays
    thighs = [0]*4
    shins = [0]*4
    hips = [0]*4
    knees = [0]*4
    slide_cyls = [0]*8
    slide_joints = [0]*8
    foot_sensors = [0]*4
    sensor_neurons = [0]*4
    motor_neurons = [0]*8
    devo_neurons = [0]*8

    delta = float(math.pi)/2.0

    for i in range(4):
        theta = delta*i
        x_pos = math.cos(theta)*height
        y_pos = math.sin(theta)*height

        thighs[i] = sim.send_cylinder(x=x_pos, y=y_pos, z=height+eps,
                                      r1=x_pos, r2=y_pos, r3=0,
                                      length=height, radius=eps,
                                      r=r, g=g, b=b
                                      )

        # main_body to thigh
        hips[i] = sim.send_hinge_joint(main_body, thighs[i],
                                       x=x_pos/2.0, y=y_pos/2.0, z=height+eps,
                                       n1=-y_pos, n2=x_pos, n3=0,
                                       lo=-math.pi/4.0, hi=math.pi/4.0
                                       )

        # hip motor
        motor_neurons[i] = sim.send_motor_neuron(joint_id=hips[i])

        # slide1
        slide_cyls[i] = sim.send_cylinder(x=x_pos, y=y_pos, z=height+eps,
                                          r1=x_pos, r2=y_pos, r3=0,
                                          length=height, radius=eps,
                                          r=r, g=g, b=b
                                          )

        # thigh to slide1
        slide_joints[i] = sim.send_slider_joint(slide_cyls[i], thighs[i], x=x_pos, y=y_pos, z=0)

        # attach slide motor later

        # now for the lower legs
        x_pos2 = math.cos(theta)*1.5*height
        y_pos2 = math.sin(theta)*1.5*height

        shins[i] = sim.send_cylinder(x=x_pos2, y=y_pos2, z=(height+eps)/2.0,
                                     r1=0, r2=0, r3=1,
                                     length=height, radius=eps,
                                     r=r, g=g, b=b
                                     )

        # slide1 to shin
        knees[i] = sim.send_hinge_joint(slide_cyls[i], shins[i],
                                        x=x_pos2, y=y_pos2, z=height+eps,
                                        n1=-y_pos, n2=x_pos, n3=0,
                                        lo=-math.pi/4.0, hi=math.pi/4.0
                                        )

        # knee motor
        motor_neurons[i+4] = sim.send_motor_neuron(knees[i])

        # slide2
        slide_cyls[i+4] = sim.send_cylinder(x=x_pos2, y=y_pos2, z=(height+eps)/2.0,
                                            r1=0, r2=0, r3=1,
                                            length=height, radius=eps,
                                            r=r, g=g, b=b
                                            )

        # shin to slide2
        slide_joints[i+4] = sim.send_slider_joint(shins[i], slide_cyls[i+4], x=0, y=0, z=height+eps)

        # attach slide motor later

        foot_sensors[i] = sim.send_touch_sensor(slide_cyls[i+4])  # rather than on shins[i]
        sensor_neurons[i] = sim.send_sensor_neuron(foot_sensors[i])

    light_sensor = sim.send_light_sensor(main_body)

    env_box = sim.send_box(x=source, y=-source, z=height,
                           length=height*2, width=height*2, height=height*2,
                           mass=3.,
                           r=1, g=248/255., b=66/255.)
    light_source = sim.send_light_source(env_box)

    count = 0
    # developing synapses linearly change from the start value to the end value during the evaluation period;
    # here we connect each sensor to each motor.
    for source_id in sensor_neurons:
        for target_id in motor_neurons:
            count += 1
            start_weight = weight_matrix[source_id, target_id, 0]
            end_weight = weight_matrix[source_id, target_id, 1]
            sim.send_developing_synapse(source_id, target_id, start_weight=start_weight, end_weight=end_weight)

    # now we send all the slide motors
    for i in range(8):
        devo_neurons[i] = sim.send_motor_neuron(joint_id=slide_joints[i])

    bias_id = sim.send_bias_neuron()
    count = 0
    for target_id in devo_neurons:
        start, end = devo_matrix[count, :]
        sim.send_developing_synapse(bias_id, target_id, start_weight=start, end_weight=end)
        count += 1

    # layouts are useful for other things not relevant to this example
    layout = {'thighs': thighs,
              'shins': shins,
              'hips': hips,
              'knees': knees,
              'feet': foot_sensors,
              'sensor_neurons': sensor_neurons,
              'motor_neurons': motor_neurons,
              'slide_joints': slide_joints,
              'slide_clys': slide_cyls,
              'bias_neuron': bias_id,
              'devo_neurons': devo_neurons,
              'light_sensor': light_sensor,
              'light_source': light_source}

    sim.create_collision_matrix('all')

    return layout

