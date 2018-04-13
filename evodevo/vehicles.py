import math
from functools import partial


def send_to_simulator(sim, weight_matrix, devo_matrix, seconds, height=0.3, eps=0.05, source=20, r=1, g=1, b=1):
    """
    A quadruped has a sphere torso with one leg on each side.

    Each leg consists of thigh and shin cylinders which develop (i.e. grow/shrink in length).

    Leg segments are connected by hip and knee joints.

    The shins (it's foot) then has a touch sensor.

    """
    floor = 2.9
    radius = height*1.5
    length = 0.1
    leg_angle = -1.0

    main_body = sim.send_sphere(x=0, y=0, z=height+eps+floor, radius=radius, mass=10.0, r=r, g=g, b=b)

    ramp = sim.send_box(x=0, y=0, z=2.5, r1=0, r2=0, r3=1,
                        length=source, width=source*2-height*4, height=0.5, mass=100,
                        r=0.1, g=1, b=1)
    ledge1 = sim.send_box(x=source/2.5, y=source/2.5, z=eps+1, r1=0, r2=0, r3=1,
                          length=1, width=1, height=2, mass=150)
    ledge2 = sim.send_box(x=source/2.5, y=-source/2.5, z=eps+1, r1=0, r2=0, r3=1,
                          length=1, width=1, height=2, mass=150)

    # stopper = sim.send_box(x=-source+0.3, y=-source/2.1, z=0.25,
    #                        length=0.5, width=0.5, height=0.5, mass=1e9)

    # id arrays
    limbs = [0]*4
    hips = [0]*4
    sensor_neurons = [0]*3
    motor_neurons = [0]*4
    slide_cyls = [0]*4
    slide_joints = [0]*4
    devo_neurons = [0]*4

    delta = float(math.pi)/2.0
    adjust = [-0.5, 0., -0.5, 0.]

    for i in range(4):
        theta = delta*(i+adjust[i])
        x_pos = math.cos(theta)*radius*1.1
        y_pos = math.sin(theta)*radius*1.1

        limbs[i] = sim.send_cylinder(x=x_pos, y=y_pos, z=height/1.1+eps+floor,
                                     r1=x_pos, r2=y_pos, r3=leg_angle,
                                     length=length, radius=eps, mass=1,
                                     r=r, g=g, b=b
                                     )

        hips[i] = sim.send_hinge_joint(main_body, limbs[i],
                                       x=x_pos, y=y_pos, z=height+eps+floor,
                                       n1=-y_pos, n2=x_pos, n3=math.pi*2,
                                       lo=-math.pi/4.0, hi=math.pi/4.0
                                       )

        motor_neurons[i] = sim.send_motor_neuron(joint_id=hips[i])

        slide_cyls[i] = sim.send_cylinder(x=x_pos, y=y_pos, z=height/1.1+eps+floor,
                                          r1=x_pos, r2=y_pos, r3=leg_angle,
                                          length=length/2.0, radius=eps, mass=1,
                                          r=0, g=g, b=0
                                          )

        slide_joints[i] = sim.send_slider_joint(slide_cyls[i], limbs[i], x=x_pos, y=y_pos, z=leg_angle,
                                                lo=-0.75, hi=0.75)

    # CPG
    sensor_neurons[0] = sim.send_function_neuron(math.sin)
    sensor_neurons[1] = sim.send_function_neuron(math.cos)

    # for fitness only
    light_sensor = sim.send_light_sensor(main_body)

    env_box = sim.send_box(x=-source, y=0, z=2*radius+eps,
                           length=source, width=1, height=radius*3,
                           mass=1e9,
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
    for i in range(4):
        devo_neurons[i] = sim.send_motor_neuron(joint_id=slide_joints[i])

    def develop(t, s, f):
        adj = 2.0
        ts = float(seconds-1)
        return s + t/ts*(f-s)*adj

    # start = 0
    # end = 1
    # bias_id = sim.send_function_neuron(partial(develop, s=start, f=end))
    # bias_id = sim.send_bias_neuron()
    count = 0
    for target_id in devo_neurons:
        start, end = devo_matrix[count, :]
        # sim.send_developing_synapse(bias_id, target_id, start_weight=start, end_weight=end)
        bias_id = sim.send_function_neuron(partial(develop, s=start, f=end))
        sim.send_synapse(bias_id, target_id, 1.0)
        count += 1

    # layouts are useful for other things not relevant to this example
    layout = {'limbs': limbs,
              'hips': hips,
              'sensor_neurons': sensor_neurons,
              'motor_neurons': motor_neurons,
              'slide_joints': slide_joints,
              'slide_clys': slide_cyls,
              'devo_neurons': devo_neurons,
              'light_sensor': light_sensor,
              'light_source': light_source}

    sim.create_collision_matrix('all')

    return layout
