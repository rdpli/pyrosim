import math
import numpy as np
import math
import pyrosim

HEIGHT = 0.3
EPS = 0.05
np.random.seed(0)


def send_to_simulator(sim, weight_matrix, devo_matrix):

    main_body = sim.send_sphere(x=0, y=0, z=HEIGHT+EPS, radius=HEIGHT/2.)

    # id arrays
    thighs = [0]*4
    shins = [0]*4
    hips = [0]*4
    knees = [0]*4
    slide_cyls = [0]*8
    slide_joints = [0]*8
    foot_sensors = [0]*4
    sensor_neurons = [0]*5
    motor_neurons = [0]*8
    devo_neurons = [0]*8

    delta = float(math.pi)/2.0

    # quadruped is a box with one leg on each side
    # each leg consists thigh and shin cylinders
    # with hip and knee joints
    # each shin/foot then has a touch sensor
    for i in range(4):
        theta = delta*i
        x_pos = math.cos(theta)*HEIGHT
        y_pos = math.sin(theta)*HEIGHT

        thighs[i] = sim.send_cylinder(x=x_pos, y=y_pos, z=HEIGHT+EPS,
                                      r1=x_pos, r2=y_pos, r3=0,
                                      length=HEIGHT, radius=EPS, capped=True
                                      )

        # main_body to thigh
        hips[i] = sim.send_hinge_joint(main_body, thighs[i],
                                       x=x_pos/2.0, y=y_pos/2.0, z=HEIGHT+EPS,
                                       n1=-y_pos, n2=x_pos, n3=0,
                                       lo=-math.pi/4.0, hi=math.pi/4.0,
                                       speed=1.0)

        motor_neurons[i] = sim.send_motor_neuron(joint_id=hips[i])

        # slide1
        slide_cyls[i] = sim.send_cylinder(x=x_pos, y=y_pos, z=HEIGHT+EPS,
                                          r1=x_pos, r2=y_pos, r3=0,
                                          length=HEIGHT, radius=EPS, capped=True
                                          )

        # thigh to slide1
        slide_joints[i] = sim.send_slider_joint(thighs[i], slide_cyls[i], x=x_pos, y=y_pos, z=0)

        # attach slide motor later

        x_pos2 = math.cos(theta)*1.5*HEIGHT
        y_pos2 = math.sin(theta)*1.5*HEIGHT

        shins[i] = sim.send_cylinder(x=x_pos2, y=y_pos2, z=(HEIGHT+EPS)/2.0,
                                     r1=0, r2=0, r3=1,
                                     length=HEIGHT, radius=EPS, capped=True
                                     )

        # slide1 to shin
        knees[i] = sim.send_hinge_joint(slide_cyls[i], shins[i],
                                        x=x_pos2, y=y_pos2, z=HEIGHT+EPS,
                                        n1=-y_pos, n2=x_pos, n3=0,
                                        lo=-math.pi/4.0, hi=math.pi/4.0
                                        )

        motor_neurons[i+4] = sim.send_motor_neuron(knees[i])

        # slide2
        slide_cyls[i+4] = sim.send_cylinder(x=x_pos2, y=y_pos2, z=(HEIGHT+EPS)/2.0,
                                            r1=0, r2=0, r3=1,
                                            length=HEIGHT, radius=EPS, capped=True
                                            )

        # shin to slide2
        slide_joints[i+4] = sim.send_slider_joint(shins[i], slide_cyls[i+4], x=0, y=0, z=HEIGHT+EPS)

        # attach slide motor later

        foot_sensors[i] = sim.send_touch_sensor(slide_cyls[i+4])  # rather than on shins[i]
        sensor_neurons[i] = sim.send_sensor_neuron(foot_sensors[i])

    count = 0
    # developing synapses linearly change from the start value to the
    # end value over the course of start time to end time
    # Here we connect each sensor to each motor, pulling weights from
    # the weight matrix
    for source_id in sensor_neurons:
        for target_id in motor_neurons:
            count += 1
            start_weight = weight_matrix[source_id, target_id, 0]
            end_weight = weight_matrix[source_id, target_id, 1]
            # sim.send_developing_synapse(source_id, target_id, start_weight=start_weight, end_weight=end_weight)

    for i in range(8):
        devo_neurons[i] = sim.send_motor_neuron(joint_id=slide_joints[i])

    # bias_id = sim.send_bias_neuron()
    count = 0
    for target_id in devo_neurons:
        # print devo_matrix[count]
        bias_id = sim.send_bias_neuron()
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
              'slide_clys': slide_cyls}

    env_box = sim.send_box(x=2, y=-2, z=HEIGHT/2.0,
                           length=HEIGHT,
                           width=HEIGHT,
                           height=HEIGHT,
                           mass=3.)

    # send the box towards the origin with specified force
    # proportional to its mass
    MASS = 3
    # sim.send_external_force(env_box, x=0, y=0, z=MASS*20., time=300)
    # sim.send_external_force(env_box, x=-MASS*70., y=MASS*73., z=0, time=320)

    sim.create_collision_matrix('all')

    return layout


if __name__ == "__main__":

    seconds = 60.0
    dt = 0.05
    eval_time = int(seconds/dt)
    print(eval_time)
    gravity = -1.0

    sim = pyrosim.Simulator(eval_time=eval_time,
                            debug=False,
                            play_paused=True,
                            gravity=gravity,
                            play_blind=False,
                            use_textures=True,
                            capture=False,
                            dt=dt)
    num_sensors = 5
    num_motors = 8

    # our weight matrix specifies the values for the starting and ending weights
    weight_matrix = np.random.rand(num_sensors+num_motors, num_sensors+num_motors, 2)
    weight_matrix = 2.0 * weight_matrix - 1.0

    devo_matrix = np.ones((8, 2), dtype=float)
    devo_matrix[:, 1] *= -1.0

    layout = send_to_simulator(sim, weight_matrix=weight_matrix, devo_matrix=devo_matrix)
    sim.start()

    sim.wait_to_finish()
