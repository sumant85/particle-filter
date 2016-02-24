__author__ = 'sumant'

from multiprocessing import Pool, Process, Queue
import random
import time

from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode

from models import MotionModel, RobotPose, SensorModel


def get_random_particles(r_map, num_particles):
    np.random.seed(1)
    idx = np.where(r_map.prob_map > 0.9)

    random.seed(9)
    rand_idx = random.sample(range(len(idx[0])), num_particles)

    np.random.seed(11)
    rand_theta = np.random.randint(-180, 180, num_particles)

    return [RobotPose(idx[0][rand_idx[i]], idx[1][rand_idx[i]], np.deg2rad(rand_theta[i]))
            for i in range(len(rand_idx))]


def get_next_particle(particle, sensor_model, motion_model, prev_pose, curr_pose, laser_readings):
    num_retires = 20
    wt = 0
    new_p = None
    while num_retires > 0:
        num_retires -= 1
        new_p = motion_model.sample(prev_pose, curr_pose, particle)
        try:
            wt = sensor_model.map.prob_map[new_p.x][new_p.y]
        except IndexError:
            pass
        if wt > 0.8:
            break

    if wt > 0:
        wt *= sensor_model.sample_list(new_p, laser_readings)
    return new_p, wt


def run_filter_batch(particles, sensor_model, motion_model, prev_pose, curr_pose, laser_readings,
                     start_idx, end_idx):
    weights = []
    new_particles = []
    for particle in particles[start_idx:end_idx]:
        p, w = get_next_particle(particle, sensor_model, motion_model,
                                 prev_pose, curr_pose, laser_readings)
        weights.append(w)
        new_particles.append(p)
    return new_particles, weights


GLOB_SCAT_HANDLE = None
GLOB_PLOT = None
def show_particles(r_map, particles, block=True):
    global GLOB_SCAT_HANDLE
    global GLOB_PLOT
    arr = np.copy(r_map.prob_map).T
    if GLOB_SCAT_HANDLE is None:
        plt.figure()
        GLOB_PLOT = plt.imshow(arr, cmap=cm.gray, origin='lower')
    else:
        GLOB_SCAT_HANDLE.remove()
    GLOB_SCAT_HANDLE = plt.scatter([p.x for p in particles], [p.y for p in particles], marker='+')
    plt.draw()
    plt.show(block=block)


def run_particle_filter(log_file, r_map, num_particles):
    particles = get_random_particles(r_map, num_particles)
    motion_model = MotionModel()
    sensor_model = SensorModel(r_map)

    # show_particles(r_map, particles, block=True)
    prev_pose = None
    with open(log_file, 'r') as fp:
        prev_ts = 0
        # The timestamp for when the particle was last stabilized
        # In stabilizing, we remove introduce new particles into the environment to check if the
        # robot is latching on to a new state.
        last_doctor_time = 0
        doctor_interval = np.random.randint(25, 45)
        for line in fp:
            arr = line.strip().split(' ')
            if arr[0] == 'O':
                continue

            # NB: All cm values are converted to dm for easier computation
            laser_readings = np.array(map(float, arr[7:-1])) / 10
            timestamp = float(arr[-1])

            # Since we are being given laser pose anyways, robot pose is immaterial
            lsr_pose = RobotPose(float(arr[4]) / 10, float(arr[5]) / 10, float(arr[6]))
            assert np.abs(lsr_pose.theta) <= np.pi

            if prev_pose:
                delta_ts = (timestamp - prev_ts)
                if lsr_pose.distance_from(prev_pose) < 1.5:
                    # print 'Continuing due to lack of movement'
                    continue

                print 'delta_ts ', delta_ts
                print timestamp
                print '------------------'
                prev_ts = timestamp
                new_particles = []
                weights = []

                # Run the motion model and get weights via sensor model for all current particles.
                for particle in particles:
                    p, w = get_next_particle(particle, sensor_model, motion_model,
                                             prev_pose, lsr_pose, laser_readings)
                    if w > 0:
                        new_particles.append(p)
                        weights.append(w)

                # Now we do some "doctoring". If the variance along x (arbitrarily chosen)
                # is very low (ideally we should check along both axes, but I'm being lazy),
                # we will reduce of these high confidence particles and introduce random particles
                # into the environment to stabilize the filter.
                # Doctoring would result in resetting of weights and addition of random
                # particles to the new_particles list
                reinit = len(new_particles) < 0.2 * num_particles
                if timestamp - last_doctor_time > doctor_interval or reinit:
                    last_doctor_time = timestamp
                    if np.std([p.x for p in new_particles]) < 4 or reinit:
                        #  Truncate some portion off the end of list
                        new_particles = new_particles[:int(len(new_particles) * 0.35)]
                        print 'Doctoring!!'

                        rand_particles = get_random_particles(r_map, num_particles)

                        # Reset weights for existing particles for purpose of doctoring
                        weights = [3.0 for _x in range(len(new_particles))]
                        weights.extend([1.0 for _x in range(num_particles)])
                        # Set weights for new particles. Thus, existing particles are more likely
                        # to be chosen.
                        new_particles.extend(rand_particles)
                        doctor_interval = np.random.randint(25, 45)

                weights = np.array(weights)
                weights /= np.sum(weights)
                new_particles_idx = np.random.choice(
                    range(len(new_particles)), size=num_particles, replace=True, p=weights)
                particles = [new_particles[_x] for _x in new_particles_idx]
                show_particles(r_map, particles, False)

            prev_pose = lsr_pose


if __name__ == '__main__':
    from map import Map
    m = Map.from_file('map/wean.dat', 800, 800)
    run_particle_filter('log/robotdata2.log', m, 8000)
