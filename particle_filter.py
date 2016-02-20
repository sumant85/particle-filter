__author__ = 'sumant'

from multiprocessing import Pool, Process, Queue
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode

from models import MotionModel, RobotPose, SensorModel


def get_random_particles(r_map, num_particles):
    # np.random.seed(87)
    idx = np.where(r_map.prob_map > 0.8)

    # random.seed(3)
    rand_idx = random.sample(range(len(idx[0])), num_particles)

    # TODO: Should theta also be randomized?
    # np.random.seed(8)
    rand_theta = np.random.uniform(-np.pi, np.pi, num_particles)

    return [RobotPose(idx[0][rand_idx[i]], idx[1][rand_idx[i]], rand_theta[i])
            for i in range(len(rand_idx))]


def get_next_particle(particle, sensor_model, motion_model, prev_pose, curr_pose, laser_readings):
    new_p = motion_model.sample(prev_pose, curr_pose, particle)
    wt = sensor_model.map.prob_map[new_p.x][new_p.y]
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


def show_particles(r_map, particles, block=True):
    from matplotlib import cm
    arr = np.copy(r_map.prob_map).T
    plt.figure()
    plt.imshow(arr, cmap=cm.gray)
    for p in particles:
        plt.scatter(p.x, p.y, marker='+')
    plt.show(block=block)


def run_particle_filter(log_file, r_map, num_particles, multiprocess=True):
    particles = get_random_particles(r_map, num_particles)
    show_particles(r_map, particles)
    motion_model = MotionModel()
    sensor_model = SensorModel(r_map)

    # if multiprocess:
    #     pool = Pool(processes=5)

    prev_pose = None
    with open(log_file, 'r') as fp:
        num_iterations = 0
        for line in fp:
            arr = line.strip().split(' ')
            # FIXME TODO Currently ignoring all O readings in log, ie depending purely on
            # the L readings (which are probably filtered values) for odometer readings
            if arr[0] == 'O':
                continue

            num_iterations += 1
            # rp = RobotPose(arr[1], arr[2], arr[3])

            # NB: All cm values are converted to dm for easier computation
            laser_readings = np.array(map(float, arr[7:-1])) / 10
            timestamp = float(arr[-1])

            # Since we are being given laser pose anyways, robot pose is immaterial
            lsr_pose = RobotPose(float(arr[4]) / 10, float(arr[5]) / 10, float(arr[6]))
            if lsr_pose.theta > np.pi:
                assert False

            if prev_pose:
                new_particles = []
                weights = []
                # TODO: Earlier code was too slow so we added parallel processing. No longer
                # required after optimizations in sensor model.
                # if multiprocess:
                #     results = []
                #     batch_size = num_particles/5
                #     for i in range(num_particles/batch_size):
                #         start_idx = i * batch_size
                #         end_idx = (i+1) * batch_size
                #         res = pool.apply_async(run_filter_batch,
                #                                (particles, sensor_model, motion_model,
                #                                 prev_pose, lsr_pose, laser_readings,
                #                                 start_idx, end_idx,))
                #         results.append(res)
                #
                #     for res in results:
                #         p, w = res.get()
                #         new_particles.extend(p)
                #         weights.extend(w)
                # else:
                for particle in particles:
                    p, w = get_next_particle(particle, sensor_model, motion_model,
                                             prev_pose, lsr_pose, laser_readings)
                    if w > 0:
                        new_particles.append(p)
                        weights.append(w)

                weights /= np.sum(weights)

                new_particles_idx = \
                    np.random.choice(np.array(range(len(new_particles))), size=num_particles,
                                     replace=True, p=weights)
                particles = [new_particles[_x] for _x in new_particles_idx]

                # Here we randomly add 1000 particles after every couple of iterations
                # to prevent the filter from converging to an incorrect value. The existing
                # particles are twice as likely to be chosen over the new particles
                if num_iterations % 5 == 0:
                    weights = [3.0 for _x in range(num_particles)]

                    num_rand = int(num_particles * 0.1)
                    rand_particles = get_random_particles(r_map, num_rand)
                    weights.extend([1.0 for _x in range(num_rand)])

                    weights = np.array(weights)
                    weights /= np.sum(weights)

                    particles.extend(rand_particles)
                    new_particles_idx = \
                    np.random.choice(np.array(range(int(num_particles * 1.1))), size=num_particles,
                                     replace=True, p=weights)
                    particles = [particles[_x] for _x in new_particles_idx]

            prev_pose = lsr_pose

            if num_iterations > 15:
                print '#########################'
                print timestamp
                num_iterations = 0
                show_particles(r_map, particles)


if __name__ == '__main__':
    from map import Map
    m = Map.from_file('map/wean.dat', 800, 800)
    run_particle_filter('log/robotdata1.log', m, 2000, False)
