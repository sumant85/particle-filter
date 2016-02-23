__author__ = 'sumant'

import math
import os

import cPickle as pickle
import pathos.multiprocessing as mp
import numpy as np
import scipy.stats


def norm_ang(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


class RobotPose(object):
    def __init__(self, x, y, theta):
        """
        :param x: Map cell x
        :param y: Map cell y
        :param theta: Angle of robot in radians between -pi and +pi
        :return:
        """
        self.x = int(x)
        self.y = int(y)
        self.theta = float(theta)
        assert -np.pi <= self.theta <= np.pi

    def distance_from(self, other):
        """
        :param RobotPose pose:
        :return: Distance in dm
        """
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


class MotionModel(object):
    def __init__(self):
        pass

    def sample(self, prev_rf, curr_rf, prev_wf):
        """Get estimated RobotPose in world frame. This algorithm has benn taken directly
        from Thurn's textbook.
        :param RobotPose prev_rf: Prev robot pose in  robot frame in dm
        :param RobotPose curr_rf: Current robot pose in  robot frame in dm
        :param RobotPose prev_wf: Prev robot pose in  world frame in radians
        :return:
        :rtype: (RobotPose, confidence)
        """
        alpha1 = 0.007
        alpha2 = 0.007
        alpha3 = 0.12
        alpha4 = 0.12

        # To avoid the standard confusion between matrix indices and x, y coordinates,
        # we declare new variables xa_* and ya_* which represent the coordinates
        # along those axes. The axes are oriented as are normally done in geometry.
        xa_prev = prev_rf.y
        xa_curr = curr_rf.y
        ya_prev = prev_rf.x
        ya_curr = curr_rf.x

        delta_rot1 = norm_ang(math.atan2(ya_curr - ya_prev, xa_curr - xa_prev) - prev_rf.theta)
        delta_trans = np.sqrt((ya_curr - ya_prev) ** 2 + (xa_curr - xa_prev) ** 2)
        delta_rot2 = norm_ang(curr_rf.theta - prev_rf.theta - delta_rot1)

        # TODO: Do this in a while loop as done in textbook algo
        sigma_rot1 = alpha1 * delta_rot1 + alpha2 * delta_trans
        sigma_trans = alpha3 * delta_trans + alpha4 * (delta_rot1 + delta_rot2)
        sigma_rot2 = alpha1 * delta_rot2 + alpha2 * delta_trans

        deltah_rot1 = delta_rot1 - sigma_rot1 * np.random.randn()
        deltah_trans = delta_trans - sigma_trans * np.random.randn()
        deltah_rot2 = delta_rot2 - sigma_rot2 * np.random.randn()

        delta_x = int(round(deltah_trans * math.cos(prev_wf.theta + deltah_rot1)))
        delta_y = int(round(deltah_trans * math.sin(prev_wf.theta + deltah_rot1)))

        nxt_theta = norm_ang(prev_wf.theta + deltah_rot1 + deltah_rot2)
        nxt_pose = RobotPose(prev_wf.x - delta_y,  # Note the switch between x and y.
                             prev_wf.y + delta_x,  # This is to maintain sanity in matrix indexing.
                             nxt_theta)
        return nxt_pose


class SensorModel(object):
    ANGLE_GRANULARITY_DEG = 15
    MAX_SENSOR_VAL_DM = 200
    SENSOR_MODEL_FILE = 'cache/sensor_model_%s.dat' % ANGLE_GRANULARITY_DEG
    MAP_MODEL_FILE = 'cache/map_model_%s.dat' % ANGLE_GRANULARITY_DEG

    def __init__(self, r_map, bypass_cache=False):
        """
        :param Map r_map: The map object
        :param bool bypass_cache: If True, we bypass cache and recompute state. This is useful
         to test with smaller versions of test maps.
        :return:
        """
        # TODO: Cleanup code here
        self.map = r_map

        # We cache 2 objects here.
        # mean_distances is a mapping between (x,y) cell indices and a list of distances along
        # angles measured from that cell, with each angle ANGLE_GRANULARITY_DEG apart.
        # sensor_model is a 2D array that stores the sensor model probabilities, given the actual
        # measurement and mean measurement.
        self.mean_distances = {}
        self.sensor_model = []

        map_file = SensorModel.MAP_MODEL_FILE
        sensor_file = SensorModel.SENSOR_MODEL_FILE

        if bypass_cache:
            self._build_map_model()
            self._build_sensor_model()
        else:
            if not os.path.isfile(map_file):
                self._build_map_model()
                self.save_object_to_file(self.mean_distances, map_file)
            else:
                with open(map_file, 'rb') as pkl_file:
                    self.mean_distances = pickle.load(pkl_file)

            if not os.path.isfile(sensor_file):
                self._build_sensor_model()
                if not bypass_cache:
                    self.save_object_to_file(self.sensor_model, sensor_file)
            else:
                with open(SensorModel.SENSOR_MODEL_FILE, 'rb') as pkl_file:
                    self.sensor_model = pickle.load(pkl_file)

    def sample_list(self, pose, measurements):
        """Get P(zt | xt)
        :param RobotPose pose: Current pose
        :param list[float] measurements: Array of measurements from 0-180 degrees with granularity
         of 1 degree
        :return:
        """
        """
        # For each ray, first discover the mean value at this position. To do this, we just add
        the pose offset to our precomputed data.
        # Compute the probability of that ray at that cell and orientation.
        # Assume all rays independent, multiply all probabilities and return result.
        """

        offset_angle = np.rad2deg(pose.theta - np.pi/2)  # -90 to align laser at center
        offset_angle = (offset_angle + 360) % 360  # We want offset angle between 0 to 360
        assert 0 <= offset_angle <= 360

        mean_dist = self.mean_distances.get((pose.x, pose.y), None)
        if mean_dist is None:
            return 0

        n = len(mean_dist)
        angle_granularity = SensorModel.ANGLE_GRANULARITY_DEG
        start_idx = int(math.ceil(offset_angle / angle_granularity))
        start_angle = start_idx * angle_granularity
        laser_idx = int(start_angle - offset_angle)
        assert laser_idx >= 0

        idx_jump = 4  # Every idx_jump * SensorModel.ANGLE_GRANULARITY_DEG degrees
        num_measurements = (180 / (angle_granularity * idx_jump)) - 1

        prob = 1
        for i in range(num_measurements):
            start_idx %= n
            lsr_mnt = min(SensorModel.MAX_SENSOR_VAL_DM, int(measurements[laser_idx]))
            mean_mnt = int(mean_dist[start_idx])
            measurement_prob = self.sensor_model[lsr_mnt][mean_mnt]

            # if np.random.randint(0, 1000) % 999 == 0:
            #     print '------------------'
            #     print lsr_mnt, mean_mnt, measurement_prob
            #     print '------------------'

            prob *= measurement_prob
            start_idx += idx_jump
            laser_idx += (idx_jump * angle_granularity)
        return prob

    @staticmethod
    def save_object_to_file(obj, to_filename):
        with open(to_filename, 'wb') as output:
            pickle.dump(obj, output)

    def _build_sensor_model(self):
        r = range(SensorModel.MAX_SENSOR_VAL_DM+1)
        self.sensor_model = [[0 for _x in r] for _y in r]
        for act_mnt in r:
            for mean_mnt in r:
                self.sensor_model[act_mnt][mean_mnt] = self._sample(mean_mnt, act_mnt)

    @staticmethod
    def _sample(mean_mnt, actual_mnt):
        """
        :param float mean_mnt: The mean measurement
        :param float actual_mnt: Actual measurement seen
        :return:
        """
        # TODO: tweak parameters
        # TODO: The exponential distribution is not looking very nice. Should we try shifting it
        # towards the mean in some way?
        sigma_hit = 1  # decimeters
        lamda_short = 0.03

        z_hit = 0.67
        z_short = 0.13
        z_max = 0.01
        z_rand = 0.19

        sens_max = SensorModel.MAX_SENSOR_VAL_DM

        if actual_mnt >= sens_max:
            p_hit = 0
            p_short = 0
            p_rand = 0
            p_max = z_max * 1.0
        else:
            distn = scipy.stats.norm(mean_mnt, sigma_hit)
            norm = distn.cdf(sens_max) ** -1
            p_hit = z_hit * norm * distn.pdf(actual_mnt)

            if mean_mnt == 0:
                norm = 0
            else:
                norm = (1 - np.exp(-lamda_short * mean_mnt)) ** -1
            p_short = z_short * norm * lamda_short * np.exp(-actual_mnt*lamda_short)

            p_rand = z_rand * (1.0 / sens_max)
            p_max = 0
        return p_hit + p_short + p_rand + p_max

    def _cache_distance(self, coords):
        x, y = coords
        if self.map.prob_map[x][y] > 0.5:
            print x, y
            dists = [self._get_distance_to_obstacle(x, y, angle)
                     for angle in range(0, 360, SensorModel.ANGLE_GRANULARITY_DEG)]
            return x, y, dists
        else:
            return -1, -1, []

    def _build_map_model(self):
        pool = mp.ProcessingPool(6)
        ret = pool.map(self._cache_distance,
                       [(_x, _y) for _x in range(self.map.x_size)
                        for _y in range(self.map.y_size)])
        self.mean_distances = {(x, y): dist for x, y, dist in ret}

    def _get_distance_to_obstacle(self, x, y, theta):
        """Returns the distance to the closest obstacle in decimeters from cell x,y
        by travelling along an angle theta.
        :param int x: X coordinate of cell
        :param int y: Y coordinate of cell
        :param float theta: Angle in degrees
        :return:
        """
        # TODO : Parameter hardcoded. Assumes that cell is 1dm x 1dm
        """
        Steps:
        # Find the direction along x and y based on theta. Theta is then converted to a value
        between 0 and 90 degrees for ease of calculation. For example if theta is 190 degrees, then
        the direction along x is towards left, and that along y is toward bottom.
        # Special case handling of 90 and 270 degrees is required since their tan is infinite.
        # Methodology -
        Move 1 unit along theta from input cell (x, y). Then project that into x and y
        coordinates to find if cell value is -1.
        # N.B> In our coordinate frm, x is along the column, y is along the row. Confusing? I know!
        """

        along_row_dir = 1
        along_col_dir = 1
        if 0 <= theta <= 90:
            along_col_dir = -1
        elif 90 < theta <= 180:
            theta = 180 - theta
            along_col_dir = -1
            along_row_dir = -1
        elif 180 < theta <= 270:
            theta -= 180
            along_row_dir = -1
        else:
            theta = 360 - theta

        # convert to radians
        theta_rad = ((np.pi * theta) / 180)

        if -np.pi > theta_rad or theta_rad > np.pi:
            assert False

        for _x in range(1, SensorModel.MAX_SENSOR_VAL_DM):
            along_col_offset = int(_x * math.sin(theta_rad))
            along_row_offset = int(_x * math.cos(theta_rad))
            row_idx = x + along_col_dir*along_col_offset
            col_idx = y + along_row_dir*along_row_offset
            if row_idx >= self.map.x_size or col_idx >= self.map.y_size:
                return _x - 1
            if self.map.prob_map[row_idx][col_idx] < 0.5:  # TODO: tweak threshold
                break
        return _x

    def _get_distance_to_obstacle_back(self, x, y, theta):
        """Returns the distance to the closest obstacle in decimeters from cell x,y
        by travelling along an angle theta.
        :param int x: X coordinate of cell
        :param int y: Y coordinate of cell
        :param float theta: Angle in degrees
        :return:
        """
        # TODO : Parameter hardcoded. Assumes that cell is 1dm x 1dm

        """
        Steps:
        # Find the direction along x and y based on theta. Theta is then converted to a value
        between 0 and 90 degrees for ease of calculation. For example if theta is 190 degrees, then
        the direction along x is towards left, and that along y is toward bottom.
        # Special case handling of 90 and 270 degrees is required since their tan is infinite.
        # Methodology -
        Move 1 unit in the direction of x starting from input cell (x, y).
        the displacement in y is computed as tan(theta) * x.
        However, there is another problem. If theta is very close to 90/270, then the computed y
         goes beyond the map boundary. This has been handled naively by checking +/- 10 degrees
         around theta. If this gives a big error, consider resolving this error in a better fashion.
        """

        x_dir = 1
        y_dir = 1
        if 0 <= theta <= 90:
            y_dir = -1
        elif 90 < theta <= 180:
            theta = 180 - theta
            x_dir = -1
            y_dir = -1
        elif 180 < theta <= 270:
            theta -= 180
            x_dir = -1
        else:
            theta = 360 - theta

        # convert to radians
        theta_rad = (np.pi * theta) / 180
        i = j = 0
        while i < SensorModel.MAX_SENSOR_VAL_DM - 1:
            i += 1
            if 80 <= theta <= 100 or 260 <= theta <= 290:
                x_dir = 0
                j += 1
            else:
                j = round(i * np.tan(theta_rad))

            x_idx = x + x_dir*i
            y_idx = y + y_dir*j
            if not (0 <= x_idx < self.map.x_size and 0 <= y_idx < self.map.y_size):
                break

            if self.map.prob_map[x_idx][y_idx] < 0.2:  # TODO: tweak threshold
                break
        if theta == 90 or theta == 270:
            dist = float(j)
        else:
            dist = i / math.cos(theta_rad)
        return dist


def plot_sensor_model():
    import matplotlib.pyplot as plt
    import time

    m = Map.from_file('map/wean.dat', 800, 800)
    sm = SensorModel(m)
    for mean_mnt in range(0, 201, 20):
        l = []
        for act_mnt in range(0, SensorModel.MAX_SENSOR_VAL_DM+1):
            prob = sm.sensor_model[act_mnt][mean_mnt]
            l.append(prob)
        plt.plot(l)
        plt.show()


def visualize_map_model(sm):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    arr = [[0 for _x in range(m.x_size)] for _y in range(m.y_size)]

    for x, y in sm.mean_distances:
        dists = sm.mean_distances[(x, y)]
        angle = 0
        for dist in dists:
            theta = angle
            along_row_dir = 1
            along_col_dir = 1
            if 0 <= theta <= 90:
                along_col_dir = -1
            elif 90 < theta <= 180:
                theta = 180 - theta
                along_col_dir = -1
                along_row_dir = -1
            elif 180 < theta <= 270:
                theta -= 180
                along_row_dir = -1
            else:
                theta = 360 - theta

            # convert to radians
            theta_rad = ((np.pi * theta) / 180)

            along_col_offset = int(dist * math.sin(theta_rad))
            along_row_offset = int(dist * math.cos(theta_rad))
            row_idx = x + along_col_dir*along_col_offset
            col_idx = y + along_row_dir*along_row_offset
            arr[row_idx][col_idx] = -1
            angle += SensorModel.ANGLE_GRANULARITY_DEG

    plt.imshow(np.array(arr), cmap=cm.gray)
    plt.show()


def test_motion_model():
    def deg2rad(x):
        return (x * np.pi) / 180
    mm = MotionModel()
    prev = RobotPose(10, 20, deg2rad(20))
    curr = RobotPose(12, 25, deg2rad(30))
    prev_rbt = RobotPose(50, 52, deg2rad(87))
    p = mm.sample(prev, curr, prev_rbt)
    print p


if __name__ == '__main__':
    from map import Map
    # cache_file = 'cache/map_model.dat_%s' % SensorModel.ANGLE_GRANULARITY_DEG
    m = Map.from_file('map/wean.dat', 800, 800)
    sm = SensorModel(m)
    # visualize_map_model(sm)
    plot_sensor_model()

    # test_motion_model()
    # # precache_sensor_model(m, cache_file)

    # cache_file = 'cache/test_sensor_model.dat'
    # precache_sensor_model(m, cache_file)
    # visualize_sensor_model(m, cache_file)