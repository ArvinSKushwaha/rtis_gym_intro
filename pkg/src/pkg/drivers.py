from typing import Tuple
import numpy.typing as npt

import numpy as np
from f110_gym.envs import F110Env
import csv
import math

# Normalizes the angle to be between -pi and pi
def normalize_angle(angle: float) -> float:
    if angle > np.pi:
        angle -= 2 * np.pi
        return normalize_angle(angle)
    elif angle < -np.pi:
        angle += 2 * np.pi
        return normalize_angle(angle)
    else:
        return angle

# Our implementation of the pure pursuit algorithm
class PurePursuitDriver:
    WHEELBASE_LENGTH = 0.3302
    MIN_LOOKAHEAD = 1.5
    LOOKAHEAD_CONST = 0.3
    K = 10
    C = 1
    # SPEED_MIN = 2
    SPEED_MIN = 1
    TURN_MAX = 0.4189 # in radians
    R = K / ( SPEED_MIN * (TURN_MAX + C) )

    # Reads in a set of waypoints from the waypoints file into a numpy array
    def __init__(self) -> None:
        self.raceline_pts = []
        self.speed_targets = []
        # with open('pkg/maps/SOCHI_centerline.csv') as raceline_file:
        with open('pkg/maps/waypoints_with_speed.csv') as raceline_file:
            raceline_reader = csv.reader(raceline_file)
            for row in raceline_reader:
                self.raceline_pts.append([float(row[0]), float(row[1])])        
                print(row[0])
                self.speed_targets.append(float(row[2]))
        
        self.waypoints = np.array(self.raceline_pts)
        self.speed_targets = np.array(self.speed_targets)


    # Function called by the gym
    def process_observation(self, ranges, ego_odom):
        # Compute the unit vector of the car's orientation
        v = np.array([np.cos(ego_odom["pose_theta"]), np.sin(ego_odom["pose_theta"])])

        # Represent the car's position as a vector
        x = np.array([ego_odom["pose_x"], ego_odom["pose_y"]], dtype=np.float64)

        # Boolean array indicating which waypoints are in front of the car
        visible_waypoints = np.einsum("ij,j->i", self.waypoints - x, v) > 0.0

        # far_enough = np.linalg.norm(self.waypoints - x, axis=1) > self.MIN_LOOKAHEAD
        # Calculate the lookahead modifier
        # TODO: for future improvement, this calculation should allow the lookahead modifier to go below 1, allowing for more precise control in corners
        lookahead_modifier = self.LOOKAHEAD_CONST * ((ego_odom.get("linear_vel_x")**2 + ego_odom.get("linear_vel_y")**2) / 10)

        # Lookaheads below MIN_LOOKAHEAD aren't useful, so throw them away
        if lookahead_modifier < 1:
            lookahead_modifier = 1

        # Boolean array indicating which waypoints are far enough to use as waypoints
        far_enough = np.linalg.norm(self.waypoints - x, axis=1) > (self.MIN_LOOKAHEAD * lookahead_modifier)

        # Find the possible waypoints
        nexts = self.waypoints[visible_waypoints & far_enough]
        next_speed_targets = self.speed_targets[visible_waypoints & far_enough]

        # Get the closest of the possible waypoints
        closest_idx = np.argmin(np.linalg.norm(nexts - x, axis=1))
        closest = nexts[closest_idx] - x
        print(nexts[closest_idx])

        # Just compute the angle between the car's orientation and the vector to the closest waypoint
        alpha = normalize_angle(
            np.arctan2(closest[1], closest[0])
            - ego_odom["pose_theta"]
        )

        # Get the speed target associated with the chosen point
        print(closest_idx)
        speed_target = next_speed_targets[closest_idx]

        # Get the true lookahead distance (the distance from the car to the current lookahead point)
        lookahead_distance = np.linalg.norm(closest)

        # Calculate the steering angle according to the pure pursuit algorithm
        steering_angle = np.arctan2(2 * self.WHEELBASE_LENGTH * np.sin(alpha), lookahead_distance)

        # Calculate speed based on current turning angle (this is a really really bad way to do this)
        offset = 1
        k = 10
        speed = k / (self.R * 0.5 * abs(steering_angle) + offset)
        print(speed) # for debugging
        print(speed_target) # for debugging
        return speed_target, steering_angle


class GapFollower:
    BUBBLE_RADIUS = 160
    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 80
    MAX_LIDAR_DIST = 3000000
    STRAIGHTS_SPEED = 9.0
    CORNERS_SPEED = 6.0
    STRAIGHTS_STEERING_ANGLE = np.pi / 18  # 10 degrees

    STEP_SIZE = 100

    def __init__(self):
        # used when calculating the angles of the LiDAR data
        self.radians_per_elem = None
        self.i = 0

    def preprocess_lidar(self, ranges):
        """Preprocess the LiDAR scan array. Expert implementation includes:
        1.Setting each value to the mean over some window
        2.Rejecting high values (eg. > 3m)
        """
        self.radians_per_elem = (2 * np.pi) / len(ranges)
        # we won't use the LiDAR data from directly behind us
        proc_ranges = np.array(ranges[135:-135])
        # sets each value to the mean over a given window
        proc_ranges = (
            np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), "same")
            / self.PREPROCESS_CONV_SIZE
        )
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        return proc_ranges

    def find_max_gap(self, free_space_ranges):
        """Return the start index & end index of the max gap in free_space_ranges
        free_space_ranges: list of LiDAR data which contains a 'bubble' of zeros
        """
        # mask the bubble
        masked = np.ma.masked_where(free_space_ranges == 0, free_space_ranges)
        # get a slice for each contigous sequence of non-bubble data
        slices = np.ma.notmasked_contiguous(masked)
        max_len = slices[0].stop - slices[0].start
        chosen_slice = slices[0]
        # I think we will only ever have a maximum of 2 slices but will handle an
        # indefinitely sized list for portablility
        for sl in slices[1:]:
            sl_len = sl.stop - sl.start
            if sl_len > max_len:
                max_len = sl_len
                chosen_slice = sl
        return chosen_slice.start, chosen_slice.stop

    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indices of max-gap range, respectively
        Return index of best point in ranges
        Naive: Choose the furthest point within ranges and go there
        """
        # do a sliding window average over the data in the max gap, this will
        # help the car to avoid hitting corners
        averaged_max_gap = (
            np.convolve(
                ranges[start_i:end_i], np.ones(self.BEST_POINT_CONV_SIZE), "same"
            )
            / self.BEST_POINT_CONV_SIZE
        )
        return averaged_max_gap.argmax() + start_i

    def get_angle(self, range_index, range_len):
        """Get the angle of a particular element in the LiDAR data and transform it into an appropriate steering angle"""
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_elem
        steering_angle = lidar_angle / 2
        return steering_angle

    def process_lidar(self, ranges):
        if self.i % self.STEP_SIZE == 0:
            print(list(ranges))

        self.i += 1
        """ Process each LiDAR scan as per the Follow Gap algorithm & publish an AckermannDriveStamped Message
        """
        proc_ranges = self.preprocess_lidar(ranges)
        # Find closest point to LiDAR
        closest = proc_ranges.argmin()

        # Eliminate all points inside 'bubble' (set them to zero)
        min_index = closest - self.BUBBLE_RADIUS
        max_index = closest + self.BUBBLE_RADIUS
        if min_index < 0:
            min_index = 0
        if max_index >= len(proc_ranges):
            max_index = len(proc_ranges) - 1
        proc_ranges[min_index:max_index] = 0

        # Find max length gap
        gap_start, gap_end = self.find_max_gap(proc_ranges)

        # Find the best point in the gap
        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        # Publish Drive message
        steering_angle = self.get_angle(best, len(proc_ranges))
        if abs(steering_angle) > self.STRAIGHTS_STEERING_ANGLE:
            speed = self.CORNERS_SPEED
        else:
            speed = self.STRAIGHTS_SPEED
        # print('Steering angle in degrees: {}'.format((steering_angle / (np.pi / 2)) * 90))
        return speed, steering_angle


# drives straight ahead at a speed of 5
class SimpleDriver:
    def process_lidar(self, ranges):
        speed = 5.0
        steering_angle = 0.0
        return speed, steering_angle


# drives toward the furthest point it sees
class AnotherDriver:
    def process_lidar(self, ranges):
        # the number of LiDAR points
        NUM_RANGES = len(ranges)
        # angle between each LiDAR point
        ANGLE_BETWEEN = 2 * np.pi / NUM_RANGES
        # number of points in each quadrant
        NUM_PER_QUADRANT = NUM_RANGES // 4

        # the index of the furthest LiDAR point (ignoring the points behind the car)
        max_idx = (
            np.argmax(ranges[NUM_PER_QUADRANT:-NUM_PER_QUADRANT]) + NUM_PER_QUADRANT
        )
        # some math to get the steering angle to correspond to the chosen LiDAR point
        steering_angle = max_idx * ANGLE_BETWEEN - (NUM_RANGES // 2) * ANGLE_BETWEEN
        speed = 5.0

        return speed, steering_angle


class DisparityExtender:
    CAR_WIDTH = 0.31
    # the min difference between adjacent LiDAR points for us to call them disparate
    DIFFERENCE_THRESHOLD = 2.0
    SPEED = 5.0
    # the extra safety room we plan for along walls (as a percentage of car_width/2)
    SAFETY_PERCENTAGE = 300.0

    def preprocess_lidar(self, ranges):
        """Any preprocessing of the LiDAR data can be done in this function.
        Possible Improvements: smoothing of outliers in the data and placing
        a cap on the maximum distance a point can be.
        """
        # remove quadrant of LiDAR directly behind us
        eighth = int(len(ranges) / 8)
        return np.array(ranges[eighth:-eighth])

    def get_differences(self, ranges):
        """Gets the absolute difference between adjacent elements in
        in the LiDAR data and returns them in an array.
        Possible Improvements: replace for loop with numpy array arithmetic
        """
        differences = [0.0]  # set first element to 0
        for i in range(1, len(ranges)):
            differences.append(abs(ranges[i] - ranges[i - 1]))
        return differences

    def get_disparities(self, differences, threshold):
        """Gets the indexes of the LiDAR points that were greatly
        different to their adjacent point.
        Possible Improvements: replace for loop with numpy array arithmetic
        """
        disparities = []
        for index, difference in enumerate(differences):
            if difference > threshold:
                disparities.append(index)
        return disparities

    def get_num_points_to_cover(self, dist, width):
        """Returns the number of LiDAR points that correspond to a width at
        a given distance.
        We calculate the angle that would span the width at this distance,
        then convert this angle to the number of LiDAR points that
        span this angle.
        Current math for angle:
            sin(angle/2) = (w/2)/d) = w/2d
            angle/2 = sininv(w/2d)
            angle = 2sininv(w/2d)
            where w is the width to cover, and d is the distance to the close
            point.
        Possible Improvements: use a different method to calculate the angle
        """
        angle = 2 * np.arcsin(width / (2 * dist))
        num_points = int(np.ceil(angle / self.radians_per_point))
        return num_points

    def cover_points(self, num_points, start_idx, cover_right, ranges):
        """'covers' a number of LiDAR points with the distance of a closer
        LiDAR point, to avoid us crashing with the corner of the car.
        num_points: the number of points to cover
        start_idx: the LiDAR point we are using as our distance
        cover_right: True/False, decides whether we cover the points to
                     right or to the left of start_idx
        ranges: the LiDAR points

        Possible improvements: reduce this function to fewer lines
        """
        new_dist = ranges[start_idx]
        if cover_right:
            for i in range(num_points):
                next_idx = start_idx + 1 + i
                if next_idx >= len(ranges):
                    break
                if ranges[next_idx] > new_dist:
                    ranges[next_idx] = new_dist
        else:
            for i in range(num_points):
                next_idx = start_idx - 1 - i
                if next_idx < 0:
                    break
                if ranges[next_idx] > new_dist:
                    ranges[next_idx] = new_dist
        return ranges

    def extend_disparities(self, disparities, ranges, car_width, extra_pct):
        """For each pair of points we have decided have a large difference
        between them, we choose which side to cover (the opposite to
        the closer point), call the cover function, and return the
        resultant covered array.
        Possible Improvements: reduce to fewer lines
        """
        width_to_cover = (car_width / 2) * (1 + extra_pct / 100)
        for index in disparities:
            first_idx = index - 1
            points = ranges[first_idx : first_idx + 2]
            close_idx = first_idx + np.argmin(points)
            far_idx = first_idx + np.argmax(points)
            close_dist = ranges[close_idx]
            num_points_to_cover = self.get_num_points_to_cover(
                close_dist, width_to_cover
            )
            cover_right = close_idx < far_idx
            ranges = self.cover_points(
                num_points_to_cover, close_idx, cover_right, ranges
            )
        return ranges

    def get_steering_angle(self, range_index, range_len):
        """Calculate the angle that corresponds to a given LiDAR point and
        process it into a steering angle.
        Possible improvements: smoothing of aggressive steering angles
        """
        lidar_angle = (range_index - (range_len / 2)) * self.radians_per_point
        steering_angle = np.clip(lidar_angle, np.radians(-90), np.radians(90))
        return steering_angle

    def process_lidar(self, ranges):
        """Run the disparity extender algorithm!
        Possible improvements: varying the speed based on the
        steering angle or the distance to the farthest point.
        """
        self.radians_per_point = (2 * np.pi) / len(ranges)
        proc_ranges = self.preprocess_lidar(ranges)
        differences = self.get_differences(proc_ranges)
        disparities = self.get_disparities(differences, self.DIFFERENCE_THRESHOLD)
        proc_ranges = self.extend_disparities(
            disparities, proc_ranges, self.CAR_WIDTH, self.SAFETY_PERCENTAGE
        )
        steering_angle = self.get_steering_angle(proc_ranges.argmax(), len(proc_ranges))
        speed = self.SPEED
        return speed, steering_angle


class LongestDistanceFollower:
    def __init__(self) -> None:
        pass

    def process_observation(
        self, ranges: np.ndarray = None, ego_odom=None
    ) -> Tuple[float, float]:
        speed, steering_angle = 10, -0.1
        fov = 4.7
        angles = np.linspace(-fov / 2, fov / 2, num=ranges.size)

        cond_arr = np.abs(angles) <= np.pi / 2
        masked_dist = np.where(cond_arr, ranges, 0)
        longest_idx = np.argmax(masked_dist)

        return speed, angles[longest_idx]


class WallFollower:
    CAR_WIDTH: float = 0.31
    CAR_LENGTH: float = 0.329
    FOV: float = 4.7
    DT: float = 1e-3
    ANGLES = np.linspace(-FOV / 2, FOV / 2, num=1080)

    def __init__(self) -> None:
        self.vel, self.steer_angle = 15.0, 0.0

    def how_bad_is_steer_angle(
        self, ranges: npt.NDArray[np.float64], steer_angle: float
    ) -> float:
        points = np.stack(
            [np.cos(self.ANGLES) * ranges, np.sin(self.ANGLES) * ranges], axis=-1
        )
        velocity = np.array(
            [np.cos(steer_angle) * self.vel, np.sin(steer_angle) * self.vel]
        )
        closest_distance2: float = ranges[np.argmin(ranges)] ** 2

        # We aim to maintain the closest distance (which should be the wall, if not, well rip)
        # Compute where we will be in `DT` seconds
        # If our steering angle is less than epsilon (1e-4). then approximate as traversing a straight line
        if abs(self.steer_angle) < 1e-6:
            points -= velocity * self.vel
            new_closest_distance2 = np.min(np.sum(points * points, axis=-1))
            return float(
                new_closest_distance2 < (self.CAR_WIDTH * 1.3) ** 2
            ) * 100 + abs(new_closest_distance2 - closest_distance2)
        # Otherwise, use R = CAR_LENGTH cot(angle)
        elif self.steer_angle > 0:
            radius = self.CAR_LENGTH / np.tan(steer_angle)
            theta = self.vel * self.DT / radius
            points -= radius * np.array([np.cos(theta) - 1, np.sin(theta)])
            rotmat = np.array(
                [
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)],
                ]
            )

        else:
            radius = -self.CAR_LENGTH / np.tan(steer_angle)
            theta = self.vel * self.DT / radius
            points -= radius * np.array([1 - np.cos(theta), np.sin(theta)])
            rotmat = np.array(
                [
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)],
                ]
            )

        points @= rotmat
        new_closest_distance2 = np.min(np.sum(points * points, axis=-1))
        return float(
            new_closest_distance2
            < ((self.CAR_WIDTH**2 + self.CAR_LENGTH**2) ** 0.5 * 1.3) ** 2
        ) * 100 + abs(new_closest_distance2 - closest_distance2)

    def process_observation(
        self, ranges: npt.NDArray[np.float64], ego_odom=None
    ) -> Tuple[float, float]:
        possible_steering_angles = np.linspace(0, np.sqrt(np.pi) / np.sqrt(8), 32)
        possible_steering_angles = np.hstack(
            [-(possible_steering_angles**2), possible_steering_angles**2]
        )

        scores = [
            self.how_bad_is_steer_angle(ranges, angle)
            for angle in possible_steering_angles
        ]
        # print(scores)
        idx = np.argmin(scores)
        angle = possible_steering_angles[idx]
        print(angle, scores[idx])
        return self.vel, angle
