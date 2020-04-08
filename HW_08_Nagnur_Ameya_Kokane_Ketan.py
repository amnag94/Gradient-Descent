import numpy as np
import math
import matplotlib.pyplot as plt

INITIAL_THETA = 58
INITIAL_RHO = 10.75
INITIAL_ALPHA = 4.5

MINIMUM_ALPHA_DELTA = 0.025
LEARNING_RATE = 0.90
MINIMUM_DIFF = 0.005


def dist_pts_to_line(points, theta, rho):
    n = len(points)
    A = math.cos(theta)
    B = math.sin(theta)
    C = -rho

    # points contains first list as x values and
    # second list as y values
    total_dist = np.sum([abs(A * point[0] + B * point[1] + C) for point in points])

    avg_dist = total_dist / n

    return avg_dist

def grad_descent_rho(points, theta, rho, alpha, init_dist):
    dist_minus_alpha = dist_pts_to_line(points, theta, rho - alpha)
    dist_plus_alpha = dist_pts_to_line(points, theta, rho + alpha)

    # print(init_dist, dist_minus_alpha, dist_plus_alpha)

    # Find better direction to go in
    # If neither then return original values
    if dist_minus_alpha < init_dist:
        step = -alpha
        dist_best = dist_minus_alpha
    elif dist_plus_alpha < init_dist:
        step = alpha
        dist_best = dist_plus_alpha
    else:
        return init_dist, rho

    rho += step

    # Find best distance using determined step
    while True:
        dist_after_step = dist_pts_to_line(points, theta, rho + step)

        if dist_after_step < dist_best:
            dist_best = dist_after_step
            rho += step
            #print("Dist = %f " % dist_best + "Rho = %f" % rho)
        else:
            break

    #print("Rho done")

    return dist_best, rho

def theta_wrap_around(theta):
    if theta > 180:
        theta = theta - 360
    if theta < -180:
        theta = theta + 360

    return theta

def grad_descent_theta(points, theta, rho, alpha, init_dist):
    dist_minus_alpha = dist_pts_to_line(points, theta - alpha, rho)
    dist_plus_alpha = dist_pts_to_line(points, theta + alpha, rho)

    # Find better direction to go in
    # If neither then return original values
    if dist_minus_alpha < init_dist:
        step = -alpha
        dist_best = dist_minus_alpha
    elif dist_plus_alpha < init_dist:
        step = alpha
        dist_best = dist_plus_alpha
    else:
        return init_dist, theta

    theta += step
    theta = theta_wrap_around(theta)

    # Find best distance using determined step
    while True:
        dist_after_step = dist_pts_to_line(points, theta + step, rho)

        if dist_after_step < dist_best:
            dist_best = dist_after_step
            theta += step
            # Wrap around to maintain proper value
            theta = theta_wrap_around(theta)
        else:
            break

    #print("Theta done")

    return dist_best, theta


def gradient_descent_for_line(theta, rho, alpha, points):

    distances = []

    while True:

        # Initial distance of points from line
        dist_before_rho = dist_pts_to_line(points, theta, rho)

        distances.append(dist_before_rho)

        # Find best distance with change in rho
        dist_best, rho = grad_descent_rho(points, theta, rho, alpha, dist_before_rho)

        # Find best distance with change in theta
        dist_best, theta = grad_descent_theta(points, theta, rho, alpha, dist_best)

        # Flip the line from the negative angle a positive angle
        # This negates rho, whatever that is
        if theta < 0:
            theta += 180
            rho = -rho

        # Distance after gradient descent on rho and theta
        final_avg_dist = dist_pts_to_line(points, theta, rho)

        print("Rho = %+9.5f " % rho + " Theta = %+9.6f  " % theta)
        print("Alpha = %+6.5f " % alpha + " Avg Dist = %8.7f\n" % final_avg_dist)

        if dist_before_rho < final_avg_dist:
            print('Distance getting worse, not improving')
        #elif dist_before_rho - final_avg_dist <= MINIMUM_DIFF:
            # Stopping condition 1
            #break

        alpha = LEARNING_RATE * alpha

        if alpha > MINIMUM_ALPHA_DELTA:
            continue
        else:
            # Stopping condition 2
            break

    distances.append(final_avg_dist)

    return theta, rho, distances

def plot_distances(distances, title, xlbl, ylbl):
    plt.title(title)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    plt.plot(distances)
    plt.show()

def main():

    points = [[2, 10], [3, 9], [4, 8], [5, 7], [6, 6], [7, 5], [8, 4], [9, 3], [10, 2]]

    theta, rho, distances = gradient_descent_for_line(INITIAL_THETA, INITIAL_RHO, INITIAL_ALPHA, points)

    plot_distances(distances, 'Average distance over iterations', 'Iteration', 'Average distance from points')

    dst_to_origin = dist_pts_to_line([[0, 0]], theta, rho)

    print('Answer:\n');
    print('A                           = %+7.5f \n' % math.cos(theta))
    print('B                           = %+7.5f \n' % math.sin(theta))
    print('theta                       = %+7.5f degrees\n' % theta)
    print('rho                         = %+7.5f \n' % rho)
    print('distance of line to origin  = %+7.5f \n' % dst_to_origin)

if __name__ == '__main__':
    main()