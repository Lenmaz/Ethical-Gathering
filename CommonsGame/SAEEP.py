import numpy as np
from ConvexHullValueIteration import CH_value_iteration
from new_utils import agent_positions, new_state

def ethical_embedding_state(hull, tolerance=1.0):
    """
    Ethical embedding operation for a single state. Considers the points in the hull of a given state and returns
    the ethical weight that guarantees optimality for the ethical point of the hull

    :param hull: set of 2-D points, coded as a numpy array
    :return: the etical weight w, a positive real number
    """

    w = 0.0

    if len(hull) < 2:
        return w
    else:


        ethically_sorted_hull = hull[hull[:,1].argsort()]



        best_ethical = ethically_sorted_hull[-1]
        second_best_ethical = [-9999, -9999]



        for point in reversed(ethically_sorted_hull[:-1]):
            #print(point, tolerance*best_ethical[1])
            if point[1] <= tolerance * best_ethical[1]:
                second_best_ethical = point
                break


        #print(best_ethical, second_best_ethical)

        individual_delta = second_best_ethical[0] - best_ethical[0]
        ethical_delta = best_ethical[1] - second_best_ethical[1]

        if ethical_delta != 0:
            w = individual_delta/ethical_delta

        return w


def ethical_embedding(hull, epsilon, tolerance=1.0):
    """
    Repeats the ethical embedding process for each state in order to select the ethical weight that guarantees
    that all optimal policies are ethical.

    :param hull: the convex-hull-value function storing a partial convex hull for each state. The states are adapted
    to the public civility game.
    :param epsilon: the epsilon positive number considered in order to guarantee ethical optimality (it does not matter
    its value as long as it is greater than 0).
    :return: the desired ethical weight
    """
    ""

    w = 0.0

    #for state in range(len(hull)):
    #

    # We only consider the initial states
    for pos in agent_positions:
        state = new_state(0, [0], True, forced_agent_apples=0, forced_grass=[True, True, True], forced_ag_pos=pos)

        print(pos, state)
        print(hull[state])

        #print("---Now we simplify----")
        #import convexhull
        #new_hull = convexhull.check_descent(hull[state], tolerance=0.9)
        #print(new_hull)

        w_temp = ethical_embedding_state(hull[state], tolerance=tolerance)

        print(w_temp)

        print("---------")
        w = max(w, w_temp)

    return w + epsilon


def Single_Agent_Ethical_Environment_Designer(tabularRL, target_joint_policy, epsilon, who_is_the_learning_agent=0, discount_factor=1.0, max_iterations=5, tolerance=1.0):
    """
    Calculates the Ethical Environment Designer in order to guarantee ethical
    behaviours in value alignment problems.


    :param env: Environment of the value alignment problem encoded as an MOMDP
    :param epsilon: any positive number greater than 0. It guarantees the success of the algorithm
    :param discount_factor: discount factor of the environment, to be set at discretion
    :param max_iterations: convergence parameter, the more iterations the more probabilities of precise result
    :return: the ethical weight that solves the ethical embedding problem
    """

    #ethical_weight = 0.0
    hull = CH_value_iteration(tabularRL, discount_factor, max_iterations, tolerance=tolerance)

    ethical_weight = ethical_embedding(hull, epsilon, tolerance=tolerance) # if every state can be a initial state

    return ethical_weight + epsilon


if __name__ == "__main__":

    epsilon = 0.1
    #who_is_the_learning_agent = 0
    discount_factor = 0.8
    max_iterations = 4
    tolerance = 0.9

    tabularRL = True
    #target_joint_policy = [np.load("policy_0.npy"), np.load("policy_1.npy")]

    w_E = Single_Agent_Ethical_Environment_Designer(tabularRL, None, epsilon, None, discount_factor, max_iterations, tolerance)

    print("Ethical weight if every state can be initial: ", w_E)
