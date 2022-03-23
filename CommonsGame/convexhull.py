from scipy.spatial import ConvexHull
import numpy as np


def get_hull(points):
    """

    Get_hull calculates the positive convex hull of a set of points, limiting it to only consider weights of the form
    (1, x, x) with x >= 0. If the number of points is too small to calculate the convex hull, the program will simply
    return the original points.

    :param points: set of 2-D points, they need to be numpy arrays
    :return: new set of 2-D points, the vertices of the calculated convex hull
    """
    try:
        hull = ConvexHull(points)

        vertices = []
        for vertex in hull.vertices:
            #print(points[vertex])
            vertices.append(points[vertex])

        vertices = np.array(vertices)

        best_individual = np.argmax(vertices[:, 0])

        #Calculating best ethical
        best_ethical = -1
        chosen_ethical = np.max(vertices[:, 1])

        where_ethical = np.argwhere(vertices[:, 1] == chosen_ethical)[:, 0]
        chosen_individual = np.max(vertices[where_ethical][:, 0])

        for i in range(len(vertices)):
            if vertices[i][0] == chosen_individual and vertices[i][1] == chosen_ethical:
                best_ethical = i

        #print(best_individual, best_ethical)

        if best_ethical < best_individual:
            vertices = np.concatenate((vertices[best_individual:], vertices[:best_ethical+1]),0)
        else:
            vertices = vertices[best_individual:best_ethical + 1]

        return vertices
    except:
        points = np.array(points)

        if np.max(points[:, 1]) == np.min(points[:, 1]):
            best_individual = np.argmax(points[:, 0])

            return np.array([points[best_individual]])

        return points


def translate_hull(point, gamma, hull):
    """
    From Barret and Narananyan's 'Learning All Optimal Policies with Multiple Criteria' (2008)

    Translation and scaling operation of convex hulls (definition 1 of the paper).

    :param point: a 2-D numpy array
    :param gamma: a real number
    :param hull: a set of 2-D points, they need to be numpy arrays
    :return: the new convex hull, a new set of 2-D points
    """

    if len(hull) == 0:
        hull = [point]
    else:
        for i in range(len(hull)):
            hull[i] = np.multiply(hull[i], gamma, casting="unsafe")
            if point == []:
                pass
            else:
                hull[i] = np.add(hull[i], point,casting="unsafe")
    return hull


def sum_hulls(hull_1, hull_2):
    """
    From Barret and Narananyan's 'Learning All Optimal Policies with Multiple Criteria' (2008)

    Sum operation of convex hulls (definition 2 of the paper)

    :param hull_1: a set of 2-D points, they need to be numpy arrays
    :param hull_2: a set of 2-D points, they need to be numpy arrays
    :return: the new convex hull, a new set of 2-D points
    """

    new_points = None

    for i in range(len(hull_1)):
        if new_points is None:
            new_points = translate_hull(hull_1[i].copy(), 1,  hull_2.copy())
        else:
            new_points = np.concatenate((new_points, translate_hull(hull_1[i].copy(), 1, hull_2.copy())), axis=0)

    return get_hull(new_points)

def max_q_value(weight, hull):
    """
    From Barret and Narananyan's 'Learning All Optimal Policies with Multiple Criteria' (2008)

    Extraction of the Q-value (definition 3 of the paper)

    :param weight: a weight vector, can be simply a list of floats
    :param hull: a set of 2-D points, they need to be numpy arrays
    :return: a real number, the best Q-value of the hull for the given weight vector
    """
    scalarised = []

    for i in range(len(hull)):
        f = np.dot(weight,hull[i])
        #print(f)
        scalarised.append(f)

    scalarised = np.array(scalarised)

    return np.max(scalarised)


def check_descent(lista_total, first_index=-1, pendiente_comparada=None, tolerance=0.99):

    if len(lista_total) + (first_index - 2) < 0:
        return lista_total

    pendientes = [0, 0]
    #print(lista_total[first_index], lista_total[first_index-1],lista_total[first_index-2])

    for i in range(1, 3):

        dy = lista_total[first_index][1] - lista_total[first_index-i][1]
        dx = lista_total[first_index][0] - lista_total[first_index-i][0]
        #print("dy, dx : ", dy, dx)
        pendientes[i-1] = dy/dx

    if pendiente_comparada is not None:
        pendientes[0] = pendiente_comparada # de esta forma seguro que el second-best nuevo tiene un weight muy parecido al second-best original

    rel_diff_pendientes = pendientes[0]/pendientes[1]
    diff_pendientes = np.abs(pendientes[1] - pendientes[0])
    #print("pendientes :", pendientes[1], pendientes[0], rel_diff_pendientes, diff_pendientes)
    if rel_diff_pendientes > tolerance:


        lista_total = np.delete(lista_total, first_index-1, axis=0)
        return check_descent(lista_total, first_index, pendiente_comparada=pendientes[0], tolerance=tolerance)
    else:
        return check_descent(lista_total, first_index-1, tolerance=tolerance)

if __name__ == "__main__":

    bad_faith = [[0.0, 1.0], [1.0, 0.98], [2.0, 0.95], [3.0, 0.92], [4.0, 0.87], [5.0, 0.82], [6.0, 0.78]]

    puntitos = [[2.1662672,  -0.42611422],
                 [2.16427041, -0.34990712],
                 [2.16416072, -0.34933476],
                 [2.16398476, -0.34854124],
                 [2.08666855, -0.01315881],
                 [2.07982331,  0.00407588],
                 [2.07155883,  0.02474646],
                 [2.07022897,  0.02757596],
                 [2.0680193,  0.03142119],
                 [2.06716995,  0.03289575],
                 [2.06627643,  0.03433905],
                 [2.06478913,  0.03659349],
                 [2.06467903,  0.03675077],
                 [2.05987179,  0.04155802]]

    puntitos2 =[[ 2.50211658, -0.39815329],
 [ 2.42168534, -0.0385245 ],
 [ 2.41180445, -0.01380875],
 [ 2.37118163,  0.04387149],
 [ 2.36279464,  0.05127011],
 [ 2.35512628,  0.05339826],
 [ 2.32289227,  0.05763836],
 [ 2.31764287,  0.05799995],
 [ 2.26921571,  0.05815388],
 [ 2.17414093,  0.05822659],
 [ 2.1694898,   0.05822848]]

    puntitos = np.array(bad_faith)

    print(len(puntitos))
    nuevos_puntitos = check_descent(puntitos, -1, tolerance=0.90)

    print("yeaaa: ", nuevos_puntitos)
    print(len(nuevos_puntitos))
    #vertices = get_hull(nuevos_puntitos)
    vertices = nuevos_puntitos

    import matplotlib.pyplot as plt
    plt.plot(vertices[:, 0], vertices[:, 1], 'o',color="blue")
    plt.plot(vertices[:,0], vertices[:,1], 'k-')
    max_q_value([1.0,0.4],vertices)
    plt.show()
