import numpy as np
import networkx as nx
from sklearn.neighbors import KDTree
import copy

def linear_scale(points:list, x_min:float, x_max:float, y_min:float, y_max:float):
    '''
    :param points: 
    :param x_min: 
    :param x_max: 
    :param y_min: 
    :param y_max: 
    :return: 
    '''
    px_min = min(points, key=lambda x: x[0])[0]
    px_max = max(points, key=lambda x: x[0])[0]
    py_min = min(points, key=lambda x: x[1])[1]
    py_max = max(points, key=lambda x: x[1])[1]
    x = (x_max - x_min) * 1.0 / (px_max - px_min)
    y = (y_max - y_min) * 1.0 / (py_max - py_min)
    newpoints = []
    for i in points:
        px = (i[0] - px_min) * x + x_min
        py = (i[1] - py_min) * y + y_min
        newpoints.append([px, py])
    return newpoints

def centor_of_class(class_a:np.ndarray):
    '''
    :param class_a: 
    :return: 
    '''
    centor = np.mean(class_a, axis=0)
    return centor


def divide_points_by_label(points:list, label:list):
    '''
    :param points: 
    :param label: 
    :return: list
    '''
    points = np.array(points)
    label = np.array(label)
    class_a = points[label==label[0]]
    class_b = points[label!=label[0]]
    return class_a.tolist(), class_b.tolist()

def point_to_class_distance(p:np.ndarray, class_a:np.ndarray):
    '''
    :param p: 
    :param class_a: 
    :return: 
    '''
    dis = np.linalg.norm(class_a-p, axis=1)
    return np.mean(dis)

def between_class_distance(class_a:np.ndarray, class_b:np.ndarray):
    '''
    between_class_distances a to b
    :param class_a: 
    :param class_b: 
    :return: 
    '''
    dis = []
    for i in range(len(class_a)):
        a = class_a[i]
        d = np.linalg.norm(class_b - a, axis=1)
        dis.append(np.mean(d))
    dis = np.mean(dis)
    return dis

def within_class_distance(class_a:np.ndarray):
    '''
    :param class_a: 
    :return: 
    '''
    centor = centor_of_class(class_a)
    return point_to_class_distance(centor, class_a)


def ABTN(points:list, label:list):
    '''
    Between-class average distances __Mean of the distance between a class point and class b point
    :param points: 
    :param label: 2-d array
    :return: 
    '''
    class_a, class_b = divide_points_by_label(points, label)
    class_a = np.array(class_a)
    class_b = np.array(class_b)
    return between_class_distance(class_a, class_b)

def AWTN(points:list, label:list):
    '''
    Within-class average distances
    :param points: 
    :param label: 
    :return: 
    '''
    class_a, class_b = divide_points_by_label(points, label)
    class_a = np.array(class_a)
    class_b = np.array(class_b)
    w_a = within_class_distance(class_a)
    w_b = within_class_distance(class_b)
    return np.mean([w_a, w_b])

def ABW(points:list, label:list):
    '''
    ABTN / AWTN
    :param points: 
    :param label: 
    :return: 
    '''
    return ABTN(points,label) / AWTN(points, label)

def CAL(points:list, label:list):
    '''
    Centers-of-mass between-class square distances over points-to-centers-of-mass within-class square distances
    :param points: 
    :param label: 
    :return: 
    '''
    class_a, class_b = divide_points_by_label(points, label)
    class_a = np.array(class_a)
    class_b = np.array(class_b)
    centor_a = centor_of_class(class_a)
    centor_b = centor_of_class(class_b)
    centor_square_distance = np.sum(np.square(centor_a - centor_b))
    classA_square_distance = np.square(np.linalg.norm(class_a-centor_a, axis=1))
    classB_square_distance = np.square(np.linalg.norm(class_b-centor_b, axis=1))
    a =  np.hstack((classA_square_distance,classB_square_distance))
    return centor_square_distance / np.mean(a)

def LDA(points:list, label:list):
    '''
    Centers-of-mass between-class distances over points-to-centers-of-mass within-class distances
    :param points: 
    :param label: 
    :return: 
    '''
    class_a, class_b = divide_points_by_label(points, label)
    class_a = np.array(class_a)
    class_b = np.array(class_b)
    centor_a = centor_of_class(class_a)
    centor_b = centor_of_class(class_b)
    centor_square_distance = np.sum(np.linalg.norm(centor_a-centor_b))
    classA_square_distance = np.linalg.norm(class_a - centor_a, axis=1)
    classB_square_distance = np.linalg.norm(class_b - centor_b, axis=1)
    a = np.hstack((classA_square_distance, classB_square_distance))
    return centor_square_distance / np.mean(a)

def SIL(points:list, label:list):
    '''
    Difference of between-class and within-class average distances normalized by the maximum of them 
    :param points: 
    :param label: 
    :return: 
    '''
    def _a(i:int, class_a:np.ndarray):
        ai = class_a[i]
        class_a = np.delete(class_a, i, axis=0)
        return np.mean(np.linalg.norm(class_a-ai, axis=1))
    def _b(i:int, class_a:np.ndarray, class_b:np.ndarray):
        ai = class_a[i]
        return np.mean(np.linalg.norm(class_b - ai, axis=1))
    class_a, class_b = divide_points_by_label(points, label)
    class_a = np.array(class_a)
    class_b = np.array(class_b)
    s = []
    for i in range(len(class_a)):
        ai = _a(i, class_a)
        bi = _b(i, class_a, class_b)
        s.append(ai / max(ai, bi))
    for i in range(len(class_b)):
        ai = _a(i, class_b)
        bi = _b(i, class_b, class_a)
        s.append(ai / max(ai, bi))
    return np.mean(s)

def CS(points:list, label:list):
    '''
    Average proportion of same-class neighbors of each point in minimum spanning tree
    :param points: 
    :param label: 
    :return: 
    '''
    G = nx.Graph()
    g = []
    points = np.array(points)
    for i in range(len(points)):
        for j in range(i, len(points)):
            g.append([i, j, np.linalg.norm(points[i]-points[j])])
        G.add_weighted_edges_from(g)
        g = []
    mst = list(nx.minimum_spanning_edges(G, algorithm='kruskal', data=False))
    s = np.ones(len(label))
    for i in mst:
        if label[i[0]] != label[i[1]]:
            s[i[0]] = 0
            s[i[1]] = 0
    return np.mean(s)

def CDM(points:list, label:list, k:int):
    '''
    Pixel-wise class-density differences with class-density estimated at pixel z as the inverse distance to its Kth nearest point of this class
    :param points: 
    :param label: 
    :param k: 
    :return: 
    '''
    pixel_width = 500
    def _to_continuous_density(class_a:np.ndarray):
        density = np.zeros((pixel_width, pixel_width))
        kdt = KDTree(class_a, leaf_size=10, metric='euclidean')
        X = []
        for i in range(pixel_width):
            for j in range(pixel_width):
                X.append([i,j])
        distance,_  = kdt.query(np.array(X), k=k, return_distance=True)
        for i in range(pixel_width):
            for j in range(pixel_width):
                index = i*pixel_width + j
                density[i][j] = 1.0 / distance[index][-1]
        return density
    points = linear_scale(points, 0, pixel_width, 0, pixel_width)
    class_a, class_b = divide_points_by_label(points, label)
    class_a = np.array(class_a)
    class_b = np.array(class_b)
    density_a = _to_continuous_density(class_a)
    density_b = _to_continuous_density(class_b)
    a = np.abs(density_a - density_b)
    sum = np.sum(a)
    return sum

def DC(points:list, label:list, e:float):
    '''
    Average of the class entropy for each pixel computed over the classes of its k-neighbors (here, k = e*D, D is the maximal Euclidean distance between points of the evaluated scatterplot)
    :param points: 
    :param label: 
    :param e: [0, 1] 
    :return: 
    '''
    pixel_width = 500
    k = max(int(e * np.sqrt(500 * 500 * 2)), 1)
    points = linear_scale(points, 0, pixel_width, 0, pixel_width)
    a = np.zeros((pixel_width, pixel_width))
    b = np.zeros((pixel_width, pixel_width))
    for i in range(len(points)):
        px = points[i][0]
        py = points[i][1]
        for pixel_x in range(int(max(0, px-k)), int(min(500, px+k))):
            for pixel_y in range(int(max(0, py-k)), int(min(500, py+k))):
                dis = (pixel_x-px) * (pixel_x-px) + (pixel_y-py)*(pixel_y-py)
                if dis<=k*k:
                    if label[i] == label[0]:
                        a[pixel_x][pixel_y] += 1
                    else:
                        b[pixel_x][pixel_y] += 1
    sum = np.zeros((pixel_width, pixel_width))
    for x in range(pixel_width):
        for y in range(pixel_width):
            if a[x][y]==0 or b[x][y]==0:
                sum[x][y] = 0
            else:
                p_xy = a[x][y] + b[x][y]
                _x = a[x][y] * 1.0 / p_xy
                _y = b[x][y] * 1.0 / p_xy
                h_xy = -1 * (_x * np.log2(_x) + _y * np.log2(_y))
                sum[x][y] = p_xy * h_xy
    Z = np.log2(2) * (np.sum(a)+np.sum(b))
    dc = 100 - np.sum(sum)/Z
    return dc


