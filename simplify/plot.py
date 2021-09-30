# thanks to Corentin Ferry for this file
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import numpy
from islplot.support import *
from islpy import *
import colorsys


def mpy_list_scalar(l, x):
    return [x * i for i in l]


def float_list_to_rgb_string(fl):
    scaled = mpy_list_scalar(fl, 255.0)
    hex_str = ['{:02x}'.format(int(i)) for i in scaled]
    w = "".join(hex_str)
    
    return "#" + w


def plot_points(ax, points, color, size):
    xs = list([point[0] for point in points])
    ys = list([point[1] for point in points])
    zs = list([point[2] for point in points])
    ax.scatter(xs, ys, zs, c=color, s=size)


def plot_bset_3d_points(ax, bset_data, only_hull, color, size):
    """
    Plot the individual points of a three dimensional isl basic set.
    :param bset_data: The islpy.BasicSet to plot.
    :param only_hull: Only plot the points on the hull of the basic set.
    """
    bset_data = bset_data.convex_hull()
    #color = colors[0]['light']
    points = bset_get_points(bset_data, only_hull)
    plot_points(ax, points, color, size)


def plot_set_3d_vertices(ax, vertices, color):
    # color = colors[0]['dark']
    plot_points(ax, vertices, color, 8.0)

def plot_set_3d_shape(ax, vertices, faces, show_border, color, alpha):
    #color = colors[0]['base']
    
    for f in faces:
        vertices_in_order = []
        for p in f:
            vertices_in_order.append(vertices[p])
        tri = art3d.Poly3DCollection([vertices_in_order])
        tri.set_facecolor(color)
        #tri.set_alpha(alpha)
        if show_border:
            tri.set_edgecolor('k')
        ax.add_collection3d(tri)


def plot_set_3d(sets_data, fig=None, alpha=0.3, show_vertices=True, show_points=True,
        show_shape=True, show_border=True, only_hull_points=True, face_colors=None, face_alpha=None, 
                points_size=1.0, points_color="#000000"):
    """
    This function plots a three dimensional convex set.
    :param set_data: The set to illustrate.
    :param show_vertices: Show the vertices at the corner of the set.
    :param show_points: Show the full integer points contained in the set.
    :param show_shape: Show the faces that form the bounds of the set.
    :param show_border: Show the border of a tile.
    :param only_hull_points: Only show the points on the hull of the basic
                             set.
    """
    
    if fig == None:
        fig_n = plt.figure()
    else:
        fig_n = fig
    ax = fig_n.gca(projection='3d')
    # index = 0
    for index in range(len(sets_data)):
        set_data = sets_data[index]
        if face_alpha == None:
            face_color_alpha = alpha
        else:
            face_color_alpha = face_alpha[index]
        
        if face_colors == None:
            face_color_val = [numpy.random.random_sample() for i in range(3)] # rand(3) # was 3
            face_color_val.append(face_color_alpha)
            face_color = float_list_to_rgb_string(face_color_val)
        else:
            face_color = face_colors[index]
            
        #vertex_color_val = mpy_list_scalar(face_color_val, 0.2)
        vertex_color_val = [0.3, 0.3, 0.3]
        #point_color_val = mpy_list_scalar(face_color_val, 0.5)
        #point_color_val = [0.5, 0.5, 0.5]
        vertex_color = float_list_to_rgb_string(vertex_color_val)
        point_color = points_color # float_list_to_rgb_string(point_color_val)
        
        vertices, faces = get_vertices_and_faces(set_data)
        if show_vertices:
            plot_set_3d_vertices(ax, vertices, vertex_color)
        if show_shape:
            plot_set_3d_shape(ax, vertices, faces, show_border, face_color, face_color_alpha)
        if show_points:
            plot_bset_3d_points(ax, set_data, only_hull_points, point_color, points_size)
        
        #index += 1
        
    return fig_n


def plot_axis():
    return None


def get_N_HexCol(N=5, alpha=1):
    HSV_tuples = [(x * 1.0 / N, 0.5, 0.5) for x in range(N)]
    hex_out = []
    for rgb in HSV_tuples:
        rgb = map(lambda x: int(x * 255), colorsys.hsv_to_rgb(*rgb))
        rgba_l = list(rgb)
        rgba_l.append(int(alpha * 255))
        rgba = tuple(rgba_l)
        hex_out.append('#%02x%02x%02x%02x' % tuple(rgba))
    return hex_out


def plot_3d_sets(L, alpha=0.6, fig=None, split=None, split_border=False, split_alpha=0.5):
    try:
        plot = plot_set_3d(L, fig=fig, show_points=False, show_vertices=True, show_shape=True, alpha=alpha)
        if split:
            plot = plot_set_3d([split], fig=plot, show_points=False, show_vertices=False, show_shape=True, show_border=split_border, alpha=split_alpha)
        ax = plot.gca(projection='3d')
        ax.set_xlabel('$i$')
        ax.set_ylabel('$j$')
        ax.set_zlabel('$k$')
        ax.view_init(20,-120)
        return ax
    except:
        pass