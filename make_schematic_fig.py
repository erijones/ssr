#!/usr/bin/env python3

import pickle
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.transforms as transforms
import mpl_toolkits.axisartist.floating_axes as floating_axes
import numpy as np
from mpl_toolkits import mplot3d
import scipy.interpolate as interpolate
import scipy.integrate as integrate

def make_landscape():

    xs = np.linspace(-1, 2, 100)
    ys = np.linspace(-1.5, 1.5, 100)
    X, Y = np.meshgrid(xs, ys)

    Z = np.zeros(np.shape(X))

    # underlying quadratic landscape
    new_Z = np.zeros(np.shape(Z))
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            new_Z[j][i] = 2.2*((x - .1)**2 + (y + .5)**2)
            Z[j][i] = Z[j][i] + new_Z[j][i]

    # steady state A at (0, 1)
    new_Z = np.zeros(np.shape(Z))
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            new_Z[j][i] = -1*np.exp(-(x**2 + (y-1)**2)*50)
            Z[j][i] = Z[j][i] + new_Z[j][i]

    # steady state A at (0, 1)
    new_Z = np.zeros(np.shape(Z))
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            new_Z[j][i] = -2*np.exp(-(.2*x**2 + .01*(y-1)**2)*50)
            Z[j][i] = Z[j][i] + new_Z[j][i]

    # steady state B at (sqrt(3), 0)
    new_Z = np.zeros(np.shape(Z))
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):

            xp = x - np.sqrt(3)
            yp = y
            translated_vec = np.array([xp, yp])

            A = np.array([[np.cos(np.pi/6), -np.sin(np.pi/6)],
                         [np.sin(np.pi/6), np.cos(np.pi/6)]])

            rot_vec = np.dot(A, translated_vec)
            u, v = rot_vec

            new_Z[j][i] = -2.7*np.exp(-(.03*u**2 + v**2)*8)
            Z[j][i] = Z[j][i] + new_Z[j][i]

    # deepen steady state B at (sqrt(3), 0)
    new_Z = np.zeros(np.shape(Z))
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            new_Z[j][i] = -2.1*np.exp(-((x-np.sqrt(3))**2 + y**2)*10)
            Z[j][i] = Z[j][i] + new_Z[j][i]

    # steady state C at (0, -1)
    new_Z = np.zeros(np.shape(Z))
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            xp = x
            yp = y + 1
            translated_vec = np.array([xp, yp])

            A = np.array([[np.cos(-np.pi/6), -np.sin(-np.pi/6)],
                         [np.sin(-np.pi/6), np.cos(-np.pi/6)]])

            rot_vec = np.dot(A, translated_vec)
            u, v = rot_vec

            new_Z[j][i] = -2*np.exp(-(.01*u**2 + 1*v**2)*10)
            Z[j][i] = Z[j][i] + new_Z[j][i]

    # deepen steady state C at (0, -1)
    new_Z = np.zeros(np.shape(Z))
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            new_Z[j][i] = -1*np.exp(-(x**2 + (y+1)**2)*10)
            Z[j][i] = Z[j][i] + new_Z[j][i]

    # hump at (0, -.5)
    new_Z = np.zeros(np.shape(Z))
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            new_Z[j][i] = 2.8*np.exp(-(.1*(x-.3)**2 + .1*(y+.5)**2)*5)
            Z[j][i] = Z[j][i] + new_Z[j][i]


    # hump at (0, 0)
    new_Z = np.zeros(np.shape(Z))
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            new_Z[j][i] = .3*np.exp(-(.1*(x-.4)**2 + .1*(y-.3)**2)*3)
            Z[j][i] = Z[j][i] + new_Z[j][i]

    # hump at (0, -1)
    new_Z = np.zeros(np.shape(Z))
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            new_Z[j][i] = 2*np.exp(-(.1*(x+.3)**2 + .1*(y+.8)**2)*5)
            Z[j][i] = Z[j][i] + new_Z[j][i]

    #ax.contourf(X, Y, Z, levels=100, cmap='hsv')
    #plt.colorbar(cax=ax)

    #plt.savefig('figs/schematic_v1.pdf')

    #plt.figure()

    return xs, ys, Z

def plot_landscape(xs, ys, Z, ax):
    X, Y = np.meshgrid(xs, ys)

    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            if Z[j][i] > 5.5:
                Z[j][i] = np.nan
            if y - x/np.sqrt(3) < -1.1:
                Z[j][i] = np.nan
            if x < -.5:
                Z[j][i] = np.nan

    #ax = plt.axes(projection='3d')
    ax.axis('off')
    #cbar = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', linewidth=0)
    cbar = ax.contour3D(X, Y, Z, 100, cmap='viridis', linewidths=.4, zorder=-1)
    ax.contour3D(X, Y, Z, 10, cmap='viridis', linewidths=1.7, zorder=-1)

    #plt.colorbar(cbar)
    #ax.view_init(elev=70, azim=260)
    ax.view_init(elev=70, azim=290)

    ax.set_xlim3d([-.5, 1.5])
    ax.set_ylim3d([-.5, 1.5])
    #plt.savefig('figs/schematic_v0.pdf')


def make_rotated_inset(fig, rotation, loc, data):
    tr = transforms.Affine2D().scale(1, 1).rotate(rotation)
    grid_helper = floating_axes.GridHelperCurveLinear(tr, extremes=(0,
        data[:,0].min(), 1, 1.1*data[:,0].max()))

    ax1_inset = floating_axes.FloatingSubplot(fig, 111, grid_helper=grid_helper)
    ax1_inset.set_position(loc)
    for side in ['top', 'right']:
        #ax1_inset.axis[side].set_axisline_style(None)
        #ax1_inset.axis[side].toggle(all=False)
        ax1_inset.axis[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax1_inset.axis[side].toggle(all=False, label=True)
        ax1_inset.axis[side].set_axisline_style('-|>', size=2)
        ax1_inset.axis[side].line.set_facecolor('black')
        ax1_inset.axis[side].label.set_fontsize(8)
        ax1_inset.axis[side].major_ticklabels.set_fontsize(8)
        #ax1_inset.axis[side].major_ticks.set_markevery(every=[0, 1, 2])

    ax1_inset.plot(data[:,1], data[:,0])
    ax1_inset.set_xlabel('$\\vec{x}$', fontsize=8)
    ax1_inset.set_ylabel('$V(\\vec{x})$', fontsize=8)

    f1 = fig.add_subplot(ax1_inset)

def integrand(y, t, interp_grad_x, interp_grad_y):
    dydt = [-interp_grad_y(*y)[0], -interp_grad_x(*y)[0]]
    return dydt

def get_steady_states(xs, ys, Z, ax=None):
    gradZ = np.gradient(Z, xs, ys)
    interp_grad_x = interpolate.interp2d(xs, ys, gradZ[0], kind='cubic')
    interp_grad_y = interpolate.interp2d(xs, ys, gradZ[1], kind='cubic')
    interp_Z = interpolate.interp2d(xs, ys, Z, kind='cubic')

    t = np.linspace(0, 10, 1001)

    steady_states = []

    for ic in [[.8, .5], [0, .8], [0, -.8]]:
        sample_traj = integrate.odeint(integrand, ic, t, args=(interp_grad_x, interp_grad_y))
        full_traj = np.array([[v[0], v[1], interp_Z(v[0], v[1])[0]] for v in sample_traj])
        steady_states.append(full_traj[-1])
        #ax1_inset.plot(full_traj[:,0], full_traj[:,1], full_traj[:,2])

    steady_states = np.array(steady_states)
    return steady_states


def plot_steady_states(steady_states, ax):
    colors = ['blue', 'orange', 'green']

    if ax:
        for i,ss in enumerate(steady_states):
            print(ss)
            ax.plot([ss[0]], [ss[1]], [ss[2]], '.', color=colors[i], ms=40, zorder=1)

def create_basins(xs, ys, Z):
    N = 200
    xss = np.linspace(xs[0], xs[-1], N+1)
    yss = np.linspace(ys[0], ys[-1], N+1)

    gradZ = np.gradient(Z, xs, ys)
    interp_grad_x = interpolate.interp2d(xs, ys, gradZ[0], kind='cubic')
    interp_grad_y = interpolate.interp2d(xs, ys, gradZ[1], kind='cubic')
    interp_Z = interpolate.interp2d(xs, ys, Z, kind='cubic')

    t = np.linspace(0, 50, 101)

    read_data = True
    filename = 'basins_N_{}'.format(N)

    if not read_data:
        basin_dict = {}
        for xx in xss:
            print(xx)
            for yy in yss:
                if interp_Z(xx, yy) > 5.5:
                    continue
                if xx < -.5:
                    continue
                if yy - xx/np.sqrt(3) < -1.1:
                    continue

                sample_traj = integrate.odeint(integrand, [xx, yy], t, args=(interp_grad_x, interp_grad_y))
                outcome = [*sample_traj[-1], interp_Z(*sample_traj[-1])[0]]
                basin_dict[xx, yy] = outcome

        with open('data/{}.pi'.format(filename), 'wb') as f:
            pickle.dump(basin_dict, f)
    else:
        with open('data/{}.pi'.format(filename), 'rb') as f:
            basin_dict  = pickle.load(f)

    return xss, yss, basin_dict

def plot_basins(basin_dict, steady_states, xss, yss, Z, ax):
    interp_Z = interpolate.interp2d(xs, ys, Z, kind='cubic')
    X, Y = np.meshgrid(xss, yss)

    colors = ['blue', 'orange', 'green']
    eps = 1e-2
    for k,ss in enumerate(steady_states):
        new_Z = np.zeros(np.shape(X))
        for i,x in enumerate(xss):
            for j,y in enumerate(yss):
                try:
                    if np.linalg.norm(basin_dict[x, y] - ss) < eps:
                        new_Z[j][i] = interp_Z(x, y)[0]
                    else:
                        new_Z[j][i] = np.nan
                except KeyError:
                    new_Z[j][i] = np.nan

        ax.contour(X, Y, new_Z, colors=colors[k], levels=50, corner_mask=False,
                linewidths=4, alpha=.15)
        ax.contourf(X, Y, new_Z, colors=colors[k], levels=50,
                corner_mask=False, alpha=.15)

def get_connectors(xs, ys, Z, steady_states):
    interp_Z = interpolate.interp2d(xs, ys, Z, kind='cubic')

    all_connectors = []

    for i,j in [[0, 1], [0, 2], [1, 2]]:
        ssa = steady_states[i]
        ssb = steady_states[j]
        connector_list = []

        for p in np.linspace(0, 1, 101):
            xx, yy = (p*ssa + (1-p)*ssb)[:2]
            zz = interp_Z(xx, yy)[0]
            connector_list.append([xx, yy, zz, p])
        all_connectors.append(connector_list)

    all_connectors = np.array(all_connectors)

    filename = 'all_connectors'
    with open('data/{}.pi'.format(filename), 'wb') as f:
        pickle.dump(all_connectors, f)

    return all_connectors

def plot_connectors(all_connectors, ax):
    for traj in all_connectors:
        ax.plot(traj[:,0], traj[:,1], traj[:,2], color='k', lw=6, ls='--',
                zorder=5)

def create_attractor_network(ax):
    import matplotlib.patches as mpatches

    filename = 'all_connectors'
    with open('data/{}.pi'.format(filename), 'rb') as f:
        all_connectors = pickle.load(f)

    sep_list = np.zeros((3,3))
    for i,(con,ab) in enumerate(zip(all_connectors, [[0,1], [0,2], [1,2]])):
        max_idx = np.argmax(con[:,2])
        p_crit = con[max_idx,3]
        sep_list[ab[0], ab[1]] = p_crit
        sep_list[ab[1], ab[0]] = 1 - p_crit

    colors = ['orange', 'blue', 'green']
    labels = ['A', 'B', 'C']
    circ_size = 0.35

    circs = []; xx = []; yy = [];
    circ_xx = []; circ_yy = []

    edge = (1+circ_size)*1.2
    ax.axis([-edge, edge, -edge, edge])
    ax.set_aspect('equal')
    plt.axis('off')

    for i in range(len(labels)):
        xx.append(np.sin(2*np.pi*i/len(labels)))
        yy.append(np.cos(2*np.pi*i/len(labels)) - .2)
        circ_xx.append(1.3*np.sin(2*np.pi*i/len(labels)))
        circ_yy.append(1.3*np.cos(2*np.pi*i/len(labels)) - .2)

        circs.append(plt.Circle((circ_xx[-1], circ_yy[-1]), circ_size, lw=0,
            color=colors[i]))
        plt.text(circ_xx[-1], circ_yy[-1], labels[i], weight='bold', ha='center',
                va='center', fontsize=28, color='white')

    for i in range(len(labels)):
        ax.add_artist(circs[i])

    # add lines connecting circles
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                continue
            arrow_color = colors[i]
            p = sep_list[i][j]
            point = np.array([xx[i]*(1-p) + xx[j]*p, yy[i]*(1-p) + yy[j]*p])
            plt.plot(point[0], point[1], marker='.', color='black',
                    markersize=20, zorder=5,
                    markerfacecolor='none')
            curve_type = 'arc3,rad=0'
            thickness = 6
            outer_color = arrow_color

            # make arrows pointing from one circle to another
            ax.plot([xx[i],point[0]], [yy[i], point[1]],
                    color=outer_color, linewidth=thickness,
                    solid_capstyle='butt',alpha = 1)

def plot_rotated_subfigs(fig):
    filename = 'all_connectors'
    with open('data/{}.pi'.format(filename), 'rb') as f:
        all_connectors = pickle.load(f)


    make_rotated_inset(fig=fig, rotation=0,
                       loc=[0, .5, 1, .5],
                       data=all_connectors[0][:,2:])
    return




if __name__ == '__main__':
    fig = plt.figure(figsize=(10, 6))

    if True:
        ax1 = plt.axes([-.03, 0, .7, 1], projection='3d')

        xs, ys, Z = make_landscape()

        steady_states = get_steady_states(xs, ys, Z, ax1)
        xss, yss, basin_dict = create_basins(xs, ys, Z)
        all_connectors = get_connectors(xs, ys, Z, steady_states)

        plot_basins(basin_dict, steady_states, xss, yss, Z, ax1)
        plot_landscape(xs, ys, Z, ax1)
        plot_connectors(all_connectors, ax1)

        #plot_steady_states(steady_states, ax1)

    if True:
        ax2 = plt.axes([.62, .05, .38, .8])

        create_attractor_network(ax2)

        #plot_rotated_subfigs(fig)

    


    plt.savefig('figs/combined_schematic_v0.pdf')


    #make_landscape_1()

