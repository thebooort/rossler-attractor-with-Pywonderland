# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:56:03 2017

@author: booort
"""

# -*- coding: utf-8 -*-

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Make 3D Animations of Lorenz Attractor with Matplotlib
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

code adapted from

"https://jakevdp.github.io/blog/2013/02/16/animating-the-lorentz-system-in-3d/"

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D


# number of particles.
num_particles = 20

# constants for Lorenz system.
alpha = 0.2
beta = 0.2
gamma = 8.0


def derivative(point, t):
    """return the tangent direction at (x,y,z)."""
    x, y, z = point
    return [-y - z,
            x + alpha * y,
            beta + z * ( x - gamma )]


fig = plt.figure(figsize=(9.1, 7.2), dpi=100)
ax = fig.add_axes([0, 0, 1, 1], projection='3d', xlim=(-70, 70),
                  ylim=(-70, 70), zlim=(-10, 70), aspect=1)
ax.view_init(30, 0)
ax.axis('off')

lines = []
points = []
colors = plt.cm.gist_ncar(np.linspace(0, 1, num_particles))

for c in colors:
    lines.extend(ax.plot([], [], '-', c=c))
    points.extend(ax.plot([], [], 'o', c=c))


x0 =  -5 + 20 * np.random.random((num_particles, 3))
for i, element in enumerate(x0):
    if element[2] != 2:
        element[2] = 0.0
t = np.linspace(0, 80, 3001)
x_t = np.array([odeint(derivative, point, t) for point in x0])


def init():
    for line, point in zip(lines, points):
        line.set_data([], [])
        line.set_3d_properties([])

        point.set_data([], [])
        point.set_3d_properties([])
    return lines + points


def animate(i):
    i = 2*i % x_t.shape[1]  # accelarate the animation.

    for line, point, x_j in zip(lines, points, x_t):
        x, y, z = x_j[:i].T
        line.set_data(x, y)
        line.set_3d_properties(z)

        # note that plot() receives a list parameter so we have
        # to write x[-1:] instead of x[-1]!
        point.set_data(x[-1:], y[-1:])
        point.set_3d_properties(z[-1:])

    ax.view_init(30, 0.3*i)
    fig.canvas.draw()
    return lines + points


anim = FuncAnimation(fig, animate, init_func=init, interval=5,
                     frames=500, blit=True)

# Note: this `dpi` has nothing to do with the dpi settings in `plt.figure()`,
# this dpi determines the size of the video together with `figsize`.
# The range of `crf` is 0~51, the lower crf the better quality and the larger
# size of the video. crf=20 already gives nice results.
anim.save('rossler.mp4', writer='ffmpeg', fps=30, dpi=200,
          codec='libx264', extra_args=['-crf', '20'])
