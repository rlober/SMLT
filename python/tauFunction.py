import math
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm


d = np.linspace(0, 100, 100)
delta = 1



t = np.arange(0,100,1)

t_mat, d_mat = np.meshgrid(t, d)

# tau = 2* np.log( t**((d/2)+2) + 3.14**2 / (3*delta))


fig = plt.figure(figsize=(16, 10), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(1, 2, 1, projection='3d')
surf_plot = ax.plot_surface([], [], [], rstride=5, cstride=5, cmap=cm.jet, linewidth=0, alpha=0.6, antialiased=False)
title = ax.set_title('Tau function evolution: delta = ')
ax.set_xlabel('t')
ax.set_ylabel('d')

tau = np.sqrt(2* np.log( t_mat**((d_mat/2)+2) + 3.14**2 / (3*1)))
surf_plot = ax.plot_surface(t_mat, d_mat, tau, rstride=5, cstride=5, cmap=cm.jet, linewidth=0, alpha=0.6, antialiased=False)
#
#
# # def init():
# #
# #     surf_plot = ax.plot_surface([], [], [], rstride=5, cstride=5, cmap=cm.jet, linewidth=0, alpha=0.6, antialiased=False)
# #
# #     return surf_plot
#
# def animate(frame):
#     ax.clear
#     delta = (frame+1.)/100.
#
#     tau = 2* np.log( t_mat**((d_mat/2)+2) + 3.14**2 / (3*delta))
#
#     surf_plot = ax.plot_surface(t_mat, d_mat, tau, rstride=5, cstride=5, cmap=cm.jet, linewidth=0, alpha=0.6, antialiased=False)
#     title.set_text('Tau function evolution: delta = '+str(delta))
#     return surf_plot, title
#
# anim = animation.FuncAnimation(fig, animate, frames=100, interval=10, blit=False, repeat=True)
#
# # file_name = './tauFunctionAnimation.mp4'
# #
# # anim.save(file_name, fps=30, extra_args=['-vcodec', 'libx264'])
#
#


plt.show(block=True)
