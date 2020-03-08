from window import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

N_long = 2048
N_short = 256
pad = N_long//4 - N_short//4

def plot_window(window, title, file, N=N_long):
    win = window(np.ones(N))
    plt.figure()
    plt.plot(win)
    plt.title(title)
    plt.xlabel('Time [samples]')
    plt.savefig('Figs/' + file + '.png')

# plot_window(SineWindow, 'Long Sine Window', 'sine')
# plot_window(lambda x : StartWindow(x, N_long, N_short), 'Start Window', 'start')
# plot_window(lambda x : StopWindow(x, N_long, N_short), 'Stop Window', 'stop')

plt.figure()
time = np.arange(2048*3)
def add_to_plot(window, off, N, label, color):
    win = np.zeros_like(time, dtype=float)
    win[off:off+N] += window(np.ones(N))
    plt.plot(win, label=label, color=color)

add_to_plot(SineWindow, 0, N_long, label='Long', color='g')
add_to_plot(lambda x : StartWindow(x, N_long, N_short), 1024, N_long, label='Start', color='b')

for n in range(8):
    add_to_plot(SineWindow, 2048 + pad + 128*n, N_short, label='Short', color='r')

add_to_plot(lambda x : StopWindow(x, N_long, N_short), 3072, N_long, label='Stop', color='k')
add_to_plot(SineWindow, 4096, N_long, label='Long', color='g')

custom_lines = [Line2D([0], [0], color='g', lw=2),
                Line2D([0], [0], color='b', lw=2),
                Line2D([0], [0], color='r', lw=2),
                Line2D([0], [0], color='k', lw=2)]
plt.legend(custom_lines, ['Long', 'Start', 'Short', 'Stop'], loc='upper right')
plt.xlabel('Time [samples]')
plt.title('Edler Style Block Switch')
plt.savefig('Figs/block-switch.png')

plt.show()
