#!/usr/bin/env python
"""Simulate MDS uncertainty.

Check MDS uncertainty distributions throughout various steps of
classical MDS [1]_.

References
----------
.. [1] Dokmanic, R. Parhizkar, J. Ranieri, and M. Vetterli,
       “Euclidean Distance Matrices: Essential theory, algorithms, and
       applications,”IEEE Signal Processing Magazine, vol. 32, no. 6,
       pp. 12–30, nov 2015.

"""

__authors__ = "D. Knowles"
__date__ = "15 Nov 2021"

import torch
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
from matplotlib.widgets import Slider, Button

def main():

    k = 10000
    """int : iterations of sampling to perform."""

    verbose = False
    """bool : print lots of debug statements."""

    s = Simulator(k, verbose)

    s.model_truth()
    s.model_bias()
    s.model_white_noise()
    s.measure()
    s.compute()
    s.check_distributions()

    s.beauty_plots()

    plt.show()


class Simulator():

    def __init__(self, k, verbose):

        self.k = k
        """int : iterations of sampling to perform"""

        self.verbose = verbose
        """bool : print lots of debug statements."""

        # easy to see multi-modal stuff
        # self.robot_positions = np.array([[0.5, 0., -0.5, 10.0],
        #                                  [2., -1.5,  1., 0.]]).T

        self.robot_positions = np.array([[1., 0., -2., 0.0],
                                         [2., -1.5,  1., 0.]]).T

        """np.ndarray : Node positions as a n x 2 np.ndarray where n is
        number of nodes in the network [m]."""

        self.n = self.robot_positions.shape[0]
        """int : Number of nodes in network."""

        self.dims = self.robot_positions.shape[1]
        """int : Dimension of state space."""

        self.sensor_std = 0.5
        """float : sensor noise standard deviation."""

        self.sensor_bias = 0.0
        """float : sensor bias added to measurements."""

    def model_truth(self):
        ranges_true = dist.pdist(self.robot_positions,
                                 metric = "euclidean")
        self.ranges_true = np.tile(ranges_true.reshape(-1,1), (1, self.k))
        """np.ndarray : truth value of ranges between all points.
        Full array shape is (self.r x self.k).
        """
        if self.verbose:
            print(ranges_true)

        self.r = self.ranges_true.shape[0]
        """int: number of ranges"""

    def model_bias(self):
        self.bias = np.random.uniform(low=0.0,
                    high = self.sensor_bias,
                    size = (self.r, 1))

    def model_white_noise(self):
        self.white_noise = np.random.normal(loc = 0.0,
                           scale = self.sensor_std,
                           size = (self.r, self.k))

    def measure(self):
        """Create a new random measurement.

        """
        self.measured_distribution = [[] for ii in range(self.r)]
        self.sqrd_distribution = [[] for ii in range(self.r)]

        self.noise = self.white_noise + self.bias
        """np.ndarray : noise added to ranges"""

        self.ranges_measured = self.ranges_true + self.noise
        """np.ndarray : measured ranges with added sensor noise."""

        if self.verbose:
            print(self.ranges_measured.shape)
            print(self.ranges_measured)


    def compute(self):
        """Compute classical MDS.

        """
        self.ranges_sqrd = self.ranges_measured**2
        if self.verbose:
            print(self.ranges_sqrd)

        self.D = np.zeros((self.k, self.n, self.n))
        cc = 0
        for ii in range(self.n-1):
            for jj in range(self.n-ii-1):
                self.D[:,jj+1+ii,ii] = self.ranges_sqrd[cc]*np.ones(self.k)
                cc += 1
        self.D += np.transpose(self.D, (0,2,1))
        if self.verbose:
            print("D check\n")
            print(self.D.shape)
            for pp in range(self.D.shape[0]):
                print(self.D[pp,:,:])


        J = np.eye(self.n) - (1./self.n)*np.ones((self.n,self.n))
        if self.verbose:
            print("J early check",J)
        J = np.expand_dims(J, 0)
        J = np.tile(J,(self.k,1,1))
        if self.verbose:
            print("J check")
            print(J.shape)
            for pp in range(J.shape[0]):
                print(J[pp,:,:])
        self.G = -0.5*np.matmul(np.matmul(J,self.D),J)

        G = torch.from_numpy(self.G)
        if self.verbose:
            print("G check:\n",G.shape)
            for pp in range(G.shape[0]):
                print(G[pp,:,:])

        U, S, Vh = torch.linalg.svd(G, full_matrices=True)
        if self.verbose:
            print("S SVD check:")
            print(S.shape)
            print(S)
        S_full = torch.diag_embed(torch.sqrt(S), dim1=1, dim2=2)
        S_full = S_full[:,:self.dims,:]
        if self.verbose:
            print("S full SVD check:")
            print(S_full.shape)
            for pp in range(S_full.shape[0]):
                print(S_full[pp,:,:])
            print("U check")
            print(U.shape)
            print(U)

        self.X = torch.matmul(S_full,torch.transpose(U,1,2))
        if self.verbose:
            print("X check")
            print(self.X.shape)
            for pp in range(self.X.shape[0]):
                print(self.X[pp,:,:])

        Y = np.expand_dims(self.robot_positions.T, 0)
        Y = torch.from_numpy(np.tile(Y,(self.k,1,1)))
        self.X = self.align(self.X,
                            self.X.clone(),
                            Y)

        I4 = np.zeros((self.dims,self.dims))
        for rr in range(self.n - 1):
            phi = np.arctan2(self.robot_positions[-1,0] \
                         - self.robot_positions[rr,0],
                           self.robot_positions[-1,1] \
                         - self.robot_positions[rr,1])
            I4 += np.array([[np.sin(phi)**2, np.sin(2*phi)/2.],
                            [np.sin(2*phi)/2., np.cos(phi)**2]])

        I4 *= (1./self.sensor_std**2)

        # compute Cramer Rao Lower Bound and plot
        self.CRLB4 = np.linalg.inv(I4)
        p = 0.95
        s = -2 * np.log(1. - p)
        w, v = np.linalg.eig(s * self.CRLB4)
        w = np.diag(w)
        t = np.linspace(0, 2*np.pi, 100)
        self.elipse = v.dot(np.sqrt(w)).dot(np.array([np.cos(t),np.sin(t)]))


    def align(self, X, Xa, Y):
        """Algins relative postions to absolute positions.

        Method also known as Procrustes analysis and taken from [1]_.

        Parameters
        ----------
        X : np.ndarray
            Positions of nodes in graph with shape (k x dims x n).
        Xa : np.ndarray
            Subset of X for which positions are known (k x dims x <=n).
        Y : np.ndarray
            Known positions of of Xa nodes of shape (k x dims x <=n).

        Returns
        -------
        X_aligned : np.ndarray
            Aligned version of X of shape (dims x n).

        References
        ----------
        .. [1] Dokmanic, R. Parhizkar, J. Ranieri, and M. Vetterli,
               “Euclidean Distance Matrices: Essential theory, algorithms, and
               applications,”IEEE Signal Processing Magazine, vol. 32, no. 6,
               pp. 12–30, nov 2015.

        """
        # calculate centroids
        Y_c = torch.mean(Y, dim = 2, keepdim = True)
        Y_bar = Y - Y_c
        Xa_c = torch.mean(Xa, dim = 2, keepdim = True)
        Xa_bar = Xa - Xa_c

        # calculate rotation
        U, S, Vh = torch.linalg.svd(torch.matmul(Xa_bar,
                                    torch.transpose(Y_bar,1,2)))

        R = torch.matmul(torch.transpose(Vh,1,2), torch.transpose(U,1,2))


        # translation and rotation
        # todo this currently assumes that X is the same as Xa
        X_aligned = torch.matmul(R,X - Xa_c) + Y_c

        return X_aligned

    def check_distributions(self):
        """Check the distribution of values.

        Looks at 1) distribution of measured values 2) distribution of
        squared values

        """
        measured_distribution = self.ranges_measured.tolist()
        sqrd_distribution = self.ranges_sqrd.tolist()

        for ii in range(self.r):
            self.measured_distribution[ii].append(measured_distribution[ii])
            self.sqrd_distribution[ii].append(sqrd_distribution[ii])

    def beauty_plots(self):
        """Plot some beautiful plots.

        """

        self.fig = plt.figure(num="Multi-dimensional Scaling " \
                                 + "Uncertainty Interactive Plotter",
                             figsize=(12,5))
        subfigs = self.fig.subfigures(nrows = 2,
                                 ncols = 2,
                                 height_ratios = [3, 1])

        # ENDING POSITIONS MAP
        subfigs[0,0].suptitle("Position Map")
        self.ax_map = subfigs[0,0].subplots(1, 1)

        self.map_scatters = []
        for ii in range(self.n):
            new_map = self.ax_map.scatter(self.X[:,0,ii],
                                  self.X[:,1,ii],
                                  c="C"+str(ii+1),
                                  s=0.5,
                                  label="robot "+str(ii+1))
            self.final_pos = self.ax_map.scatter(self.robot_positions[ii,0],
                                self.robot_positions[ii,1],
                                marker = "*",
                                c="C"+str(ii+1),
                                s = 50.0,
                                edgecolors="k"
                                )
            self.map_scatters.append(new_map)

        self.crlb_plot = self.ax_map.plot(self.elipse[0,:] \
                                        + self.robot_positions[-1,0],
                                          self.elipse[1,:] \
                                        + self.robot_positions[-1,0],
                                        "k")

        plt.xlim(-22,22)
        plt.ylim(-12,12)
        self.ax_map.set_aspect("equal", adjustable="datalim")
        plt.xlabel("X axis [m]")
        plt.ylabel("Y axis [m]")
        plt.legend()

        # ENDING X & Y POSITION DISTRIBUTIONS
        subfigs[0,1].suptitle("Robot X & Y Position Distributions")
        self.xy_distrib_axes = subfigs[0,1].subplots(self.n, 2)
        self.x_hists = []
        self.y_hists = []
        for ii in range(self.n):

            hist, bin_edges = np.histogram(self.X[:, 0, ii].numpy(),
                                            bins = 50, density = True)
            self.xy_distrib_axes[ii,0].hist(self.X[:, 0, ii].numpy(),
                     bins = 50, density = True, color="C"+str(ii+1))
            # axes[ii,0].set_title("Robot " + str(ii+1))

            hist, bin_edges = np.histogram(self.X[:, 1, ii].numpy(),
                                            bins = 50, density = True)
            self.xy_distrib_axes[ii,1].hist(self.X[:, 1, ii].numpy(),
                     bins = 50, density = True, color="C"+str(ii +1))
            # axes[ii,1].set_title("Robot " + str(ii+1))

        # MEASURED RANGES UNCERTAINTY
        subfigs[1,1].suptitle("Initial Ranges Uncertainty Map")
        self.ax_ranges = subfigs[1,1].subplots(1, 1)

        hist, bin_edges = np.histogram(self.noise[0,:],
                                        bins = 50, density = True)
        x_bins = (bin_edges[1:] + bin_edges[:-1])/2.
        xlimmax = max(abs(self.bias[0,0] - bin_edges[0]),
                      abs(bin_edges[-1] - self.bias[0,0]))
        self.ax_ranges.set_xlim(-xlimmax+self.bias[0,0],xlimmax+self.bias[0,0])
        # xl,xr = self.ax_ranges.get_xlim()
        self.ranges_hist = self.ax_ranges.bar(x_bins, hist, width = (xlimmax*2)/50.)
        # print(self.ranges_hist)

        x_axis = np.linspace(-xlimmax+self.bias[0,0], xlimmax+self.bias[0,0], 100)
        self.ranges_normal_plt = self.ax_ranges.plot(x_axis, stats.norm.pdf(x_axis,
                                  self.bias[0,0], self.sensor_std),"r")

        # make all of the slider axes
        num_sliders = 4
        slider_axes = subfigs[1,0].subplots(num_sliders, 4,
                        gridspec_kw={'width_ratios': [1, 4, 0.5, 1.5]})

        # remove all far left axis lines (they're just a filler)
        for ii in range(num_sliders):
            slider_axes[ii,0].axis("off")
            slider_axes[ii,2].axis("off")

        # Standard deviation horizontal slider
        self.std_slider = Slider(
            ax=slider_axes[0,1],
            label="Meas. Std.",
            valmin=0.0,
            valmax=10,
            valinit=self.sensor_std,
            valstep=0.1
        )
        self.std_slider.on_changed(self.update_std)

        # Standard deviation reset button
        std_button = Button(slider_axes[0,3], 'Reset',
                            hovercolor='0.975')
        std_button.on_clicked(self.reset_std_slider)
        # add dummy reference so button isn't garbage collected
        slider_axes[0,3]._button = std_button

        # bias horizontal slider
        self.bias_slider = Slider(
            ax=slider_axes[1,1],
            label="Meas. Bias",
            valmin=0.0,
            valmax=10,
            valinit=self.sensor_bias,
            valstep=0.1
        )
        self.bias_slider.on_changed(self.update_bias)

        # bias reset button
        bias_button = Button(slider_axes[1,3], 'Reset',
                            hovercolor='0.975')
        bias_button.on_clicked(self.reset_bias_slider)
        # add dummy reference so button isn't garbage collected
        slider_axes[1,3]._button = bias_button

        # posx horizontal slider
        self.posx_slider = Slider(
            ax=slider_axes[2,1],
            label="X Position",
            valmin=-15,
            valmax=15,
            valinit=self.robot_positions[-1,0],
            valstep=0.1
        )
        self.posx_slider.on_changed(self.update_posx)

        # posx reset button
        posx_button = Button(slider_axes[2,3], 'Reset',
                            hovercolor='0.975')
        posx_button.on_clicked(self.reset_posx_slider)
        # add dummy reference so button isn't garbage collected
        slider_axes[2,3]._button = posx_button

        # posy horizontal slider
        self.posy_slider = Slider(
            ax=slider_axes[3,1],
            label="Y Position",
            valmin=-8.,
            valmax=8.,
            valinit=self.robot_positions[-1,1],
            valstep=0.1
        )
        self.posy_slider.on_changed(self.update_posy)

        # posy reset button
        posy_button = Button(slider_axes[3,3], 'Reset',
                            hovercolor='0.975')
        posy_button.on_clicked(self.reset_posy_slider)
        # add dummy reference so button isn't garbage collected
        slider_axes[3,3]._button = posy_button

        plt.subplots_adjust(hspace=0.5, right=0.95, bottom = 0.18)


    def update_plots(self):

        self.measure()
        self.compute()
        self.check_distributions()

        # update map
        for ii in range(self.n):
            new_map_data = np.zeros((self.k,2))
            new_map_data[:,0] = self.X[:,0,ii]
            new_map_data[:,1] = self.X[:,1,ii]
            self.map_scatters[ii].set_offsets(new_map_data)

            hist, bin_edges = np.histogram(self.X[:, 0, ii].numpy(),
                                            bins = 50, density = True)
            self.xy_distrib_axes[ii,0].clear()
            self.xy_distrib_axes[ii,0].hist(self.X[:, 0, ii].numpy(),
                     bins = 50, density = True, color="C"+str(ii+1))
            self.xy_distrib_axes[ii,1].clear()
            hist, bin_edges = np.histogram(self.X[:, 1, ii].numpy(),
                                            bins = 50, density = True)
            self.xy_distrib_axes[ii,1].hist(self.X[:, 1, ii].numpy(),
                     bins = 50, density = True, color="C"+str(ii +1))

        self.crlb_plot[0].set_xdata(self.elipse[0,:] \
                                  + self.robot_positions[-1,0])
        self.crlb_plot[0].set_ydata(self.elipse[1,:] \
                                  + self.robot_positions[-1,1])

        self.final_pos.set_offsets(self.robot_positions[-1,:])

        hist, bin_edges = np.histogram(self.noise[0,:],
                                        bins = 50, density = True)

        [bar.set_height(hist[i]) for i, bar in enumerate(self.ranges_hist)]
        [bar.set_x(bin_edges[i]) for i, bar in enumerate(self.ranges_hist)]
        width = (bin_edges[-1]-bin_edges[0])/50.
        [bar.set_width(width) for i, bar in enumerate(self.ranges_hist)]

        x_axis = np.linspace(bin_edges[0], bin_edges[-1], 100)
        self.ranges_normal_plt[0].set_xdata(x_axis)
        self.ranges_normal_plt[0].set_ydata(stats.norm.pdf(x_axis,
                                         self.bias[0,0], self.sensor_std))
        self.ax_ranges.relim()
        xlimmax = max(abs(self.bias[0,0] - bin_edges[0]),
                      abs(bin_edges[-1] - self.bias[0,0]))
        self.ax_ranges.set_xlim(-xlimmax+self.bias[0,0],xlimmax+self.bias[0,0])
        self.ax_ranges.autoscale_view()

        self.fig.canvas.draw_idle()

    def update_std(self, val):
        self.sensor_std = self.std_slider.val
        self.model_white_noise()
        self.update_plots()

    def update_bias(self, val):
        self.sensor_bias = self.bias_slider.val
        self.model_bias()
        self.update_plots()

    def update_posx(self, val):
        self.robot_positions[-1,0] = self.posx_slider.val
        self.model_truth()
        self.update_plots()

    def update_posy(self, val):
        self.robot_positions[-1,1] = self.posy_slider.val
        self.model_truth()
        self.update_plots()

    def reset_std_slider(self, event):
        self.std_slider.reset()

    def reset_bias_slider(self, event):
        self.bias_slider.reset()

    def reset_posx_slider(self, event):
        self.posx_slider.reset()

    def reset_posy_slider(self, event):
        self.posy_slider.reset()

if __name__ == "__main__":
    main()
