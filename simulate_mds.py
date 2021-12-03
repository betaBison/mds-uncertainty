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

    k = 1000
    """int : iterations of sampling to perform."""

    verbose = False
    """bool : print lots of debug statements."""

    s = Simulator(k, verbose)
        # if kk % np.ceil(int(k/10)) == 0:
        #     print(round(100.*(kk+1)/k,0),"% complete")
    s.measure()
    s.compute()
    s.check_distributions()

    # s.plot_distrubtions()
    s.beauty_plots()

    plt.show()


class Simulator():

    def __init__(self, k, verbose):

        self.k = k
        """int : iterations of sampling to perform"""

        self.verbose = False
        """bool : print lots of debug statements."""

        self.robot_positions = np.array([[-1., 0., -1., 10.0],
                                         [-2., 0.,  2., 0.]]).T
        # self.robot_positions = np.array([[0., 0., 0, 0.0],
        #                                  [-2., 0.,  2., 10.]]).T
        """np.ndarray : Node positions as a n x 2 np.ndarray where n is
        number of nodes in the network [m]."""

        self.n = self.robot_positions.shape[0]
        """int : Number of nodes in network."""

        self.dims = self.robot_positions.shape[1]
        """int : Dimension of state space."""

        self.sensor_std = 0.1
        """float : sensor noise standard deviation."""

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

        self.measured_distribution = [[] for ii in range(self.r)]
        self.sqrd_distribution = [[] for ii in range(self.r)]

    def measure(self):
        """Create a new random measurement.

        """
        self.noise = np.random.normal(loc = 0.0,
                           scale = self.sensor_std,
                           size = (self.r, self.k))
                           # np.random.uniform(low = -0.5,
                           #                     high = 0.5,
                           #                     size = self.r)
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

        # # U, S, Vh = torch.linalg.svd(G, full_matrices = True)
        # L, Q = torch.linalg.eigh(G)
        # L = torch.flip(L,[1])
        # if self.verbose:
        #     print("L check")
        #     print(L.shape)
        #     print(L)
        #
        #     print("Q check:")
        #     print(Q.shape)
        #     print("Q:\n",Q)
        # S = torch.zeros(L.shape, dtype=torch.float64)
        # if self.verbose:
        #     print("S1 check:",S.shape)
        # S[:,:self.dims] = torch.sqrt(L[:,:self.dims])
        #
        # S_full = torch.diag_embed(S, dim1=1, dim2=2)
        # S_full = S_full[:,:self.dims,:]
        # print("S full check:")
        # print(S_full.shape)
        # for pp in range(S_full.shape[0]):
        #     print(S_full[pp,:,:])

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

        # U, S, V = np.linalg.svd(self.G)
        # S = np.diag(S)[:self.dims,:]
        self.X = torch.matmul(S_full,torch.transpose(U,1,2))
        # self.X = torch.matmul(S_full,Q)
        if self.verbose:
            print("X check")
            print(self.X.shape)
            for pp in range(self.X.shape[0]):
                print(self.X[pp,:,:])

        # self.X = self.align(self.X.copy(),
        #                     self.X.copy(),
        #                     self.robot_positions.copy().T)

    def align(self, X, Xa, Y):
        """Algins relative postions to absolute positions.

        Method also known as Procrustes analysis and taken from [1]_.

        Parameters
        ----------
        X : np.ndarray
            Positions of nodes in graph with shape (dims x n).
        Xa : np.ndarray
            Subset of X for which positions are known (dims x <=n).
        Y : np.ndarray
            Known positions of of Xa nodes of shape (dims x <=n).

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

        # find centroids
        xa_c = Xa.dot(np.ones((Xa.shape[1],1)))/Xa.shape[1]
        y_c = Y.dot(np.ones((Y.shape[1],1)))/Y.shape[1]
        Xa_bar = Xa - np.tile(xa_c,(1,Xa.shape[1]))
        Y_bar = Y - np.tile(y_c,(1,Y.shape[1]))

        # calculate rotation
        U, S, Vh = np.linalg.svd(Xa_bar.dot(Y_bar.T))


        V = Vh.T
        R = V.dot(U.T)

        # translation and rotation
        row_1 = np.ones((1,X.shape[1]))
        X_aligned = R.dot(X - xa_c.dot(row_1)) + y_c.dot(row_1)

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

    def plot_distrubtions(self):
        """Plot the distribution of a provided list.

        Parameters
        ----------
        l : list
            List of values to plot the histogram for.

        """

        # MEASURED RANGES UNCERTAINTY
        fig = plt.figure(figsize=(3,9))
        for ii in range(self.r):
            plt.subplot(self.r, 1, ii+1)

            hist, bin_edges = np.histogram(self.measured_distribution[ii],
                                            bins = 20, density = True)
            plt.hist(self.measured_distribution[ii],
                     bins = 20, density = True)

            x_axis = np.linspace(bin_edges[0], bin_edges[-1], 100)
            plt.plot(x_axis, stats.norm.pdf(x_axis,
                     self.ranges_true[ii], self.sensor_std),"r")
        plt.suptitle("Uncertainty \n distribution \n of measured ranges")
        fig.tight_layout()

        # SQUARED RANGES UNCERTAINTY
        fig = plt.figure(figsize=(3,9))
        for ii in range(self.r):
            plt.subplot(self.r, 1, ii+1)

            hist, bin_edges = np.histogram(self.sqrd_distribution[ii],
                                            bins = 20, density = True)
            plt.hist(self.sqrd_distribution[ii],
                     bins = 20, density = True)

            x_axis = np.linspace(bin_edges[0], bin_edges[-1], 100)
            plt.plot(x_axis, stats.ncx2.pdf(x_axis, df=1.,
                     nc=(self.ranges_true[ii]/self.sensor_std)**2,
                     scale = (self.sensor_std**2)),"r")

        plt.suptitle("Uncertainty \n distribution \n of squared ranges")
        fig.tight_layout()

        # ENDING X POSITION DISTRIBUTIONS
        fig = plt.figure(figsize=(3,6))
        for ii in range(self.n):
            plt.subplot(self.n, 1, ii+1)

            hist, bin_edges = np.histogram(self.pos_distribution[0,ii,:],
                                            bins = 20, density = True)
            plt.hist(self.pos_distribution[0,ii,:],
                     bins = 20, density = True, color="C"+str(ii+1))
            plt.title("Robot " + str(ii+1))

        plt.suptitle("Uncertainty \n distribution \n of X position")
        fig.tight_layout()

        # ENDING Y POSITION DISTRIBUTIONS
        fig = plt.figure(figsize=(3,6))
        for ii in range(self.n):
            plt.subplot(self.n, 1, ii+1)

            hist, bin_edges = np.histogram(self.pos_distribution[1,ii,:],
                                            bins = 20, density = True)
            plt.hist(self.pos_distribution[1,ii,:],
                     bins = 20, density = True, color="C"+str(ii+1))
            plt.title("Robot " + str(ii+1))

        plt.suptitle("Uncertainty \n distribution \n of Y position")
        fig.tight_layout()


        # ENDING POSITIONS MAP
        fig = plt.figure()
        for ii in range(self.n):
            plt.scatter(self.X[:,0,ii],
                        self.X[:,1,ii],
                        c="C"+str(ii+1),
                        label="robot "+str(ii+1))
        plt.axis("equal")
        plt.legend()
        fig.tight_layout()

    def beauty_plots(self):
        """Plot some beautiful plots.

        """

        self.fig = plt.figure(figsize=(12,5))
        subfigs = self.fig.subfigures(nrows = 2,
                                 ncols = 2,
                                 height_ratios = [3, 1])

        # ENDING POSITIONS MAP
        subfigs[0,0].suptitle("Position Map")
        ax_map = subfigs[0,0].subplots(1, 1)

        self.map_scatters = []
        for ii in range(self.n):
            new_map = plt.scatter(self.X[:,0,ii],
                                  self.X[:,1,ii],
                                  c="C"+str(ii+1),
                                  label="robot "+str(ii+1))
            self.map_scatters.append(new_map)
        ax_map.set_aspect("equal", adjustable="datalim")
        plt.xlabel("X axis [m]")
        plt.ylabel("Y axis [m]")
        plt.legend()

        # ENDING X & Y POSITION DISTRIBUTIONS
        subfigs[0,1].suptitle("Robot X & Y Position Distributions")
        axes = subfigs[0,1].subplots(self.n, 2)
        self.x_hists = []
        self.y_hists = []
        for ii in range(self.n):

            hist, bin_edges = np.histogram(self.X[:, 0, ii].numpy(),
                                            bins = 50, density = True)
            axes[ii,0].hist(self.X[:, 0, ii].numpy(),
                     bins = 50, density = True, color="C"+str(ii+1))
            # axes[ii,0].set_title("Robot " + str(ii+1))

            hist, bin_edges = np.histogram(self.X[:, 1, ii].numpy(),
                                            bins = 50, density = True)
            axes[ii,1].hist(self.X[:, 1, ii].numpy(),
                     bins = 50, density = True, color="C"+str(ii +1))
            # axes[ii,1].set_title("Robot " + str(ii+1))

        # MEASURED RANGES UNCERTAINTY
        subfigs[1,1].suptitle("Initial Ranges Uncertainty Map")
        self.ax_ranges = subfigs[1,1].subplots(1, 1)

        hist, bin_edges = np.histogram(self.noise.reshape(-1,1),
                                        bins = 50, density = True)
        x_bins = (bin_edges[1:] + bin_edges[:-1])/2.
        xlimmax = max(abs(bin_edges[0]),abs(bin_edges[-1]))
        self.ax_ranges.set_xlim(-xlimmax,xlimmax)
        xl,xr = self.ax_ranges.get_xlim()
        self.ranges_hist = self.ax_ranges.bar(x_bins, hist, width = (xr-xl)/50.)
        # print(self.ranges_hist)

        x_axis = np.linspace(bin_edges[0], bin_edges[-1], 100)
        self.ranges_normal_plt = self.ax_ranges.plot(x_axis, stats.norm.pdf(x_axis,
                                  0, self.sensor_std),"r")

        # make all of the slider axes
        num_sliders = 4
        slider_axes = subfigs[1,0].subplots(num_sliders, 3,
                        gridspec_kw={'width_ratios': [1, 4, 2]})

        # remove all far left axis lines (they're just a filler)
        for ii in range(num_sliders):
            slider_axes[ii,0].axis("off")
            # .set_visible(False)
            # slider_axes[ii,0].yaxis.set_visible(False)

        # Standard deviation horizontal slider
        self.std_slider = Slider(
            ax=slider_axes[0,1],
            label="Meas. Std.",
            valmin=0.0,
            valmax=10,
            valinit=0.1,
            valstep=0.01
        )
        self.std_slider.on_changed(self.update_plots)

        # Standard deviation reset button
        std_button = Button(slider_axes[0,2], 'Reset',
                            hovercolor='0.975')
        std_button.on_clicked(self.reset_std_slider)
        # add dummy reference so button isn't garbage collected
        slider_axes[0,2]._button = std_button

        plt.subplots_adjust(hspace=0.5, right=0.95, bottom = 0.18)


    def update_plots(self, val):
        self.sensor_std = self.std_slider.val

        self.measure()
        self.compute()
        self.check_distributions()

        # update map
        for ii in range(self.n):
            new_map_data = np.zeros((self.k,2))
            new_map_data[:,0] = self.X[:,0,ii]
            new_map_data[:,1] = self.X[:,1,ii]
            self.map_scatters[ii].set_offsets(new_map_data)


        hist, bin_edges = np.histogram(self.noise.reshape(-1,1),
                                        bins = 50, density = True)

        [bar.set_height(hist[i]) for i, bar in enumerate(self.ranges_hist)]
        [bar.set_x(bin_edges[i]) for i, bar in enumerate(self.ranges_hist)]
        width = (bin_edges[-1]-bin_edges[0])/50.
        [bar.set_width(width) for i, bar in enumerate(self.ranges_hist)]

        x_axis = np.linspace(bin_edges[0], bin_edges[-1], 100)
        self.ranges_normal_plt[0].set_xdata(x_axis)
        self.ranges_normal_plt[0].set_ydata(stats.norm.pdf(x_axis,
                                         0, self.sensor_std))
        self.ax_ranges.relim()
        xlimmax = max(abs(bin_edges[0]),abs(bin_edges[-1]))
        self.ax_ranges.set_xlim(-xlimmax,xlimmax)
        self.ax_ranges.autoscale_view()

        self.fig.canvas.draw_idle()

    def reset_std_slider(self, event):
        self.std_slider.reset()




if __name__ == "__main__":
    main()
