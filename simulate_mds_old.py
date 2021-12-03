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


def main():

    k = 1
    """int : iterations of sampling to perform"""

    s = Simulator(k)
        # if kk % np.ceil(int(k/10)) == 0:
        #     print(round(100.*(kk+1)/k,0),"% complete")
    s.measure()
    s.compute()
    s.check_distributions()

    # s.plot_distrubtions()
    s.beauty_plots()

    plt.show()


class Simulator():

    def __init__(self, k):

        self.k = k
        """int : iterations of sampling to perform"""

        # self.robot_positions = np.array([[-1., 0., -1., 0.0],
        #                                  [-2., 0.,  2., 10.]]).T
        self.robot_positions = np.array([[-1., 0., -1., 10.0],
                                         [-2., 0.,  2., 0.]]).T
        """np.ndarray : Node positions as a n x 2 np.ndarray where n is
        number of nodes in the network [m]."""

        self.n = self.robot_positions.shape[0]
        """int : Number of nodes in network."""

        self.dims = self.robot_positions.shape[1]
        """int : Dimension of state space."""

        self.sensor_std = 0.0
        """float : sensor noise standard deviation."""

        ranges_true = dist.pdist(self.robot_positions,
                                 metric = "euclidean")
        self.ranges_true = np.tile(ranges_true.reshape(-1,1), (1, self.k))
        """np.ndarray : truth value of ranges between all points.
        Full array shape is (self.r x self.k).
        """
        print(ranges_true)

        self.r = self.ranges_true.shape[0]
        """int: number of ranges"""

        self.measured_distribution = [[] for ii in range(self.r)]
        self.sqrd_distribution = [[] for ii in range(self.r)]
        self.pos_distribution = []

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

        print(self.ranges_measured.shape)
        print(self.ranges_measured)


    def compute(self):
        """Compute classical MDS.

        """
        self.ranges_sqrd = self.ranges_measured**2
        print(self.ranges_sqrd)

        self.D = np.zeros((self.k, self.n, self.n))
        cc = 0
        for ii in range(self.n-1):
            for jj in range(self.n-ii-1):
                self.D[:,jj+1+ii,ii] = self.ranges_sqrd[cc]*np.ones(self.k)
                cc += 1
        self.D += np.transpose(self.D, (0,2,1))
        self.D = self.D[0,:,:]
        print("D check\n")
        print(self.D)
        print(self.D.shape)

        J = np.eye(self.n) - (1./self.n)*np.ones((self.n,self.n))
        print("J check")
        print(J.shape)
        print(J)

        self.G = -0.5*J.dot(self.D).dot(J)

        print("G check:\n",self.G.shape)
        print(self.G)

        w, v = np.linalg.eig(self.G)
        print("W shape")
        print(w)
        print(np.linalg.eigvals(self.G))
        print("V shape")


        U, S, V = np.linalg.svd(self.G)
        print("S check")
        print(S)
        S = np.diag(S)[:self.dims,:]
        self.X = np.sqrt(S).dot(U.T)
        print("X check")
        print(self.X.shape)
        print(self.X)

        self.X = self.align(self.X.copy(),
                            self.X.copy(),
                            self.robot_positions.copy().T)

    def align(self, X, Xa, Y):
        """Algins relative postions to absolute positions.

        Method also known as Procrustes analysis and taken from [1]_.

        Parameters# ENDING X & Y POSITION DISTRIBUTIONS
        subfigs[0,1].suptitle("Robot X & Y Position Distributions")
        axes = subfigs[0,1].subplots(self.n, 2)
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
        print("xa_c:\n",xa_c)
        print("y_c:\n",y_c)
        Xa_bar = Xa - np.tile(xa_c,(1,Xa.shape[1]))
        Y_bar = Y - np.tile(y_c,(1,Y.shape[1]))
        print("Xa_bar:\n",Xa_bar)
        print("Y_bar:\n",Y_bar)

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
        if len(self.pos_distribution) == 0:
            self.pos_distribution = np.expand_dims(self.X,2)
        else:
            self.pos_distribution = np.concatenate((self.pos_distribution,
            np.expand_dims(self.X,2)),axis=2)

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
            plt.scatter(self.pos_distribution[0,ii,:],
                        self.pos_distribution[1,ii,:],
                        c="C"+str(ii+1),
                        label="robot "+str(ii+1))
        plt.axis("equal")
        plt.legend()
        fig.tight_layout()

    def beauty_plots(self):
        """Plot some beautiful plots.

        """

        fig = plt.figure(figsize=(12,5))
        # fig_axes = fig.add_gridspec(3,2)
        # subfigs = np.array([[fig.add_subplot(fig_axes[0:2, 0]),
        #             fig.add_subplot(fig_axes[0:2, 1])],
        #            [fig.add_subplot(fig_axes[2, 0]),
        #             fig.add_subplot(fig_axes[2, 1])]])

        subfigs = fig.subfigures(nrows = 2,
                                 ncols = 2,
                                 height_ratios = [3, 1])

        # for outerind, subfig in enumerate(subfigs.flat):
        #
        #     print(outerind, subfig)
        #     subfig.suptitle(f'Subfig {outerind}')
        #     axs = subfig.subplots(2, 1)
        #     for innerind, ax in enumerate(axs.flat):
        #         ax.set_title(f'outer={outerind}, inner={innerind}', fontsize='small')
        #         ax.set_xticks([])
        #         ax.set_yticks([])

        # ENDING POSITIONS MAP
        subfigs[0,0].suptitle("Position Map")
        ax_map = subfigs[0,0].subplots(1, 1)

        for ii in range(self.n):
            plt.scatter(self.pos_distribution[0,ii,:],
                        self.pos_distribution[1,ii,:],
                        c="C"+str(ii+1),
                        label="robot "+str(ii+1))
        ax_map.set_aspect("equal", adjustable="datalim")
        plt.xlabel("X axis [m]")
        plt.ylabel("Y axis [m]")
        plt.legend()

        # ENDING X & Y POSITION DISTRIBUTIONS
        subfigs[0,1].suptitle("Robot X & Y Position Distributions")
        axes = subfigs[0,1].subplots(self.n, 2)
        for ii in range(self.n):

            hist, bin_edges = np.histogram(self.pos_distribution[0,ii,:],
                                            bins = 20, density = True)
            axes[ii,0].hist(self.pos_distribution[0,ii,:],
                     bins = 20, density = True, color="C"+str(ii+1))
            axes[ii,0].set_title("Robot " + str(ii+1))

            hist, bin_edges = np.histogram(self.pos_distribution[1, ii ,:],
                                            bins = 20, density = True)
            axes[ii,1].hist(self.pos_distribution[1, ii, :],
                     bins = 20, density = True, color="C"+str(ii +1))
            axes[ii,1].set_title("Robot " + str(ii+1))



        # plt.tight_layout(pad=5.0, h_pad = 5.0, w_pad = 10.0)
        # fig.tight_layout(pad=10)
        # fig.tight_layout(w_pad = 5.0)
        # fig.tight_layout(h_pad = 1.0)


        # # ENDING Y POSITION DISTRIBUTIONS
        # fig = plt.figure(figsize=(3,6))
        # for ii in range(self.n):
        #     plt.subplot(self.n, 1, ii+1)
        #
        #     hist, bin_edges = np.histogram(self.pos_distribution[1,ii,:],
        #                                     bins = 20, density = True)
        #     plt.hist(self.pos_distribution[1,ii,:],
        #              bins = 20, density = True, color="C"+str(ii+1))
        #     plt.title("Robot " + str(ii+1))
        #
        # plt.suptitle("Uncertainty \n distribution \n of Y position")






if __name__ == "__main__":
    main()
