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

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist


def main():

    k = 10000
    """int : iterations of sampling to perform"""

    s = Simulator()
    for kk in range(k):
        if kk % np.ceil(int(k/10)) == 0:
            print(round(100.*(kk+1)/k,0),"% complete")
        s.measure()
        s.compute()
        s.check_distributions()

    s.plot_distrubtions()

    plt.show()


class Simulator():

    def __init__(self):

        self.robot_positions = np.array([[0., 2., 4., 2.],
                                         [0., 0., 0., 4.]]).T
        """np.ndarray : Node positions as a n x 2 np.ndarray where n is
        number of nodes in the network [m]."""

        self.n = self.robot_positions.shape[0]
        """int : Number of nodes in network."""

        self.dims = self.robot_positions.shape[1]
        """int : Dimension of state space."""

        self.sensor_std = 0.05
        """float : sensor noise standard deviation."""

        self.ranges_true = dist.pdist(self.robot_positions,
                                      metric = "euclidean")
        """np.ndarray : truth value of ranges between all points."""
        print(self.ranges_true)

        self.r = len(self.ranges_true)
        """int: number of ranges"""

        self.measured_distribution = [[] for ii in range(self.r)]
        self.sqrd_distribution = [[] for ii in range(self.r)]
        self.pos_distribution = []

    def measure(self):
        """Create a new random measurement.

        """

        self.ranges_measured = self.ranges_true \
                             + np.random.normal(loc = 0.0,
                                                scale = self.sensor_std,
                                                size = self.r)
        """np.ndarray : measured ranges with added sensor noise."""


    def compute(self):
        """Compute classical MDS.

        """
        self.ranges_sqrd = self.ranges_measured**2

        self.D = np.zeros((self.n,self.n))
        cc = 0
        for ii in range(self.n-1):
            for jj in range(self.n-ii-1):
                self.D[jj+1+ii,ii] = self.ranges_sqrd[cc]
                cc += 1
        self.D += self.D.T


        J = np.eye(self.n) - (1./self.n)*np.ones((self.n,self.n))
        self.G = -0.5*J.dot(self.D).dot(J)
        U, S, V = np.linalg.svd(self.G)
        S = np.diag(S)[:self.dims,:]
        self.X = np.sqrt(S).dot(U.T)

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
        plt.figure()
        for ii in range(self.r):
            plt.subplot(2, 3, ii+1)

            hist, bin_edges = np.histogram(self.measured_distribution[ii],
                                            bins = 20, density = True)
            plt.hist(self.measured_distribution[ii],
                     bins = 20, density = True)

            x_axis = np.linspace(bin_edges[0], bin_edges[-1], 100)
            plt.plot(x_axis, stats.norm.pdf(x_axis,
                     self.ranges_true[ii], self.sensor_std),"r")
        plt.suptitle("Uncertainty distribution of measured ranges")

        # SQUARED RANGES UNCERTAINTY
        plt.figure()
        for ii in range(self.r):
            plt.subplot(2, 3, ii+1)

            hist, bin_edges = np.histogram(self.sqrd_distribution[ii],
                                            bins = 20, density = True)
            plt.hist(self.sqrd_distribution[ii],
                     bins = 20, density = True)

            x_axis = np.linspace(bin_edges[0], bin_edges[-1], 100)
            plt.plot(x_axis, stats.ncx2.pdf(x_axis, df=1.,
                     nc=(self.ranges_true[ii]/self.sensor_std)**2,
                     scale = (self.sensor_std**2)),"r")

        plt.suptitle("Uncertainty distribution of squared ranges")


        # ENDING POSITIONS
        plt.figure()
        for ii in range(self.n):
            plt.scatter(self.pos_distribution[0,ii,:],
                        self.pos_distribution[1,ii,:],c="C"+str(ii+1))
        plt.axis("equal")



if __name__ == "__main__":
    main()
