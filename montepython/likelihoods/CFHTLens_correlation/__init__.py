from montepython.likelihood_class import Likelihood
import io_mp

import scipy.integrate
from scipy import interpolate as itp
from scipy import special
import os
import numpy as np
import math
# Adapted from Julien Lesgourgues's likelihood euclid_lensing
# and from Adam Moss's cosmomc likelihood for weak lensing


class CFHTLens_correlation(Likelihood):

    def __init__(self, path, data, command_line):

        Likelihood.__init__(self, path, data, command_line)

        # Force the cosmological module to store Pk for redshifts up to
        # max(self.z)
        self.need_cosmo_arguments(data, {'output': 'mPk'})
        self.need_cosmo_arguments(data, {'z_max_pk': self.zmax})
        # Force the cosmological module to store Pk for k up to an arbitrary
        # number
        self.need_cosmo_arguments(data, {'P_k_max_1/Mpc': self.k_max})

        # Define array of l values, and initialize them
        # It is a logspace
        self.l = np.exp(self.dlnl*np.arange(self.nlmax))

        # Read dn_dz from window files
        self.z_p = np.zeros(self.nzmax)
        zptemp = np.zeros(self.nzmax)
        self.p = np.zeros((self.nzmax, self.nbin))
        for i in range(self.nbin):
            window_file_path = os.path.join(
                self.data_directory, self.window_file[i])
            if os.path.exists(window_file_path):
                zptemp = np.loadtxt(window_file_path, usecols=[0])
                if (i > 0 and np.sum((zptemp-self.z_p)**2) > 1e-6):
                    raise io_mp.LikelihoodError(
                        "The redshift values for the window files "
                        "at different bins do not match")
                self.z_p = zptemp
                self.p[:, i] = np.loadtxt(window_file_path, usecols=[1])

        # Read measurements of xi+ and xi-
        nt = (self.nbin)*(self.nbin+1)/2
        self.theta_bins = np.zeros(2*self.ntheta)
        self.xi_obs = np.zeros(self.ntheta*nt*2)
        xipm_file_path = os.path.join(
            self.data_directory, self.xipm_file)
        if os.path.exists(xipm_file_path):
            self.theta_bins = np.loadtxt(xipm_file_path)[:, 0]
            if (np.sum(
                (self.theta_bins[:self.ntheta] -
                    self.theta_bins[self.ntheta:])**2) > 1e-6):
                raise io_mp.LikelihoodError(
                    "The angular values at which xi+ and xi- "
                    "are observed do not match")
            temp = np.loadtxt(xipm_file_path)[:, 1:]
        k = 0
        for j in range(nt):
            for i in range(2*self.ntheta):
                self.xi_obs[k] = temp[i, j]
                k = k + 1

        # Read covariance matrix
        ndim = (self.ntheta)*(self.nbin)*(self.nbin+1)
        covmat = np.zeros((ndim, ndim))
        covmat_file_path = os.path.join(self.data_directory, self.covmat_file)
        if os.path.exists(covmat_file_path):
            covmat = np.loadtxt(covmat_file_path)
        covmat = covmat/self.ah_factor

        # Read angular cut values (OPTIONAL)
        if(self.use_cut_theta):
            cut_values = np.zeros((self.nbin, 2))
            cutvalues_file_path = os.path.join(
                self.data_directory, self.cutvalues_file)
            if os.path.exists(cutvalues_file_path):
                cut_values = np.loadtxt(cutvalues_file_path)

        # Normalize selection functions
        self.p_norm = np.zeros(self.nbin, 'float64')
        for Bin in xrange(self.nbin):
            self.p_norm[Bin] = np.sum(0.5*(
                self.p[1:, Bin]+self.p[:-1, Bin])*(
                self.z_p[1:]-self.z_p[:-1]))

        # Compute theta mask
        if (self.use_cut_theta):
            mask = np.zeros(2*nt*self.ntheta)
            iz = 0
            for izl in range(self.nbin):
                for izh in range(izl, self.nbin):
                    # this counts the bin combinations
                    # iz=1 =>(1,1), iz=2 =>(1,2) etc
                    iz = iz + 1
                    for i in range(self.ntheta):
                        j = (iz-1)*2*self.ntheta
                        xi_plus_cut = max(
                            cut_values[izl, 0], cut_values[izh, 0])
                        xi_minus_cut = max(
                            cut_values[izl, 1], cut_values[izh, 1])
                        if (self.theta_bins[i] > xi_plus_cut):
                            mask[j+i] = 1
                        if (self.theta_bins[i] > xi_minus_cut):
                            mask[self.ntheta + j+i] = 1
        else:
            mask = np.ones(2*nt*self.ntheta)

        self.num_mask = np.sum(mask)
        self.mask_indices = np.zeros(self.num_mask)
        j = 0
        for i in range(self.ntheta*nt*2):
            if (mask[i] == 1):
                self.mask_indices[j] = i
                j = j+1
        self.mask_indices = np.int32(self.mask_indices)
        # Precompute masked inverse
        self.wl_invcov = np.zeros((self.num_mask, self.num_mask))
        self.wl_invcov = covmat[self.mask_indices][:, self.mask_indices]
        self.wl_invcov = np.linalg.inv(self.wl_invcov)

        # Fill array of discrete z values
        # self.z = np.linspace(0, self.zmax, num=self.nzmax)

        ################
        # Noise spectrum
        ################

        # Number of galaxies per steradian
        self.noise = 3600.*self.gal_per_sqarcmn*(180./math.pi)**2

        # Number of galaxies per steradian per bin
        self.noise = self.noise/self.nbin

        # Noise spectrum (diagonal in bin*bin space, independent of l and Bin)
        self.noise = self.rms_shear**2/self.noise

        return

    def loglkl(self, cosmo, data):

        # One wants to obtain here the relation between z and r, this is done
        # by asking the cosmological module with the function z_of_r
        self.r = np.zeros(self.nzmax, 'float64')
        self.dzdr = np.zeros(self.nzmax, 'float64')

        self.r, self.dzdr = cosmo.z_of_r(self.z_p)

        # Compute now the selection function p(r) = p(z) dz/dr normalized
        # to one. The np.newaxis helps to broadcast the one-dimensional array
        # dzdr to the proper shape. Note that p_norm is also broadcasted as
        # an array of the same shape as p_z
        self.p_r = self.p*(self.dzdr[:, np.newaxis]/self.p_norm)

        # Compute function g_i(r), that depends on r and the bin
        # g_i(r) = 2r(1+z(r)) int_r^+\infty drs p_r(rs) (rs-r)/rs

        g = np.zeros((self.nzmax, self.nbin), 'float64')
        for Bin in xrange(self.nbin):
            for nr in xrange(1, self.nzmax-1):
                fun = self.p_r[nr:, Bin]*(self.r[nr:]-self.r[nr])/self.r[nr:]
                g[nr, Bin] = np.sum(0.5*(
                    fun[1:]+fun[:-1])*(self.r[nr+1:]-self.r[nr:-1]))
                g[nr, Bin] *= 2.*self.r[nr]*(1.+self.z_p[nr])

        # Get power spectrum P(k=l/r,z(r)) from cosmological module
        pk = np.zeros((self.nlmax, self.nzmax), 'float64')
        for index_l in xrange(self.nlmax):
            for index_z in xrange(1, self.nzmax):
#                if (self.l[index_l]/self.r[index_z] > self.k_max):
#                    raise io_mp.LikelihoodError(
#                        "you should increase CFHTLens_correlation.k_max up"
#                        " to at least %g" % (self.l[index_l]/self.r[index_z]))
#                pk[index_l, index_z] = cosmo.pk(
#                    self.l[index_l]/self.r[index_z], self.z_p[index_z])
                if (self.l[index_l]/self.r[index_z] > self.k_max):
                    pk[index_l, index_z] = 0.0
                else:
                    pk[index_l, index_z] = cosmo.pk(
                        self.l[index_l]/self.r[index_z], self.z_p[index_z])

        # Recover the non_linear scale computed by halofit. If no scale was
        # affected, set the scale to one, and make sure that the nuisance
        # parameter epsilon is set to zero
        k_sigma = np.zeros(self.nzmax, 'float64')
        if (cosmo.nonlinear_method == 0):
            k_sigma[:] = 1.e6
        else:
            k_sigma = cosmo.nonlinear_scale(self.z_p, self.nzmax)

        # Define the alpha function, that will characterize the theoretical
        # uncertainty. Chosen to be 0.001 at low k, raise between 0.1 and 0.2
        # to self.theoretical_error
        alpha = np.zeros((self.nlmax, self.nzmax), 'float64')
        if self.theoretical_error != 0:
            for index_l in range(self.nlmax):
                k = self.l[index_l]/self.r[1:]
                alpha[index_l, 1:] = np.log(1.+k[:]/k_sigma[1:])/(
                    1.+np.log(1.+k[:]/k_sigma[1:]))*self.theoretical_error

        # recover the e_th_nu part of the error function
        e_th_nu = self.coefficient_f_nu*cosmo.Omega_nu/cosmo.Omega_m()

        # Compute the Error E_th_nu function
        if 'epsilon' in self.use_nuisance:
            E_th_nu = np.zeros((self.nlmax, self.nzmax), 'float64')
            for index_l in range(1, self.nlmax):
                E_th_nu[index_l, :] = np.log(
                    1.+self.l[index_l]/k_sigma[:]*self.r[:]) / (
                    1.+np.log(1.+self.l[index_l]/k_sigma[:]*self.r[:]))*e_th_nu

        # Add the error function, with the nuisance parameter, to P_nl_th, if
        # the nuisance parameter exists
                for index_l in range(self.nlmax):
                    epsilon = data.mcmc_parameters['epsilon']['current']*(
                        data.mcmc_parameters['epsilon']['scale'])
                    pk[index_l, :] *= (1.+epsilon*E_th_nu[index_l, :])

        # Start loop over l for computation of C_l^shear
        Cl_integrand = np.zeros((self.nzmax, self.nbin, self.nbin), 'float64')
        Cl = np.zeros((self.nlmax, self.nbin, self.nbin), 'float64')
        # Start loop over l for computation of E_l
        if self.theoretical_error != 0:
            El_integrand = np.zeros((self.nzmax, self.nbin, self.nbin),
                                    'float64')
            El = np.zeros((self.nlmax, self.nbin, self.nbin), 'float64')

        for nl in xrange(self.nlmax):

            # find Cl_integrand = (g(r) / r)**2 * P(l/r,z(r))
            for Bin1 in xrange(self.nbin):
                for Bin2 in xrange(self.nbin):
                    Cl_integrand[1:, Bin1, Bin2] = g[1:, Bin1]*g[1:, Bin2]/(
                        self.r[1:]**2)*pk[nl, 1:]
                    if self.theoretical_error != 0:
                        El_integrand[1:, Bin1, Bin2] = g[1:, Bin1]*(
                            g[1:, Bin2])/(
                            self.r[1:]**2)*pk[nl, 1:]*alpha[nl, 1:]

            # Integrate over r to get C_l^shear_ij = P_ij(l)
            # C_l^shear_ij = 9/16 Omega0_m^2 H_0^4 \sum_0^rmax dr (g_i(r)
            # g_j(r) /r**2) P(k=l/r,z(r)) dr
            # It is then multiplied by 9/16*Omega_m**2
            # and then by (h/2997.9)**4 to be dimensionless
            # (since P(k)*dr is in units of Mpc**4)
            for Bin1 in xrange(self.nbin):
                for Bin2 in xrange(self.nbin):
                    Cl[nl, Bin1, Bin2] = np.sum(0.5*(
                        Cl_integrand[1:, Bin1, Bin2] +
                        Cl_integrand[:-1, Bin1, Bin2])*(
                        self.r[1:]-self.r[:-1]))
                    Cl[nl, Bin1, Bin2] *= 9./16.*(cosmo.Omega_m())**2
                    Cl[nl, Bin1, Bin2] *= (cosmo.h()/2997.9)**4

                    if self.theoretical_error != 0:
                        El[nl, Bin1, Bin2] = np.sum(0.5*(
                            El_integrand[1:, Bin1, Bin2] +
                            El_integrand[:-1, Bin1, Bin2])*(
                            self.r[1:]-self.r[:-1]))
                        El[nl, Bin1, Bin2] *= 9./16.*(cosmo.Omega_m())**2
                        El[nl, Bin1, Bin2] *= (cosmo.h()/2997.9)**4
                    if Bin1 == Bin2:
                        Cl[nl, Bin1, Bin2] += self.noise

        # Spline Cl[nl,Bin1,Bin2] along l
        spline_Cl = np.empty((self.nbin, self.nbin), dtype=(list, 3))
        for Bin1 in xrange(self.nbin):
            for Bin2 in xrange(Bin1, self.nbin):
                spline_Cl[Bin1, Bin2] = list(itp.splrep(
                    self.l, Cl[:, Bin1, Bin2]))
                if Bin2 > Bin1:
                    spline_Cl[Bin2, Bin1] = spline_Cl[Bin1, Bin2]

        # Convert C_l to xi's
        thetamin = np.min(self.theta_bins)*0.8
        thetamax = np.max(self.theta_bins)*1.2
        nthetatot = np.ceil(math.log(thetamax/thetamin)/self.dlntheta) + 1
        nthetatot = np.int32(nthetatot)

        lmin = self.l[0]
        lmax = self.l[self.nlmax-1]

        theta = np.zeros(nthetatot, 'float64')
        xi1 = np.zeros((nthetatot, self.nbin, self.nbin), 'float64')
        xi2 = np.zeros((nthetatot, self.nbin, self.nbin), 'float64')
        i1p = np.zeros((self.nbin, self.nbin), 'float64')
        i2p = np.zeros((self.nbin, self.nbin), 'float64')

        a2r = math.pi/(180.*60.)

        for it in range(nthetatot):
            theta[it] = thetamin*math.exp(self.dlntheta*it)
            xmin = lmin*theta[it]*a2r  # Convert from arcmin to radians
            xmax = lmax*theta[it]*a2r
            x = xmin
            lp = 0
            while (x < self.xstop and x < xmax):
                lll = x/(theta[it]*a2r)
                if(lll > lmax):
                    raise io_mp.LikelihoodError(
                        "ERROR: l>lmax")
                Bessel0 = special.j0(x)
                Bessel4 = special.jv(4, x)
                for ib in range(self.nbin):
                    for jb in range(ib, self.nbin):
                        Cval = lll*itp.splev(lll, spline_Cl[ib, jb])
                        i1 = Cval*Bessel0
                        i2 = Cval*Bessel4
                        xi1[it, ib, jb] += 0.5*(i1p[ib, jb]+i1)*(lll-lp)
                        xi2[it, ib, jb] += 0.5*(i2p[ib, jb]+i2)*(lll-lp)
                        i1p[ib, jb] = i1
                        i2p[ib, jb] = i2
                x = x+self.dx
                lp = lll

            for ib in range(self.nbin):
                for jb in range(ib, self.nbin):
                    xi1[it, jb, ib] = xi1[it, ib, jb]
                    xi2[it, jb, ib] = xi2[it, ib, jb]

        xi1 = xi1/(2.*math.pi)
        xi2 = xi2/(2.*math.pi)

        # Get xi's in column vector format
        xi1_theta = np.empty((self.nbin, self.nbin), dtype=(list, 3))
        for Bin1 in xrange(self.nbin):
            for Bin2 in xrange(Bin1, self.nbin):
                xi1_theta[Bin1, Bin2] = list(itp.splrep(
                    theta, xi1[:, Bin1, Bin2]))
                if Bin2 > Bin1:
                    xi1_theta[Bin2, Bin1] = xi1_theta[Bin1, Bin2]
        xi2_theta = np.empty((self.nbin, self.nbin), dtype=(list, 3))
        for Bin1 in xrange(self.nbin):
            for Bin2 in xrange(Bin1, self.nbin):
                xi2_theta[Bin1, Bin2] = list(itp.splrep(
                    theta, xi2[:, Bin1, Bin2]))
                if Bin2 > Bin1:
                    xi2_theta[Bin2, Bin1] = xi2_theta[Bin1, Bin2]

        iz = 0
        xi = np.zeros(np.size(self.xi_obs), 'float64')
        for izl in range(self.nbin):
            for izh in range(izl, self.nbin):
                iz = iz + 1  # this counts the bin combinations
                for i in range(self.ntheta):
                    j = (iz-1)*2*self.ntheta
                    xi[j+i] = itp.splev(
                        self.theta_bins[i], xi1_theta[izl, izh])
                    xi[self.ntheta + j+i] = itp.splev(
                        self.theta_bins[i], xi2_theta[izl, izh])

        vec = xi[self.mask_indices] - self.xi_obs[self.mask_indices]
        chi2 = np.dot(vec, np.dot(self.wl_invcov, vec))

        return -chi2/2.
