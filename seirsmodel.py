from __future__ import absolute_import, division, print_function

import networkx as networkx
import numpy as numpy
import scipy as scipy
import scipy.integrate

REAL_POPULATION = 18000000


class SEIRSModel():
    """
    A class to simulate the Deterministic SEIRS Model
    ===================================================
    Params: beta    Rate of transmission (exposure) 
            sigma   Rate of infection (upon exposure) 
            gamma   Rate of recovery (upon infection) 
            xi      Rate of re-susceptibility (upon recovery)  
            mu_I    Rate of infection-related death  
            mu_0    Rate of baseline death   
            nu      Rate of baseline birth
            
            beta_D  Rate of transmission (exposure) for individuals with detected infections
            sigma_D Rate of infection (upon exposure) for individuals with detected infections
            gamma_D Rate of recovery (upon infection) for individuals with detected infections
            mu_D    Rate of infection-related death for individuals with detected infections
            theta_E Rate of baseline testing for exposed individuals
            theta_I Rate of baseline testing for infectious individuals
            psi_E   Probability of positive test results for exposed individuals
            psi_I   Probability of positive test results for exposed individuals
            q       Probability of quarantined individuals interacting with others
            
            initE   Init number of exposed individuals       
            initI   Init number of infectious individuals      
            initD_E Init number of detected infectious individuals
            initD_I Init number of detected infectious individuals   
            initR   Init number of recovered individuals     
            initF   Init number of infection-related fatalities
                    (all remaining nodes initialized susceptible)   
    """
    def __init__(self,
                 initN,
                 beta,
                 sigma,
                 gamma,
                 xi=0,
                 mu_I=0,
                 mu_0=0,
                 nu=0,
                 p=0,
                 beta_D=None,
                 sigma_D=None,
                 gamma_D=None,
                 mu_D=None,
                 theta_E=0,
                 theta_I=0,
                 psi_E=0,
                 psi_I=0,
                 q=0,
                 mu_H=0,
                 mu_ICU=0,
                 initE=0,
                 initI=10,
                 initD_E=0,
                 initD_I=0,
                 initR=0,
                 initF=0,
                 initH=0):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Model Parameters:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.xi = xi
        self.mu_I = mu_I
        self.mu_0 = mu_0
        self.nu = nu
        self.p = p
        self.mu_H = mu_H
        self.mu_ICU = mu_ICU

        # Testing-related parameters:
        self.beta_D = beta_D if beta_D is not None else self.beta
        self.sigma_D = sigma_D if sigma_D is not None else self.sigma
        self.gamma_D = gamma_D if gamma_D is not None else self.gamma
        self.mu_D = mu_D if mu_D is not None else self.mu_I
        self.theta_E = theta_E if theta_E is not None else self.theta_E
        self.theta_I = theta_I if theta_I is not None else self.theta_I
        self.psi_E = psi_E if psi_E is not None else self.psi_E
        self.psi_I = psi_I if psi_I is not None else self.psi_I
        self.q = q if q is not None else self.q

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Timekeeping:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.t = 0
        self.tmax = 0  # will be set when run() is called
        self.tseries = numpy.array([0])

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Initialize Counts of inidividuals with each state:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.N = numpy.array([int(initN)])
        self.numE = numpy.array([int(initE)])
        self.numI = numpy.array([int(initI)])
        self.numD_E = numpy.array([int(initD_E)])
        self.numD_I = numpy.array([int(initD_I)])
        self.numR = numpy.array([int(initR)])
        self.numF = numpy.array([int(initF)])
        self.numH = numpy.array([int(initH)])
        self.numS = numpy.array([
            self.N[-1] - self.numE[-1] - self.numI[-1] - self.numD_E[-1] -
            self.numD_I[-1] - self.numR[-1] - self.numF[-1]
        ])
        assert (
            self.numS[0] >= 0
        ), "The specified initial population size N must be greater than or equal to the initial compartment counts."

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    @staticmethod
    def system_dfes(t, variables, beta, sigma, gamma, xi, mu_I, mu_0, nu,
                    beta_D, sigma_D, gamma_D, mu_D, theta_E, theta_I, psi_E,
                    psi_I, q, mu_H):

        S, E, I, D_E, D_I, R, F, H = variables  # varibles is a list with compartment counts as elements

        N = S + E + I + D_E + D_I + R

        dS = -(beta * S * I) / N - q * (beta_D * S *
                                        D_I) / N + xi * R + nu * N - mu_0 * S

        dE = (beta * S * I) / N + q * (
            beta_D * S * D_I) / N - sigma * E - theta_E * psi_E * E - mu_0 * E

        dI = sigma * E - gamma * I - mu_I * I - theta_I * psi_I * I - mu_0 * I

        dDE = theta_E * psi_E * E - sigma_D * D_E - mu_0 * D_E

        dDI = theta_I * psi_I * I + sigma_D * D_E - gamma_D * D_I - mu_D * D_I - mu_0 * D_I

        dR = gamma * I + gamma_D * D_I - xi * R - mu_0 * R

        # dF = mu_I * I + mu_D * D_I

        dF = mu_I * dI + mu_D * dDI
        dH = mu_H * dI

        return [dS, dE, dI, dDE, dDI, dR, dF, dH]

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def run_epoch(self, runtime, dt=0.1):

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create a list of times at which the ODE solver should output system values.
        # Append this list of times as the model's timeseries
        t_eval = numpy.arange(start=self.t, stop=self.t + runtime, step=dt)

        # Define the range of time values for the integration:
        t_span = (self.t, self.t + runtime)

        # Define the initial conditions as the system's current state:
        # (which will be the t=0 condition if this is the first run of this model,
        # else where the last sim left off)

        init_cond = [
            self.numS[-1], self.numE[-1], self.numI[-1], self.numD_E[-1],
            self.numD_I[-1], self.numR[-1], self.numF[-1], self.numH[-1]
        ]

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Solve the system of differential eqns:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        solution = scipy.integrate.solve_ivp(
            lambda t, X: SEIRSModel.
            system_dfes(t, X, self.beta, self.sigma, self.gamma, self.xi, self.
                        mu_I, self.mu_0, self.nu, self.beta_D, self.sigma_D,
                        self.gamma_D, self.mu_D, self.theta_E, self.theta_I,
                        self.psi_E, self.psi_I, self.q, self.mu_H),
            t_span=[self.t, self.tmax],
            y0=init_cond,
            t_eval=t_eval)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Store the solution output as the model's time series and data series:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        self.tseries = numpy.append(self.tseries, solution['t'])
        self.numS = numpy.append(self.numS, solution['y'][0])
        self.numE = numpy.append(self.numE, solution['y'][1])
        self.numI = numpy.append(self.numI, solution['y'][2])
        self.numD_E = numpy.append(self.numD_E, solution['y'][3])
        self.numD_I = numpy.append(self.numD_I, solution['y'][4])
        self.numR = numpy.append(self.numR, solution['y'][5])
        self.numF = numpy.append(self.numF, solution['y'][6])
        self.numH = numpy.append(self.numH, solution['y'][7])

        self.t = self.tseries[-1]

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def run(self, T, dt=0.1, checkpoints=None, verbose=False):

        if (T > 0):
            self.tmax += T
        else:
            return False

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Pre-process checkpoint values:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (checkpoints):
            numCheckpoints = len(checkpoints['t'])
            paramNames = [
                'beta', 'sigma', 'gamma', 'xi', 'mu_I', 'mu_0', 'nu', 'beta_D',
                'sigma_D', 'gamma_D', 'mu_D', 'theta_E', 'theta_I', 'psi_E',
                'psi_I', 'q'
            ]
            for param in paramNames:
                # For params that don't have given checkpoint values (or bad value given),
                # set their checkpoint values to the value they have now for all checkpoints.
                if (param not in list(checkpoints.keys())
                        or not isinstance(checkpoints[param],
                                          (list, numpy.ndarray))
                        or len(checkpoints[param]) != numCheckpoints):
                    checkpoints[param] = [getattr(self, param)
                                          ] * numCheckpoints

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Run the simulation loop:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if (not checkpoints):
            self.run_epoch(runtime=self.tmax, dt=dt)

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            print("t = %.2f" % self.t)
            if (verbose):
                print("\t S   = " + str(self.numS[-1]))
                print("\t E   = " + str(self.numE[-1]))
                print("\t I   = " + str(self.numI[-1]))
                print("\t D_E = " + str(self.numD_E[-1]))
                print("\t D_I = " + str(self.numD_I[-1]))
                print("\t R   = " + str(self.numR[-1]))
                print("\t F   = " + str(self.numF[-1]))

        else:  # checkpoints provided
            for checkpointIdx, checkpointTime in enumerate(checkpoints['t']):
                # Run the sim until the next checkpoint time:
                self.run_epoch(runtime=checkpointTime - self.t, dt=dt)
                # Having reached the checkpoint, update applicable parameters:
                print("[Checkpoint: Updating parameters]")
                for param in paramNames:
                    setattr(self, param, checkpoints[param][checkpointIdx])

                #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                print("t = %.2f" % self.t)
                if (verbose):
                    print("\t S   = " + str(self.numS[-1]))
                    print("\t E   = " + str(self.numE[-1]))
                    print("\t I   = " + str(self.numI[-1]))
                    print("\t D_E = " + str(self.numD_E[-1]))
                    print("\t D_I = " + str(self.numD_I[-1]))
                    print("\t R   = " + str(self.numR[-1]))
                    print("\t F   = " + str(self.numF[-1]))

            if (self.t < self.tmax):
                self.run_epoch(runtime=self.tmax - self.t, dt=dt)

        return True

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def total_num_infections(self, t_idx=None):
        if (t_idx is None):
            return (self.numE[:] + self.numI[:] + self.numD_E[:] +
                    self.numD_I[:])
        else:
            return (self.numE[t_idx] + self.numI[t_idx] + self.numD_E[t_idx] +
                    self.numD_I[t_idx])

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def plot(self,
             ax=None,
             plot_S='line',
             plot_E='line',
             plot_I='line',
             plot_R='line',
             plot_F='line',
             plot_H='line',
             plot_ICU='line',
             plot_D_E='line',
             plot_D_I='line',
             combine_D=True,
             color_S='tab:green',
             color_E='orange',
             color_I='crimson',
             color_R='tab:blue',
             color_F='black',
             color_H='#123478',
             color_ICU='yellow',
             color_D_E='mediumorchid',
             color_D_I='mediumorchid',
             color_reference='#E0E0E0',
             dashed_reference_results=None,
             dashed_reference_label='reference',
             shaded_reference_results=None,
             shaded_reference_label='reference',
             vlines=[],
             vline_colors=[],
             vline_styles=[],
             vline_labels=[],
             ylim=None,
             xlim=None,
             legend=True,
             title=None,
             side_title=None,
             plot_percentages=True):

        import matplotlib.pyplot as pyplot
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Create an Axes object if None provided:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (not ax):
            fig, ax = pyplot.subplots()

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Prepare data series to be plotted:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        Fseries = self.numF / self.N if plot_percentages else self.numF
        Hseries = self.numH / self.N if plot_percentages else self.numH
        ICU_series = Hseries.copy() * self.mu_ICU

        Eseries = self.numE / self.N if plot_percentages else self.numE
        Dseries = (self.numD_E +
                   self.numD_I) / self.N if plot_percentages else (
                       self.numD_E + self.numD_I)
        D_Eseries = self.numD_E / self.N if plot_percentages else self.numD_E
        D_Iseries = self.numD_I / self.N if plot_percentages else self.numD_I
        Iseries = self.numI / self.N if plot_percentages else self.numI
        Rseries = self.numR / self.N if plot_percentages else self.numR
        Sseries = self.numS / self.N if plot_percentages else self.numS

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the reference data:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (dashed_reference_results):
            dashedReference_tseries = dashed_reference_results.tseries[::int(
                self.N / 100)]
            dashedReference_IDEstack = (
                dashed_reference_results.numI +
                dashed_reference_results.numD_I +
                dashed_reference_results.numD_E + dashed_reference_results.numE
            )[::int(self.N / 100)] / (self.N if plot_percentages else 1)
            ax.plot(dashedReference_tseries,
                    dashedReference_IDEstack,
                    color='#E0E0E0',
                    linestyle='--',
                    label='$I+D+E$ (' + dashed_reference_label + ')',
                    zorder=0)
        if (shaded_reference_results):
            shadedReference_tseries = shaded_reference_results.tseries
            shadedReference_IDEstack = (shaded_reference_results.numI +
                                        shaded_reference_results.numD_I +
                                        shaded_reference_results.numD_E +
                                        shaded_reference_results.numE) / (
                                            self.N if plot_percentages else 1)
            ax.fill_between(shaded_reference_results.tseries,
                            shadedReference_IDEstack,
                            0,
                            color='#EFEFEF',
                            label='$I+D+E$ (' + shaded_reference_label + ')',
                            zorder=0)
            ax.plot(shaded_reference_results.tseries,
                    shadedReference_IDEstack,
                    color='#E0E0E0',
                    zorder=1)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the stacked variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        topstack = numpy.zeros_like(self.tseries)
        if (any(Fseries) and plot_F == 'stacked'):
            ax.fill_between(numpy.ma.masked_where(Fseries <= 0, self.tseries),
                            numpy.ma.masked_where(Fseries <= 0,
                                                  topstack + Fseries),
                            topstack,
                            color=color_F,
                            alpha=0.5,
                            label='$F$',
                            zorder=2)
            ax.plot(numpy.ma.masked_where(Fseries <= 0, self.tseries),
                    numpy.ma.masked_where(Fseries <= 0, topstack + Fseries),
                    color=color_F,
                    zorder=3)
            topstack = topstack + Fseries
        if (any(Hseries) and plot_H == 'stacked'):
            ax.fill_between(numpy.ma.masked_where(Hseries <= 0, self.tseries),
                            numpy.ma.masked_where(Hseries <= 0,
                                                  topstack + Hseries),
                            topstack,
                            color=color_H,
                            alpha=0.5,
                            label='$Hospitalized$',
                            zorder=2)
            ax.plot(numpy.ma.masked_where(Hseries <= 0, self.tseries),
                    numpy.ma.masked_where(Hseries <= 0, topstack + Hseries),
                    color=color_H,
                    zorder=3)
        if (any(ICU_series) and plot_ICU == 'stacked'):
            max_ICU = numpy.max(ICU_series)
            if plot_percentages: max_ICU *= REAL_POPULATION

            ax.fill_between(numpy.ma.masked_where(ICU_series <= 0,
                                                  self.tseries),
                            numpy.ma.masked_where(ICU_series <= 0,
                                                  topstack + ICU_series),
                            topstack,
                            color=color_ICU,
                            alpha=0.5,
                            label='$ICU$ max: %d' % max_ICU,
                            zorder=4)
            ax.plot(numpy.ma.masked_where(ICU_series <= 0, self.tseries),
                    numpy.ma.masked_where(ICU_series <= 0,
                                          topstack + ICU_series),
                    color=color_ICU,
                    zorder=5)
            topstack = topstack + ICU_series
        if (any(Eseries) and plot_E == 'stacked'):
            ax.fill_between(numpy.ma.masked_where(Eseries <= 0, self.tseries),
                            numpy.ma.masked_where(Eseries <= 0,
                                                  topstack + Eseries),
                            topstack,
                            color=color_E,
                            alpha=0.5,
                            label='$E$',
                            zorder=2)
            ax.plot(numpy.ma.masked_where(Eseries <= 0, self.tseries),
                    numpy.ma.masked_where(Eseries <= 0, topstack + Eseries),
                    color=color_E,
                    zorder=3)
            topstack = topstack + Eseries
        if (combine_D and plot_D_E == 'stacked' and plot_D_I == 'stacked'):
            ax.fill_between(numpy.ma.masked_where(Dseries <= 0, self.tseries),
                            numpy.ma.masked_where(Dseries <= 0,
                                                  topstack + Dseries),
                            topstack,
                            color=color_D_E,
                            alpha=0.5,
                            label='$D_{all}$',
                            zorder=2)
            ax.plot(numpy.ma.masked_where(Dseries <= 0, self.tseries),
                    numpy.ma.masked_where(Dseries <= 0, topstack + Dseries),
                    color=color_D_E,
                    zorder=3)
            topstack = topstack + Dseries
        else:
            if (any(D_Eseries) and plot_D_E == 'stacked'):
                ax.fill_between(numpy.ma.masked_where(D_Eseries <= 0,
                                                      self.tseries),
                                numpy.ma.masked_where(D_Eseries <= 0,
                                                      topstack + D_Eseries),
                                topstack,
                                color=color_D_E,
                                alpha=0.5,
                                label='$D_E$',
                                zorder=2)
                ax.plot(numpy.ma.masked_where(D_Eseries <= 0, self.tseries),
                        numpy.ma.masked_where(D_Eseries <= 0,
                                              topstack + D_Eseries),
                        color=color_D_E,
                        zorder=3)
                topstack = topstack + D_Eseries
            if (any(D_Iseries) and plot_D_I == 'stacked'):
                ax.fill_between(numpy.ma.masked_where(D_Iseries <= 0,
                                                      self.tseries),
                                numpy.ma.masked_where(D_Iseries <= 0,
                                                      topstack + D_Iseries),
                                topstack,
                                color=color_D_I,
                                alpha=0.5,
                                label='$D_I$',
                                zorder=2)
                ax.plot(numpy.ma.masked_where(D_Iseries <= 0, self.tseries),
                        numpy.ma.masked_where(D_Iseries <= 0,
                                              topstack + D_Iseries),
                        color=color_D_I,
                        zorder=3)
                topstack = topstack + D_Iseries
        if (any(Iseries) and plot_I == 'stacked'):
            ax.fill_between(numpy.ma.masked_where(Iseries <= 0, self.tseries),
                            numpy.ma.masked_where(Iseries <= 0,
                                                  topstack + Iseries),
                            topstack,
                            color=color_I,
                            alpha=0.5,
                            label='$I$',
                            zorder=2)
            ax.plot(numpy.ma.masked_where(Iseries <= 0, self.tseries),
                    numpy.ma.masked_where(Iseries <= 0, topstack + Iseries),
                    color=color_I,
                    zorder=3)
            topstack = topstack + Iseries
        if (any(Rseries) and plot_R == 'stacked'):
            ax.fill_between(numpy.ma.masked_where(Rseries <= 0, self.tseries),
                            numpy.ma.masked_where(Rseries <= 0,
                                                  topstack + Rseries),
                            topstack,
                            color=color_R,
                            alpha=0.5,
                            label='$R$',
                            zorder=2)
            ax.plot(numpy.ma.masked_where(Rseries <= 0, self.tseries),
                    numpy.ma.masked_where(Rseries <= 0, topstack + Rseries),
                    color=color_R,
                    zorder=3)
            topstack = topstack + Rseries
        if (any(Sseries) and plot_S == 'stacked'):
            ax.fill_between(numpy.ma.masked_where(Sseries <= 0, self.tseries),
                            numpy.ma.masked_where(Sseries <= 0,
                                                  topstack + Sseries),
                            topstack,
                            color=color_S,
                            alpha=0.5,
                            label='$S$',
                            zorder=2)
            ax.plot(numpy.ma.masked_where(Sseries <= 0, self.tseries),
                    numpy.ma.masked_where(Sseries <= 0, topstack + Sseries),
                    color=color_S,
                    zorder=3)
            topstack = topstack + Sseries

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the shaded variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (any(Fseries) and plot_F == 'shaded'):
            ax.fill_between(numpy.ma.masked_where(Fseries <= 0, self.tseries),
                            numpy.ma.masked_where(Fseries <= 0, Fseries),
                            0,
                            color=color_F,
                            alpha=0.5,
                            label='$F$',
                            zorder=4)
            ax.plot(numpy.ma.masked_where(Fseries <= 0, self.tseries),
                    numpy.ma.masked_where(Fseries <= 0, Fseries),
                    color=color_F,
                    zorder=5)
        if (any(Hseries) and plot_H == 'shaded'):
            ax.fill_between(numpy.ma.masked_where(Hseries <= 0, self.tseries),
                            numpy.ma.masked_where(Hseries <= 0, Hseries),
                            0,
                            color=color_H,
                            alpha=0.5,
                            label='$H$',
                            zorder=4)
            ax.plot(numpy.ma.masked_where(Hseries <= 0, self.tseries),
                    numpy.ma.masked_where(Hseries <= 0, Hseries),
                    color=color_H,
                    zorder=5)
        if (any(Eseries) and plot_E == 'shaded'):
            ax.fill_between(numpy.ma.masked_where(Eseries <= 0, self.tseries),
                            numpy.ma.masked_where(Eseries <= 0, Eseries),
                            0,
                            color=color_E,
                            alpha=0.5,
                            label='$E$',
                            zorder=4)
            ax.plot(numpy.ma.masked_where(Eseries <= 0, self.tseries),
                    numpy.ma.masked_where(Eseries <= 0, Eseries),
                    color=color_E,
                    zorder=5)
        if (combine_D and
            (any(Dseries) and plot_D_E == 'shaded' and plot_D_E == 'shaded')):
            ax.fill_between(numpy.ma.masked_where(Dseries <= 0, self.tseries),
                            numpy.ma.masked_where(Dseries <= 0, Dseries),
                            0,
                            color=color_D_E,
                            alpha=0.5,
                            label='$D_{all}$',
                            zorder=4)
            ax.plot(numpy.ma.masked_where(Dseries <= 0, self.tseries),
                    numpy.ma.masked_where(Dseries <= 0, Dseries),
                    color=color_D_E,
                    zorder=5)
        else:
            if (any(D_Eseries) and plot_D_E == 'shaded'):
                ax.fill_between(numpy.ma.masked_where(D_Eseries <= 0,
                                                      self.tseries),
                                numpy.ma.masked_where(D_Eseries <= 0,
                                                      D_Eseries),
                                0,
                                color=color_D_E,
                                alpha=0.5,
                                label='$D_E$',
                                zorder=4)
                ax.plot(numpy.ma.masked_where(D_Eseries <= 0, self.tseries),
                        numpy.ma.masked_where(D_Eseries <= 0, D_Eseries),
                        color=color_D_E,
                        zorder=5)
            if (any(D_Iseries) and plot_D_I == 'shaded'):
                ax.fill_between(numpy.ma.masked_where(D_Iseries <= 0,
                                                      self.tseries),
                                numpy.ma.masked_where(D_Iseries <= 0,
                                                      D_Iseries),
                                0,
                                color=color_D_I,
                                alpha=0.5,
                                label='$D_I$',
                                zorder=4)
                ax.plot(numpy.ma.masked_where(D_Iseries <= 0, self.tseries),
                        numpy.ma.masked_where(D_Iseries <= 0, D_Iseries),
                        color=color_D_I,
                        zorder=5)
        if (any(Iseries) and plot_I == 'shaded'):
            ax.fill_between(numpy.ma.masked_where(Iseries <= 0, self.tseries),
                            numpy.ma.masked_where(Iseries <= 0, Iseries),
                            0,
                            color=color_I,
                            alpha=0.5,
                            label='$I$',
                            zorder=4)
            ax.plot(numpy.ma.masked_where(Iseries <= 0, self.tseries),
                    numpy.ma.masked_where(Iseries <= 0, Iseries),
                    color=color_I,
                    zorder=5)
        if (any(Sseries) and plot_S == 'shaded'):
            ax.fill_between(numpy.ma.masked_where(Sseries <= 0, self.tseries),
                            numpy.ma.masked_where(Sseries <= 0, Sseries),
                            0,
                            color=color_S,
                            alpha=0.5,
                            label='$S$',
                            zorder=4)
            ax.plot(numpy.ma.masked_where(Sseries <= 0, self.tseries),
                    numpy.ma.masked_where(Sseries <= 0, Sseries),
                    color=color_S,
                    zorder=5)
        if (any(Rseries) and plot_R == 'shaded'):
            ax.fill_between(numpy.ma.masked_where(Rseries <= 0, self.tseries),
                            numpy.ma.masked_where(Rseries <= 0, Rseries),
                            0,
                            color=color_R,
                            alpha=0.5,
                            label='$R$',
                            zorder=4)
            ax.plot(numpy.ma.masked_where(Rseries <= 0, self.tseries),
                    numpy.ma.masked_where(Rseries <= 0, Rseries),
                    color=color_R,
                    zorder=5)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the line variables:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (any(Fseries) and plot_F == 'line'):
            ax.plot(numpy.ma.masked_where(Fseries <= 0, self.tseries),
                    numpy.ma.masked_where(Fseries <= 0, Fseries),
                    color=color_F,
                    label='$F$',
                    zorder=6)
        if (any(Hseries) and plot_H == 'line'):
            ax.plot(numpy.ma.masked_where(Hseries <= 0, self.tseries),
                    numpy.ma.masked_where(Hseries <= 0, Hseries),
                    color=color_H,
                    label='$H$',
                    zorder=6)
        if (any(Eseries) and plot_E == 'line'):
            ax.plot(numpy.ma.masked_where(Eseries <= 0, self.tseries),
                    numpy.ma.masked_where(Eseries <= 0, Eseries),
                    color=color_E,
                    label='$E$',
                    zorder=6)
        if (combine_D and
            (any(Dseries) and plot_D_E == 'line' and plot_D_E == 'line')):
            ax.plot(numpy.ma.masked_where(Dseries <= 0, self.tseries),
                    numpy.ma.masked_where(Dseries <= 0, Dseries),
                    color=color_D_E,
                    label='$D_{all}$',
                    zorder=6)
        else:
            if (any(D_Eseries) and plot_D_E == 'line'):
                ax.plot(numpy.ma.masked_where(D_Eseries <= 0, self.tseries),
                        numpy.ma.masked_where(D_Eseries <= 0, D_Eseries),
                        color=color_D_E,
                        label='$D_E$',
                        zorder=6)
            if (any(D_Iseries) and plot_D_I == 'line'):
                ax.plot(numpy.ma.masked_where(D_Iseries <= 0, self.tseries),
                        numpy.ma.masked_where(D_Iseries <= 0, D_Iseries),
                        color=color_D_I,
                        label='$D_I$',
                        zorder=6)
        if (any(Iseries) and plot_I == 'line'):
            ax.plot(numpy.ma.masked_where(Iseries <= 0, self.tseries),
                    numpy.ma.masked_where(Iseries <= 0, Iseries),
                    color=color_I,
                    label='$I$',
                    zorder=6)
        if (any(Sseries) and plot_S == 'line'):
            ax.plot(numpy.ma.masked_where(Sseries <= 0, self.tseries),
                    numpy.ma.masked_where(Sseries <= 0, Sseries),
                    color=color_S,
                    label='$S$',
                    zorder=6)
        if (any(Rseries) and plot_R == 'line'):
            ax.plot(numpy.ma.masked_where(Rseries <= 0, self.tseries),
                    numpy.ma.masked_where(Rseries <= 0, Rseries),
                    color=color_R,
                    label='$R$',
                    zorder=6)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the vertical line annotations:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (len(vlines) > 0 and len(vline_colors) == 0):
            vline_colors = ['gray'] * len(vlines)
        if (len(vlines) > 0 and len(vline_labels) == 0):
            vline_labels = [None] * len(vlines)
        if (len(vlines) > 0 and len(vline_styles) == 0):
            vline_styles = [':'] * len(vlines)
        for vline_x, vline_color, vline_style, vline_label in zip(
                vlines, vline_colors, vline_styles, vline_labels):
            if (vline_x is not None):
                ax.axvline(x=vline_x,
                           color=vline_color,
                           linestyle=vline_style,
                           alpha=1,
                           label=vline_label)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the horizontal line annotations:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ICU_available_beds = 1772
        ax.axhline(y=ICU_available_beds / REAL_POPULATION,
                   color=color_ICU,
                   linestyle=":",
                   alpha=1,
                   label="ICU_available_beds: %d" % ICU_available_beds)

        Total_available_beds = 39955
        ax.axhline(y=Total_available_beds / REAL_POPULATION,
                   color=color_H,
                   linestyle=":",
                   alpha=1,
                   label="Total_available_beds: %d" % Total_available_beds)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Draw the plot labels:
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ax.set_xlabel('days')
        ax.set_ylabel('percent of population'
                      if plot_percentages else 'number of individuals')
        ax.set_xlim(0, (max(self.tseries) if not xlim else xlim))
        ax.set_ylim(0, ylim)
        if (plot_percentages):
            ax.set_yticklabels(['{:,.1%}'.format(y) for y in ax.get_yticks()])

        if (legend):
            legend_handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend(legend_handles[::-1],
                      legend_labels[::-1],
                      loc='upper right',
                      facecolor='white',
                      edgecolor='none',
                      framealpha=0.9,
                      prop={'size': 12})
        if (title):
            ax.set_title(title, size=12)
        if (side_title):
            ax.annotate(side_title, (0, 0.5),
                        xytext=(-45, 0),
                        ha='right',
                        va='center',
                        size=12,
                        rotation=90,
                        xycoords='axes fraction',
                        textcoords='offset points')

        return ax
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def figure_basic(self,
                     plot_S='line',
                     plot_E='line',
                     plot_I='line',
                     plot_R='line',
                     plot_F='line',
                     plot_H='line',
                     plot_ICU='line',
                     plot_D_E='line',
                     plot_D_I='line',
                     combine_D=True,
                     color_S='tab:green',
                     color_E='orange',
                     color_I='crimson',
                     color_R='tab:blue',
                     color_F='black',
                     color_H='#123478',
                     color_ICU='yellow',
                     color_D_E='mediumorchid',
                     color_D_I='mediumorchid',
                     color_reference='#E0E0E0',
                     dashed_reference_results=None,
                     dashed_reference_label='reference',
                     shaded_reference_results=None,
                     shaded_reference_label='reference',
                     vlines=[],
                     vline_colors=[],
                     vline_styles=[],
                     vline_labels=[],
                     ylim=None,
                     xlim=None,
                     legend=True,
                     title=None,
                     side_title=None,
                     plot_percentages=True,
                     figsize=(12, 8),
                     use_seaborn=True,
                     show=True):

        import matplotlib.pyplot as pyplot

        fig, ax = pyplot.subplots(figsize=figsize)

        if (use_seaborn):
            import seaborn
            seaborn.set_style('ticks')
            seaborn.despine()

        self.plot(ax=ax,
                  plot_S=plot_S,
                  plot_E=plot_E,
                  plot_I=plot_I,
                  plot_R=plot_R,
                  plot_F=plot_F,
                  plot_H=plot_H,
                  plot_ICU=plot_ICU,
                  plot_D_E=plot_D_E,
                  plot_D_I=plot_D_I,
                  combine_D=combine_D,
                  color_S=color_S,
                  color_E=color_E,
                  color_I=color_I,
                  color_R=color_R,
                  color_F=color_F,
                  color_H=color_H,
                  color_ICU=color_ICU,
                  color_D_E=color_D_E,
                  color_D_I=color_D_I,
                  color_reference=color_reference,
                  dashed_reference_results=dashed_reference_results,
                  dashed_reference_label=dashed_reference_label,
                  shaded_reference_results=shaded_reference_results,
                  shaded_reference_label=shaded_reference_label,
                  vlines=vlines,
                  vline_colors=vline_colors,
                  vline_styles=vline_styles,
                  vline_labels=vline_labels,
                  ylim=ylim,
                  xlim=xlim,
                  legend=legend,
                  title=title,
                  side_title=side_title,
                  plot_percentages=plot_percentages)

        if (show):
            pyplot.show()

        return fig, ax
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def figure_infections(self,
                          plot_S=False,
                          plot_E=False,
                          plot_I='stacked',
                          plot_R=False,
                          plot_F='shaded',
                          plot_H='stacked',
                          plot_ICU='stacked',
                          plot_D_E=False,
                          plot_D_I=False,
                          combine_D=True,
                          color_S='tab:green',
                          color_E='orange',
                          color_I='crimson',
                          color_R='tab:blue',
                          color_F='black',
                          color_H='#123478',
                          color_ICU='yellow',
                          color_D_E='mediumorchid',
                          color_D_I='mediumorchid',
                          color_reference='#E0E0E0',
                          dashed_reference_results=None,
                          dashed_reference_label='reference',
                          shaded_reference_results=None,
                          shaded_reference_label='reference',
                          vlines=[],
                          vline_colors=[],
                          vline_styles=[],
                          vline_labels=[],
                          ylim=None,
                          xlim=None,
                          legend=True,
                          title=None,
                          side_title=None,
                          plot_percentages=True,
                          figsize=(12, 8),
                          use_seaborn=True,
                          show=True):

        import matplotlib.pyplot as pyplot
        fig, ax = pyplot.subplots(figsize=figsize)

        if (use_seaborn):
            import seaborn
            seaborn.set_style('ticks')
            seaborn.despine()

        self.plot(ax=ax,
                  plot_S=plot_S,
                  plot_E=plot_E,
                  plot_I=plot_I,
                  plot_R=plot_R,
                  plot_F=plot_F,
                  plot_H=plot_H,
                  plot_ICU=plot_ICU,
                  plot_D_E=plot_D_E,
                  plot_D_I=plot_D_I,
                  combine_D=combine_D,
                  color_S=color_S,
                  color_E=color_E,
                  color_I=color_I,
                  color_R=color_R,
                  color_F=color_F,
                  color_H=color_H,
                  color_ICU=color_ICU,
                  color_D_E=color_D_E,
                  color_D_I=color_D_I,
                  color_reference=color_reference,
                  dashed_reference_results=dashed_reference_results,
                  dashed_reference_label=dashed_reference_label,
                  shaded_reference_results=shaded_reference_results,
                  shaded_reference_label=shaded_reference_label,
                  vlines=vlines,
                  vline_colors=vline_colors,
                  vline_styles=vline_styles,
                  vline_labels=vline_labels,
                  ylim=ylim,
                  xlim=xlim,
                  legend=legend,
                  title=title,
                  side_title=side_title,
                  plot_percentages=plot_percentages)

        if (show):
            pyplot.show()

        return fig, ax


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define a custom method for generating
# power-law-like graphs with exponential tails
# both above and below the degree mean and
# where the mean degree be easily down-shifted
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def custom_exponential_graph(base_graph=None,
                             scale=100,
                             min_num_edges=0,
                             m=9,
                             n=None):
    # Generate a random preferential attachment power law graph as a starting point.
    # By the way this graph is constructed, it is expected to have 1 connected component.
    # Every node is added along with m=8 edges, so the min degree is m=8.
    if (base_graph):
        graph = base_graph.copy()
    else:
        assert (
            n is not None
        ), "Argument n (number of nodes) must be provided when no base graph is given."
        graph = networkx.barabasi_albert_graph(n=n, m=m)

    # To get a graph with power-law-esque properties but without the fixed minimum degree,
    # We modify the graph by probabilistically dropping some edges from each node.
    for node in graph:
        neighbors = list(graph[node].keys())
        quarantineEdgeNum = int(
            max(
                min(numpy.random.exponential(scale=scale, size=1),
                    len(neighbors)), min_num_edges))
        quarantineKeepNeighbors = numpy.random.choice(neighbors,
                                                      size=quarantineEdgeNum,
                                                      replace=False)
        for neighbor in neighbors:
            if (neighbor not in quarantineKeepNeighbors):
                graph.remove_edge(node, neighbor)

    return graph


#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def plot_degree_distn(graph, max_degree=None, show=True, use_seaborn=True):
    import matplotlib.pyplot as pyplot
    if (use_seaborn):
        import seaborn
        seaborn.set_style('ticks')
        seaborn.despine()
    # Get a list of the node degrees:
    if type(graph) == numpy.ndarray:
        nodeDegrees = graph.sum(axis=0).reshape(
            (graph.shape[0], 1))  # sums of adj matrix cols
    elif type(graph) == networkx.classes.graph.Graph:
        nodeDegrees = [d[1] for d in graph.degree()]
    else:
        raise BaseException(
            "Input an adjacency matrix or networkx object only.")
    # Calculate the mean degree:
    meanDegree = numpy.mean(nodeDegrees)
    # Generate a histogram of the node degrees:
    pyplot.hist(nodeDegrees,
                bins=range(max(nodeDegrees)),
                alpha=0.5,
                color='tab:blue',
                label=('mean degree = %.1f' % meanDegree))
    pyplot.xlim(0, max(nodeDegrees) if not max_degree else max_degree)
    pyplot.xlabel('degree')
    pyplot.ylabel('num nodes')
    pyplot.legend(loc='upper right')
    if (show):
        pyplot.show()
