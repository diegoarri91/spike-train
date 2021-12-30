import itertools
import pickle
import numpy as np
import matplotlib.pyplot as plt

from .utils import extend_trues, get_arg, get_dt, searchsorted, shift_mask


class SpikeTrains:
    r"""
    Implements spike trains operations such as computation of psth, ISI distribution, dot products and different
    similarity measures between sets of spike trains[1, 2].

    References
    [1] Gerstner et al. 2014. "Neuronal dynamics: From single neurons to networks and models of cognition"
    [2] Naud et al. 2011. “Improved Similarity Measures for Small Sets of Spike Trains”
    """

    def __init__(self, t, mask):
        self.t = t
        self.mask = mask
        self.dt = get_dt(t)
        self.ntrains = mask.shape[1]

    def average_dot_product(self, st, kernel1, kernel2):
        r"""
        Returns the average inner product between all self spike trains and all st spike trains.
        """
        average_spike_train_self = self.average_spiketrain()
        average_spike_train_st = st.average_spiketrain()
        return average_spike_train_self.inner_product(average_spike_train_st, kernel1, kernel2)[0]

    def average_spiketrain(self):
        r""""
        Returns the average spike train of the spike trains. Also called population activity or instantaneous firing
        rate. Eq.2.9, Naud et al. 2011.
        """
        return SpikeTrains(self.t, np.sum(self.mask, 1)[:, None] / self.ntrains)

    def average_squared_norm(self, kernel1, kernel2):
        r"""
        Returns L (Eq2.10, Naud et al. 2011), the average of the squared norms of the spike trains.
        """
        return np.mean(self.norm_squared(kernel1, kernel2))

    def convolve(self, kernel):
        r"""
        Returns the convolution of the spike trains with the kernel.
        """
        return kernel.convolve(self.t, self.mask / self.dt)

    def inner_product(self, st, kernel1, kernel2):
        r"""
        Returns the inner product between the spike trains and a second SpikeTrains instance st. The inner product is
        defined from the two kernels kernel1 and kernel2 that are each convolved with the two SpikeTrains instances.
        """
        assert np.all(st.t == self.t), 'The two SpikeTrains instances are expected to have the same time points.'
        self_conv = kernel1.convolve(self.t, self.mask / self.dt)
        st_conv = kernel2.convolve(self.t, st.mask / self.dt)
        return np.sum(self_conv * st_conv, 0) * self.dt

    def inner_product_matrix(self, st, kernel1, kernel2):
        r"""
        Returns a matrix which entries are all possible inner products between the spike trains in self and the spike
        trains in st. inner_product_matrix[i, j] = <self_i, st_j>
        """
        # By making n1 <= n2 we make the computation more efficient
        nself, nst = self.ntrains, st.ntrains
        if nself <= nst:
            n1, n2 = nself, nst
            mask_spk1, mask_spk2 = self.mask, st.mask
        else:
            n2, n1 = nself, nst
            mask_spk2, mask_spk1 = self.mask, st.mask

        inner_product_matrix = np.zeros((n1, n2)) * np.nan
        index1 = np.arange(0, n1)
        for ii in range(n2):
            index2 = (index1 + ii) % n2
            st_self = SpikeTrains(self.t, mask_spk1)
            st2 = SpikeTrains(self.t, np.roll(mask_spk2, -ii, axis=1)[:, :n1])
            inner_product_matrix[(index1, index2)] = st_self.inner_product(st2, kernel1, kernel2)

        if nself > nst:
            inner_product_matrix = inner_product_matrix.T

        return inner_product_matrix

    def fano_factor(self, bins):
        spk_count = self.get_spike_count(bins, average_sweeps=False)
        return np.var(spk_count, 1) / np.mean(spk_count, 1)

    def interspike_intervals(self, concatenate=True):
        r"""
        Returns all the Interspike Intervals (ISIs) from the spike trains
        """
        isis = []
        for j in range(self.ntrains):
            if np.sum(self.mask[:, j]) > 1:
                isis_j = np.diff(self.t[self.mask[:, j]], axis=0)
            else:
                isis_j = np.array([])
            isis.append(isis_j)
        if concatenate:
            isis = np.concatenate(isis, axis=0)
        return isis

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as pk_f:
            dic = pickle.load(pk_f)

        arg0 = dic.get('arg0', 0)
        t = np.arange(arg0, arg0 + dic['shape'][0], 1) * dic['dt']
        mask = np.zeros(dic['shape'], dtype=bool)
        mask[dic['arg_spikes']] = True
        return cls(t, mask)

    def Ma(self, st, kernel1, kernel2, biased=True):
        r"""
        Returns the angle-based similarity (Ma) between the set of spike trains in self and the set in st (Eq.2.27,
        Naud et al. 2011). This quantity is biased or unbiased depending on the population norms used.
        """
        average_dot_self_st = self.average_dot_product(st, kernel1, kernel2)
        population_norm_self = self.population_norm(kernel1, kernel2, biased=biased)
        population_norm_st = st.population_norm(kernel1, kernel2, biased=biased)
        ma = average_dot_self_st / (population_norm_self * population_norm_st)
        return ma

    def Md(self, st, kernel1, kernel2, biased=True):
        r"""
        Returns the distance-based similarity (Md) between the set of spike trains in self and the set in st (Eq.2.27,
        Naud et al. 2011). This quantity is biased or unbiased depending on the population norms used.
        """
        average_dot_self_st = self.average_dot_product(st, kernel1, kernel2)
        population_norm_self = self.population_norm(kernel1, kernel2, biased=biased)
        population_norm_st = st.population_norm(kernel1, kernel2, biased=biased)
        md = 2. * average_dot_self_st / (population_norm_self ** 2. + population_norm_st ** 2.)
        return md

    def norm(self, kernel1, kernel2):
        r"""
        Returns a np.array with the norms of the spike trains
        """
        return np.sqrt(norm_squared(self, kernel1, kernel2))

    def norm_squared(self, kernel1, kernel2):
        r"""
        Returns a np.array with the squared norms of the spike trains
        """
        return self.inner_product(self, kernel1, kernel2)

    def plot(self, ax=None, offset=0, figsize=(8, 4), **kwargs):
        r"""
        Plots the spike train as a raster plot. Returns the matplotlib axes.
        """
        kwargs.setdefault('color', 'C0')
        kwargs.setdefault('marker', 'o')
        kwargs.setdefault('ms', 7)
        kwargs.setdefault('mew', 1)

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            x_extra_range = (self.t[-1] - self.t[0]) * .04
            ax.set_xlim(self.t[0] - x_extra_range, self.t[-1] + x_extra_range)
            y_extra_range = (self.ntrains - 1) * .04
            ax.set_ylim(-y_extra_range, self.ntrains - 1 + y_extra_range)

        if self.mask.ndim == 2:
            arg_spikes = np.where(self.mask)
        else:
            arg_spikes = np.where(self.mask.reshape(len(self.t), -1, order='F'))
        ax.plot(self.t[arg_spikes[0]], offset + arg_spikes[1], linestyle='', **kwargs)

        return ax

    @classmethod
    def poisson_process_hom(cls, rate, time_end, dt, ntrains=1, time_start=0.):
        r"""
        Returns a SpikeTrains instance with samples from a homogeneous Poisson Process with the given rate parameter
        """
        t = np.arange(time_start, time_end, dt)
        rand = np.random.rand(len(t), ntrains)
        mask = rate * dt > rand
        return SpikeTrains(t, mask)
    
    @classmethod
    def poisson_process_inh(cls, rate, dt, ntrains=1, time_start=0.):
        r"""
        Returns a SpikeTrains instance with samples from an inhhomogeneous Poisson Process with the given time dependent
        rate
        """
        t = np.arange(len(rate)) * dt - time_start
        rand_shape = rate.shape + (ntrains,)
        rand = np.random.rand(*rand_shape)
        mask = rate[..., None] * dt > rand
        return SpikeTrains(t, mask)

    def population_norm(self, kernel1, kernel2, biased=True):
        r"""
        Returns the norm of the population activity (average spike train)
        """
        return np.sqrt(self.population_norm_squared(kernel1, kernel2, biased=biased))

    def population_norm_squared(self, kernel1, kernel2, biased=True):
        r"""
        Returns the squared norm of the population activity (average spike train). The estimator is biased when it
        incorporates inner products of a spike train with itself (Eq.3.4, Naud et al. 2011) and is unbiased when it
        excludes these terms (Eq.3.6, Naud et al. 2011).
        """
        average_spike_train = self.average_spiketrain()
        if biased or self.ntrains == 1:
            pop_norm_squared = average_spike_train.norm_squared(kernel1, kernel2)[0]
        else:
            inner_sum_all = self.ntrains ** 2. * average_spike_train.norm_squared(kernel1, kernel2)[0] # ij & ji
            inner_sum_self = np.sum(self.inner_product(self, kernel1, kernel2))
            pop_norm_squared = (inner_sum_all - inner_sum_self) / (self.ntrains * (self.ntrains - 1))
        return pop_norm_squared

    def prespike_average(self, signal, tl, tr=0, throw_spikes=True):

        argl, argr = get_arg([tl, tr], self.dt)

        signal = np.copy(signal)
        mask_around_spk = extend_trues(shift_mask(self.mask, 1, fill_value=False), 0, argl)

        if throw_spikes:
            mask_spk = self.mask & ~mask_around_spk
        else:
            mask_spk = self.mask
            signal[mask_around_spk] = np.nan

        mask_spk[:argl] = False
        index = np.where(mask_spk)
        sta = [signal[i_spk - argl:i_spk - argr + 1, sw] for i_spk, sw in zip(*index)]
        sta = np.stack(sta, 1)

        t_sta = np.arange(-argl, -argr + 1, 1) * self.dt

        return t_sta, sta

    def psth(self, kernel):
        r"""
        Returns the Peri-Stimulus Histogram (PSTH) of the spike trains using the given kernel.
        """
        average_spike_train = self.average_spiketrain()
        psh = average_spike_train.convolve(kernel)
        return psh

    def reliability(self, kernel1, kernel2):
        r"""
        Returns the reliability, R, of the spike trains (Eqs. 2.15 & 3.7, Naud et al. 2011).
        """
        average_spike_train = self.average_spiketrain()
        inner_sum_all = self.ntrains ** 2. * average_spike_train.norm_squared(kernel1, kernel2)[0]  # ij & ji
        inner_sum_self = np.sum(self.inner_product(self, kernel1, kernel2))
        population_norm_squared = (inner_sum_all - inner_sum_self) / (self.ntrains - 1)
        rel = population_norm_squared / inner_sum_self
        return rel

    def restrict(self, start_time=None, end_time=None, start_time_zero=True):
        r"""
        Returns a new SpikeTrain restricted to the time window between start_time and end_time. If start_time_zero is
        True the new SpikeTrain starts at t=0
        """
        start_time = start_time if start_time is not None else self.t[0]
        end_time = end_time if end_time is not None else self.t[-1] + self.dt
        arg0, argf = searchsorted(self.t, [start_time, end_time])

        t = self.t[arg0:argf]
        mask = self.mask[arg0:argf]

        if start_time_zero:
            t = t - t[0]

        return SpikeTrains(t, mask)

    def save(self, path):
        dic = {'arg0': get_arg(self.t[0], self.dt), 'dt': self.dt, 'arg_spikes': np.where(self.mask),
               'shape': self.mask.shape}
        with open(path, 'wb') as pk_f:
            pickle.dump(dic, pk_f)

    def spike_triggered_average(self, signal, start_time, end_time, refractory_time=None, average=True):
        r"""
        Returns the spike triggered average of the given signal. That is the values of the signal previous to the spikes
        in the spike trains. start_time and end_time define the time window to return.
        """
        start_idx = get_arg(start_time, self.dt)
        end_idx = get_arg(end_time, self.dt)

        left_padding = np.empty((start_idx, ) + signal.shape[1:]) * np.nan
        right_padding = np.empty((end_idx,) + signal.shape[1:]) * np.nan
        signal = np.concatenate((left_padding, signal, right_padding), axis=0)

        left_padding = np.zeros((start_idx,) + self.mask.shape[1:], dtype=bool)
        right_padding = np.zeros((end_idx,) + self.mask.shape[1:], dtype=bool)
        mask_spk = np.concatenate((left_padding, self.mask, right_padding), axis=0)

        if refractory_time is not None:
            ref_idx = get_arg(refractory_time, self.dt)
            mask_spk = mask_spk & ~extend_trues(shift_mask(mask_spk, 1, fill_value=False), 0, ref_idx)

        index = np.where(mask_spk)
        sta = [signal[i_spk - start_idx:i_spk + end_idx + 1, sw] for i_spk, sw in zip(*index)]
        sta = np.stack(sta, 1)
        if average:
            sta = np.nanmean(sta, 1)
        t_sta = np.arange(-start_idx, end_idx + 1, 1) * self.dt

        return t_sta, sta

    def sliding_fano_factor(self, kernel):
        r"""
        Returns the sliding fano factor of the spike trains by first convolving them with the kernel.
        """
        conv = self.convolve(kernel)
        mean, var = np.mean(conv, 1), np.var(conv, 1, ddof=1)
        fano = np.ones(len(conv))
        mask = mean > 0
        fano[mask] = var[mask] / mean[mask]
        return fano

    def spike_count(self, time_bins):
        r"""
        Returns an np.array with the spike counts in the given time bins.
        """
        arg_bins = searchsorted(self.t, time_bins)
        spk_count = [np.sum(self.mask[arg0:arg1], 0) for arg0, arg1 in zip(arg_bins[:-1], arg_bins[1:])]
        spk_count = np.stack(spk_count, 0)
        return spk_count

    def index_trains(self, idx):
        r"""
        Returns a new SpikeTrains index with the spike trains indexed by idx.
        """
        return SpikeTrains(self.t, self.mask[:, idx])

    def spike_times(self):
        r"""
        Returns a tuple containing a np.array with the spike times in the first element and the train indexes in the
        other elements.
        """
        args = np.where(self.mask_)
        t_spk = (self.t[args[0]],) + args[1:]
        return t_spk
