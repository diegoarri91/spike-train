import itertools
import pickle
import numpy as np
import matplotlib.pyplot as plt

from .utils import extend_trues, get_arg, get_dt, searchsorted, shift_mask


class SpikeTrains:
    r"""
    Implements spike trains operations such as computation of psth, ISI distribution, dot products and different
    similarity measures between sets of spike trains[1, 2].

    [1] Gerstner et al 2014. "Neuronal dynamics: From single neurons to networks and models of cognition"
    [2] Naud et al 2011. “Improved Similarity Measures for Small Sets of Spike Trains”
    """

    def __init__(self, t, mask):
        self.t = t
        self.mask = mask
        self.dt = get_dt(t)
        self.ntrains = mask.shape[1]

    def average_dot_product(self, st, kernel1, kernel2):
        r"""
        Computes the average dot product between all self spike trains and all st spike trains
        """
        average_spike_train_self = self.average_spiketrain()
        average_spike_train_st = st.average_spiketrain()
        return average_spike_train_self.dot_product(average_spike_train_st, kernel1, kernel2)[0]

    def average_spiketrain(self):
        r""""
        Computes the average spike train or instantaneous firing rate of the spike trains
        """
        return SpikeTrains(self.t, np.sum(self.mask, 1)[:, None] / self.ntrains)

    def average_squared_norm(self, kernel1, kernel2):
        r"""
        Returns L (Eq Naud 2011), the average of the squared norms of the spike trains
        """
        return np.mean(self.norm_squared(kernel1, kernel2))

    def convolve(self, kernel):
        r"""
        Returns the convolution of the spike trains with the kernel
        """
        return kernel.convolve(self.t, self.mask / self.dt)

    def cosine(self, st, kernel1, kernel2):
        r"""
        Returns the cosine of the angle between spike trains
        """
        self_conv1 = kernel1.convolve(self.t, self.mask / self.dt)
        self_conv2 = kernel2.convolve(self.t, self.mask / self.dt)
        st_conv1 = kernel1.convolve(self.t, st.mask / self.dt)
        st_conv2 = kernel2.convolve(self.t, st.mask / self.dt)

        self_norm = np.sqrt(np.sum(self_conv1 * self_conv2, 0) * self.dt)
        st_norm = np.sqrt(np.sum(st_conv1 * st_conv2, 0) * self.dt)
        dot = np.sum(self_conv1 * st_conv2, 0) * self.dt

        return dot / (self_norm * st_norm)

    def cosine_matrix(self, st, kernel1, kernel2):

        dot_product_matrix = self.dot_product_matrix(st, kernel1, kernel2)

        self_norm, st_norm = self.norm(kernel1, kernel2), st.norm(kernel1, kernel2)
        norm_product_matrix = np.outer(self_norm, st_norm)

        return dot_product_matrix / norm_product_matrix

    def dot_product(self, st, kernel1, kernel2):
        argf = min(len(self.t), len(st.t))

        self_conv = kernel1.convolve(self.t[:argf], self.mask[:argf] / self.dt)
        st_conv = kernel2.convolve(st.t[:argf], st.mask[:argf] / self.dt)

        return np.sum(self_conv * st_conv, 0) * self.dt

    def dot_product_matrix(self, st, kernel1, kernel2):

        # OJO SI TIENEN DIFERENTE dt self e ic
        # N1<=N2
        # dot_product_matrix[i, j] = <Si, Sj>

        Nself = self.ntrains
        Nst = st.ntrains

        if Nself <= Nst:
            N1, N2 = Nself, Nst
            mask_spk1, mask_spk2 = self.mask, st.mask
        else:
            N2, N1 = Nself, Nst
            mask_spk2, mask_spk1 = self.mask, st.mask

        index1 = np.arange(0, N1)
        dot_product_matrix = np.zeros((N1, N2)) * np.nan

        for ii in range(N2):
            index2 = (index1 + ii) % N2

            st_self = SpikeTrains(self.t, mask_spk1)
            st2 = SpikeTrains(self.t, np.roll(mask_spk2, -ii, axis=1)[:, :N1])
            dot_product_matrix[(index1, index2)] = st_self.dot_product(st2, kernel1, kernel2)

        if Nself > Nst:
            dot_product_matrix = dot_product_matrix.T

        return dot_product_matrix

    def fano_factor(self, bins):
        spk_count = self.get_spike_count(bins, average_sweeps=False)
        return np.var(spk_count, 1) / np.mean(spk_count, 1)

    def get_outlier_trials(self, b=5):
        n_spikes = np.sum(self.mask, 0)
        median = np.median(n_spikes)
        mad = np.median(np.abs(n_spikes - median))
        deviations = np.abs(n_spikes - median)
        return deviations > b * mad

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

        average_dot_self_st = self.average_dot_product(st, kernel1, kernel2)
        population_norm_self = self.population_norm(kernel1, kernel2, biased=biased)
        population_norm_st = st.population_norm(kernel1, kernel2, biased=biased)

        return average_dot_self_st / (population_norm_self * population_norm_st)

    def Md(self, st, kernel1, kernel2, biased=True):

        average_dot_self_st = self.average_dot_product(st, kernel1, kernel2)
        population_norm_self = self.population_norm(kernel1, kernel2, biased=biased)
        population_norm_st = st.population_norm(kernel1, kernel2, biased=biased)

        return 2. * average_dot_self_st / (population_norm_self ** 2. + population_norm_st ** 2.)

    def multiple_correlation_matrix(self, sts, delta):

        # n_neurons = self.nsweeps + np.sum(st.nsweeps for st in sts)

        # correlation_matrix = np.zeros( (n_neurons, n_neurons) )

        sts = [self] + sts

        dic = {}
        for (ii, st1), (jj, st2) in itertools.combinations_with_replacement(enumerate(sts), 2):
            dic[ii, jj] = st1.dot_product_matrix(st2, delta)
            dic[jj, ii] = dic[ii, jj]

        return np.block([[dic[ii, jj] for jj in range(len(sts))] for ii in range(len(sts))])

        # for ii, st in enumerate(sts):
        # correlation_matrix[ii*st.nsweeps:(ii+1)*st.nsweeps, ii*st.nsweeps:(ii+1)*st.nsweeps] = st.correlation_matrix(st, delta)

    def norm(self, kernel1, kernel2):
        r"""
        Returns a np.array with the norms of the spike trains
        """
        return np.sqrt(norm_squared(self, kernel1, kernel2))

    def norm_squared(self, kernel1, kernel2):
        r"""
        Returns a np.array with the squared norms of the spike trains
        """
        return self.dot_product(self, kernel1, kernel2)

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

        arg_spikes = np.where(self.mask)
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

        average_spike_train = self.average_spiketrain()

        if biased or self.ntrains == 1:
            population_norm = average_spike_train.norm(kernel1, kernel2)[0]

        else:
            dot_sum_all = self.ntrains ** 2. * average_spike_train.dot_product(average_spike_train, kernel1, kernel2)[
                0]  # ij & ji
            dot_sum_ii = np.sum(self.dot_product(self, kernel1, kernel2))
            population_norm = np.sqrt((dot_sum_all - dot_sum_ii) / (self.ntrains * (self.ntrains - 1.)))

        return population_norm

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

    def psth(self, kernel, average_sweeps=True):
        average_spike_train = self.average_spiketrain()
        return average_spike_train.convolve(kernel)

    def reliability(self, kernel1, kernel2):

        average_spike_train = self.average_spiketrain()

        dot_sum_all = self.ntrains ** 2. * average_spike_train.dot_product(average_spike_train, kernel1, kernel2)[0]  # ij & ji
        dot_ii = self.dot_product(self, kernel1, kernel2)
        population_norm_squared = (dot_sum_all - np.sum(dot_ii)) / (self.ntrains * (self.ntrains - 1.))

        return population_norm_squared / np.mean(dot_ii)

    def reliability2(self, kernel1, kernel2):
        # Naud et al 2011 equation 2.15
        N = self.ntrains

        dot_product_matrix = self.dot_product_matrix(self, kernel1, kernel2)
        mean_dot_ij = 2. / (N * (N - 1.)) * np.sum(dot_product_matrix[np.triu_indices(N, k=1)])

        L = np.mean(np.diagonal(dot_product_matrix))  # average of the trials norms squared 1/nsweeps*(sum(norm**2. ) )

        return mean_dot_ij / L

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

    def signal_before_spikes(self, signal, tl, tr, t_ref=None, average=True):

        argl = get_arg(tl, self.dt)
        argr = get_arg(tr, self.dt)

        signal = np.concatenate((np.zeros((argl, ) + signal.shape[1:]) * np.nan,
                                 signal,
                                 np.zeros((argr,) + signal.shape[1:]) * np.nan), axis=0)
        mask_spk = np.concatenate((np.zeros((argl, ) + self.mask.shape[1:], dtype=bool),
                                   self.mask,
                                   np.zeros((argr,) + self.mask.shape[1:], dtype=bool)), axis=0)
        if t_ref is not None:
            arg_ref = get_arg(t_ref, self.dt)
            # mask_around_spk = extend_trues(shift_mask(mask_spk, 1, fill_value=False), 0, arg_ref) # doesnt do what I wanted
            # signal[mask_around_spk] = np.nan
            mask_spk = mask_spk & ~extend_trues(shift_mask(mask_spk, 1, fill_value=False), 0, arg_ref)

        index = np.where(mask_spk)
        sta = np.stack([signal[i_spk - argl:i_spk + argr + 1, sw] for i_spk, sw in zip(*index)], 1)

        if average:
            sta = np.nanmean(sta, 1)

        t_sta = np.arange(-argl, argr + 1, 1) * self.dt

        return t_sta, sta

    def sliding_fano_factor(self, kernel):

        conv = self.convolve(kernel)
        mean, var = np.mean(conv, 1), np.var(conv, 1, ddof=1)
        fano = np.ones(len(conv))
        mask = mean > 0
        fano[mask] = var[mask] / mean[mask]

        return fano

    def spike_count(self, tbins, normed=False):
        arg_bins = searchsorted(self.t, tbins)
        if not normed:
            spk_count = np.stack([np.sum(self.mask[arg0:argf], 0) for arg0, argf in zip(arg_bins[:-1], arg_bins[1:])], 0)
        else:
            spk_count = np.stack([np.sum(self.mask[arg0:argf], 0) / (argf - arg0) for arg0, argf in zip(arg_bins[:-1], arg_bins[1:])], 0)
        return spk_count

    def spike_count2(self, bins, average_sweeps=False):
        # 08/09/2018
        # Given arbitrary time bins computes the spike counts in those bins for each sweep
        # unless average=True

        spk_count = np.zeros((len(bins) - 1, self.ntrains))

        mask_spk = self.mask

        for sw in range(self.ntrains):

            t_spk = self.t[mask_spk[:, sw]]

            spk_count[:,sw], _ = np.histogram(t_spk, bins=bins)

        if average_sweeps:
            spk_count = np.mean(spk_count, 1)
            if spk_count.size==1:
                return spk_count[0]
            else:
                return spk_count
        if len(bins) == 2:
            return spk_count
        else:
            return np.squeeze(spk_count)

    def subsample(self, dt):

        n_sample = get_arg(dt, self.dt)
        t = self.t[::n_sample]
        arg_spikes = np.where(self.mask)
        arg_spikes = (np.array(np.floor(arg_spikes[0] / n_sample), dtype=int), ) + arg_spikes[1:]
        mask_spikes = np.zeros((len(t), ) + self.mask.shape[1:], dtype=bool)
        mask_spikes[arg_spikes] = True

        return SpikeTrains(t, mask_spikes)

    def sweeps(self, idx):
        return SpikeTrains(self.t, self.mask[:, idx])

    def t_spikes(self):
        args = np.where(self.mask_)
        t_spk = (self.t[args[0]],) + args[1:]
        return t_spk

    def var(self, kernel1, kernel2):
        return self.ntrains / (self.ntrains - 1) * (
                self.average_squared_norm(kernel1, kernel2) - self.population_norm(kernel1, kernel2, biased=True) ** 2)
    