import numpy as np
from scipy.spatial.distance import cdist

from pymoo.algorithms.genetic_algorithm import GeneticAlgorithm
# from pymoo.factory import get_decomposition
from pymoo.factory import get_from_list
from pymoo.operators.crossover.simulated_binary_crossover import SimulatedBinaryCrossover
from pymoo.operators.mutation.polynomial_mutation import PolynomialMutation
from pymoo.operators.sampling.random_sampling import FloatRandomSampling
from pymoo.util.display import MultiObjectiveDisplay
from pymoo.util.misc import set_if_none

import autograd.numpy as anp

from pymoo.model.decomposition import Decomposition


class Tchebicheff_ed(Decomposition):

    def _do(self, F, weights, **kwargs):
        # v = anp.abs(F - self.utopian_point) * weights
        v = anp.abs(F - self.ideal_point) / (anp.abs(self.nadir_point - self.ideal_point) + 1e-5) / (weights + 1e-5)
        tchebi = v.max(axis=1)
        return tchebi


# ---------------------------------------------------------------------------------------------------------
# Binary Tournament Selection Function
# ---------------------------------------------------------------------------------------------------------
from pymoo.operators.selection.tournament_selection import compare
import math
from pymoo.util.misc import random_permuations


def tournament_selection(utility, n_select, n_parents=2, pressure=2):
    # number of random individuals needed
    n_random = n_select * n_parents * pressure

    # number of permutations needed
    n_perms = math.ceil(n_random / len(utility))

    # get random permutations and reshape them
    P = random_permuations(n_perms, len(utility))[:n_random]
    P = np.reshape(P, (n_select * n_parents, pressure))

    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):

        a, b = P[i, 0], P[i, 1]

        S[i] = compare(a, utility[a], b, utility[b],
                       method='larger_is_better')

        # if rank or domination relation didn't make a decision compare by crowding
        if np.isnan(S[i]):
            S[i] = P[i, np.random.choice(pressure)]

    return S[:, None].astype(np.int, copy=False)


# =========================================================================================================
# DECOMPOSITION
# =========================================================================================================

def get_decomposition_options():
    from pymoo.decomposition.pbi import PBI
    from pymoo.decomposition.tchebicheff import Tchebicheff
    from pymoo.decomposition.weighted_sum import WeightedSum
    from pymoo.decomposition.asf import ASF
    from pymoo.decomposition.aasf import AASF
    from pymoo.decomposition.perp_dist import PerpendicularDistance

    DECOMPOSITION = [
        ("weighted-sum", WeightedSum),
        ("tchebi", Tchebicheff),
        ("tchebi_ed", Tchebicheff_ed),
        ("pbi", PBI),
        ("asf", ASF),
        ("aasf", AASF),
        ("perp_dist", PerpendicularDistance)
    ]

    return DECOMPOSITION


def get_decomposition(name, *args, d={}, **kwargs):
    return get_from_list(get_decomposition_options(), name, args, {**d, **kwargs})


# =========================================================================================================
# Implementation
# =========================================================================================================

class MOEAD(GeneticAlgorithm):

    def __init__(self,
                 ref_dirs,
                 n_neighbors=20,
                 decomposition='auto',
                 prob_neighbor_mating=0.9,
                 display=MultiObjectiveDisplay(),
                 init_scheme=0,
                 type_select='random',
                 n_offsprings=20,
                 utility_ini=0.1,
                 utility_type='descend',
                 **kwargs):
        """

        Parameters
        ----------
        ref_dirs
        n_neighbors
        decomposition
        prob_neighbor_mating
        display
        kwargs
        """

        self.n_neighbors = n_neighbors
        self.prob_neighbor_mating = prob_neighbor_mating
        self.decomposition = decomposition
        self.init_scheme = init_scheme
        self.type_select = type_select
        self.n_offspr = n_offsprings
        self.utility_ini = utility_ini
        self.utility_type = utility_type

        set_if_none(kwargs, 'pop_size', len(ref_dirs))
        set_if_none(kwargs, 'sampling', FloatRandomSampling())
        set_if_none(kwargs, 'crossover', SimulatedBinaryCrossover(prob=1.0, eta=20))
        set_if_none(kwargs, 'mutation', PolynomialMutation(prob=None, eta=20))
        set_if_none(kwargs, 'survival', None)
        set_if_none(kwargs, 'selection', None)

        super().__init__(display=display, **kwargs)

        # initialized when problem is known
        if isinstance(self.decomposition, Tchebicheff_ed) or self.decomposition == "tchebi_ed":
            self.ref_dirs = ref_dirs[ref_dirs[:, 0].argsort(), :]
        else:
            self.ref_dirs = ref_dirs[ref_dirs[:, 1].argsort(), :]

        if self.ref_dirs.shape[0] < self.n_neighbors or self.n_neighbors <= 0:
            print("Setting number of neighbours to population size: %s" % self.ref_dirs.shape[0])
            self.n_neighbors = self.ref_dirs.shape[0]

        if self.ref_dirs.shape[0] < self.n_offspr or self.n_offspr <= 0:
            print("Setting number of offsprings to population size: %s" % self.ref_dirs.shape[0])
            self.n_offspr = self.ref_dirs.shape[0]

        # neighbours includes the entry by itself intentionally for the survival method
        self.neighbors = np.argsort(cdist(self.ref_dirs, self.ref_dirs), axis=1, kind='quicksort')[:, :self.n_neighbors]

        self.n_max_rep = 2

    def _initialize(self):

        if isinstance(self.decomposition, str):

            # set a string
            decomp = self.decomposition

            # for one or two objectives use tchebi otherwise pbi
            if decomp == 'auto':
                if self.problem.n_obj <= 2:
                    decomp = 'tchebi'
                else:
                    decomp = 'pbi'

            # set the decomposition object
            self._decomposition = get_decomposition(decomp)

        else:
            self._decomposition = self.decomposition

        super()._initialize()
        self.ideal_point = np.min(self.pop.get("F"), axis=0)
        self.nadir_point = np.max(self.pop.get("F"), axis=0)

        #
        if isinstance(self._decomposition, Tchebicheff_ed):
            inds = self.pop.get("F")[:, 0].argsort()
        else:
            inds = self.pop.get("F")[:, 1].argsort()
        if self.init_scheme == 1:
            self.pop.set("X", self.pop.get("X")[inds])
            self.pop.set("F", self.pop.get("F")[inds])
        elif self.init_scheme == 2:
            f = self.pop.get("F")
            inds_cur = []
            for i in range(len(inds)):
                if isinstance(self._decomposition, Tchebicheff_ed):
                    _ = i
                else:
                    _ = len(self.pop) - i - 1
                inds_rem = [i for i in inds.tolist() if i not in inds_cur]
                if len(inds_rem) > 1:
                    FVs = self._decomposition.do(f[inds_rem, :], weights=self.ref_dirs[_][None, :],
                                                 ideal_point=self.ideal_point, nadir_point=self.nadir_point)
                    # print(FVs)
                    # print(FVs.argsort())
                    # print(inds_rem[FVs.argsort()[0]])
                    if isinstance(self._decomposition, Tchebicheff_ed):
                        inds_cur.append(inds_rem[FVs.argsort()[0]])
                    else:
                        inds_cur.insert(0, inds_rem[FVs.argsort()[0]])
                else:
                    if isinstance(self._decomposition, Tchebicheff_ed):
                        inds_cur.append(inds_rem[0])
                    else:
                        inds_cur.insert(0, inds_rem[0])

            self.pop.set("X", self.pop.get("X")[inds_cur])
            self.pop.set("F", self.pop.get("F")[inds_cur])

        if self.utility_type == 'descend':
            v_bg = self.utility_ini
            v_fn = self.utility_ini / 10
            v_gp = (v_fn - v_bg) / (len(self.pop) - 1)
            self.utility = np.full(len(self.pop), self.utility_ini)
            for _ in range(len(self.pop)):
                self.utility[_] = v_bg + _ * v_gp
        elif self.utility_type == 'ascend':
            v_bg = self.utility_ini / 10
            v_fn = self.utility_ini
            v_gp = (v_fn - v_bg) / (len(self.pop) - 1)
            self.utility = np.full(len(self.pop), self.utility_ini)
            for _ in range(len(self.pop)):
                self.utility[_] = v_bg + _ * v_gp
        elif self.utility_type == 'uniform':
            self.utility = np.full(len(self.pop), self.utility_ini)

        self.a_util = 0.99

    def _next(self):
        repair, crossover, mutation = self.mating.repair, self.mating.crossover, self.mating.mutation

        # retrieve the current population
        pop = self.pop

        # iterate for each member of the population in random order
        if self.type_select == 'random':
            inds = np.random.permutation(len(pop))[:self.n_offspr]
        else:
            print(self.utility)
            inds = np.random.choice([_ for _ in range(self.pop_size)], self.n_offspr,
                                    replace=False,
                                    p=self.utility / (np.sum(self.utility)))
        if isinstance(self._decomposition, Tchebicheff_ed):
            tar_ind = 0
        else:
            tar_ind = self.pop_size - 1
        if tar_ind not in inds:
            inds[np.random.choice(self.n_offspr)] = tar_ind
        for i in inds:
            # all neighbors of this individual and corresponding weights
            if np.random.random() < self.prob_neighbor_mating:
                flag_neighbor = True
                N = self.neighbors[i, :]
                parents = N[np.random.permutation(self.n_neighbors)][:crossover.n_parents]
            else:
                flag_neighbor = False
                N = np.array([int(_) for _ in range(self.pop_size)], dtype=np.int64)
                parents = np.random.permutation(self.pop_size)[:crossover.n_parents]

            # do recombination and create an offspring
            off = crossover.do(self.problem, pop, parents[None, :])
            off = mutation.do(self.problem, off)
            off = off[np.random.randint(0, len(off))]

            # repair first in case it is necessary
            if repair:
                off = self.repair.do(self.problem, off, algorithm=self)

            # evaluate the offspring
            self.evaluator.eval(self.problem, off)

            # update the ideal point
            self.ideal_point = np.min(np.vstack([self.ideal_point, off.F]), axis=0)

            # calculate the decomposed values for each neighbor
            FV = self._decomposition.do(pop[N].get("F"), weights=self.ref_dirs[N, :],
                                        ideal_point=self.ideal_point, nadir_point=self.nadir_point)
            off_FV = self._decomposition.do(off.F[None, :], weights=self.ref_dirs[N, :],
                                            ideal_point=self.ideal_point, nadir_point=self.nadir_point)

            # get the absolute index in F where offspring is better than the current F (decomposed space)
            FV_diff = off_FV - FV
            inds_neg = FV_diff < 0
            I = FV_diff[inds_neg].argsort()[:self.n_max_rep]
            # I = np.where(off_FV < FV)[0]
            # print(inds_neg)
            # print(I)
            I = N[inds_neg][I]
            if self.type_select != 'random' and len(I) > 0:
                tmp = (pop[I].get("F")[:, 0] - off.get("F")[0]) / (self.nadir_point[0] - self.ideal_point[0] + 1e-5)
                inds_pos = tmp > 0
                uti = np.sum(tmp[inds_pos])
                tmp_n = np.sum(inds_pos)
                if tmp_n > 0:
                    uti /= tmp_n
                self.utility[i] = uti + self.a_util * self.utility[i]
            pop[I] = off

        self.nadir_point = np.max(np.vstack([self.nadir_point, off.F]), axis=0)

# parse_doc_string(MOEAD.__init__)
