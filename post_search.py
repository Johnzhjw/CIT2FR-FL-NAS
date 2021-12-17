import os
import json
import argparse
import numpy as np
from pymoo.factory import get_decomposition
from pymoo.visualization.scatter import Scatter
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.model.decision_making import DecisionMaking, normalize, find_outliers_upper_tail, NeighborFinder

_DEBUG = True


class HighTradeoffPoints(DecisionMaking):

    def __init__(self, epsilon=0.125, n_survive=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.n_survive = n_survive  # number of points to be selected

    def _do(self, F, **kwargs):
        n, m = F.shape

        if self.normalize:
            F = normalize(F, self.ideal_point, self.nadir_point, estimate_bounds_if_none=True)

        neighbors_finder = NeighborFinder(F, epsilon=0.125, n_min_neigbors="auto", consider_2d=False)

        mu = np.full(n, - np.inf)

        # for each solution in the set calculate the least amount of improvement per unit deterioration
        for i in range(n):
            # for each neighbour in a specific radius of that solution
            neighbors = neighbors_finder.find(i)

            # calculate the trade-off to all neighbours
            diff = F[neighbors] - F[i]

            # calculate sacrifice and gain
            sacrifice = np.maximum(0, diff).sum(axis=1)
            gain = np.maximum(0, -diff).sum(axis=1)

            np.warnings.filterwarnings('ignore')
            tradeoff = sacrifice / gain

            # otherwise find the one with the smalled one
            mu[i] = np.nanmin(tradeoff)
        if self.n_survive is not None:
            return np.argsort(mu)[-self.n_survive:]
        else:
            return find_outliers_upper_tail(mu)  # return points with trade-off > 2*sigma


def main(args):
    # preferences
    if args.prefer is not None:
        preferences = {}
        for p in args.prefer.split("+"):
            k, v = p.split("#")
            if k == 'top1':
                preferences[k] = 100 - float(v)  # assuming top-1 accuracy
            else:
                preferences[k] = float(v)
        weights = np.fromiter(preferences.values(), dtype=float)

    archive = json.load(open(args.expr))['archive']
    subnets, top1, sec_obj = [v[0] for v in archive], [v[1] for v in archive], [v[2] for v in archive]
    sort_idx = np.argsort(top1)
    F = np.column_stack((top1, sec_obj))[sort_idx, :]
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    pf = F[front, :]
    ps = np.array(subnets)[sort_idx][front]

    if args.prefer is not None:
        # choose the architectures thats closest to the preferences
        I = get_decomposition("asf").do(pf, weights).argsort()[:args.n]
    else:
        # choose the architectures with highest trade-off
        dm = HighTradeoffPoints(n_survive=args.n)
        I = dm.do(pf)

    # always add most accurate architectures
    I = np.append(I, 0)
    I = np.array([int(_) for _ in range(len(pf))], dtype=np.int64)
    I = I[pf.reshape(-1, 2)[:, 0] <= preferences['top1']]
    if len(I) == 0:
        I = np.append(I, 0)

    # create the supernet
    n_channel_in = 3
    n_classes = 1000
    if 'CerebralInfarction.csv' in args.data:
        n_channel_in = 60
        n_classes = 2
    elif 'ALF_Data.csv' in args.data:
        n_channel_in = 28
        n_classes = 2
    elif 'LC25000' in args.data:
        n_channel_in = 3
        n_classes = 5
    from evaluator import OFAEvaluator
    supernet = OFAEvaluator(n_channel_in=n_channel_in, n_classes=n_classes, model_path=args.supernet_path,
                            flag_not_image=args.flag_not_image)
    supernet_FR = OFAEvaluator(n_channel_in=n_channel_in, n_classes=n_classes, model_path=args.supernet_path,
                               flag_not_image=args.flag_not_image, flag_fuzzy=True)

    for idx in I:
        save = os.path.join(args.save, "net-flops@{:.0f}".format(pf[idx, 1]))
        os.makedirs(save, exist_ok=True)
        if not ps[idx]['f']:
            subnet, _ = supernet.sample({'ks': ps[idx]['ks'], 'e': ps[idx]['e'], 'd': ps[idx]['d'], 'f': ps[idx]['f']})
        else:
            subnet, _ = supernet_FR.sample(
                {'ks': ps[idx]['ks'], 'e': ps[idx]['e'], 'd': ps[idx]['d'], 'f': ps[idx]['f']})
        with open(os.path.join(save, "net.subnet"), 'w') as handle:
            json.dump(ps[idx], handle)
        with open(os.path.join(save, "net.few_stats"), 'w') as handle:
            json.dump({'top1': 100 - pf[idx][0], 'f2': pf[idx][1]}, handle)
        supernet.save_net_config(save, subnet, "net.config")
        supernet.save_net(save, subnet, "net.inherited")

    if _DEBUG:
        print('The whole nondominated pf:', pf)
        print('The selected nondominated pf:', pf[I])
        print(ps[I])
        plot = Scatter()
        plot.add(pf, alpha=0.2)
        plot.add(pf[I, :], color="red", s=100)
        plot.show()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=str, default='.tmp99_evo00',
                        help='location of dir to save')
    parser.add_argument('--expr', type=str, default='.save_test_evo00/perf_test.stats',
                        help='location of search experiment dir')
    parser.add_argument('--prefer', type=str, default='top1#99+flops#150',
                        help='preferences in choosing architectures (top1#80+flops#150)')
    parser.add_argument('-n', type=int, default=1,
                        help='number of architectures desired')
    parser.add_argument('--supernet_path', type=str, default='./model_1_cur_FL_w1.0',
                        help='file path to supernet weights')
    parser.add_argument('--data', type=str, default='../data/LC25000',
                        help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='LC25000',
                        help='name of the dataset (imagenet, cifar10, cifar100, ...)')
    parser.add_argument('--flag_not_image', action='store_true', default=False,
                        help='The inputs are not images')

    cfgs = parser.parse_args()
    print(cfgs)

    main(cfgs)
