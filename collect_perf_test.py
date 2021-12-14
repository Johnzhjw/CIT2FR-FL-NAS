import os
import json
import time
import argparse


def _resume_from_dir(read_path, sec_obj='flops'):
    """ resume search from a previous iteration """
    import glob

    archive = []
    for file in glob.glob(os.path.join(read_path, "net_*.subnet")):
        arch = json.load(open(file))
        pre, ext = os.path.splitext(file)
        stats = json.load(open(pre + ".stats"))
        archive.append((arch, 100 - stats['top1'][1], stats[sec_obj]))

    return archive


def main(cfgs):
    # read
    archive = _resume_from_dir(cfgs.read_path, cfgs.sec_obj)

    # dump the statistics
    os.makedirs(cfgs.save_path, exist_ok=True)
    with open(os.path.join(cfgs.save_path, "perf_test.stats"), "w") as handle:
        json.dump({'archive': archive}, handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='.save_test_evo00',
                        help='location of dir to save')
    parser.add_argument('--str_time', type=str, default=None,
                        help='time string')
    parser.add_argument('--read_path', type=str, default='../evo00/search-search-FL-20211211-221358/*',
                        help='read from where')
    parser.add_argument('--sec_obj', type=str, default='flops',
                        help='second objective to optimize simultaneously')

    cfgs = parser.parse_args()
    cfgs.str_time = time.strftime("%Y%m%d-%H%M%S")
    # cfgs.save_path = '{}-{}'.format(cfgs.save_path, cfgs.str_time)
    print(cfgs)

    main(cfgs)
