import os
import glob

import numpy as np
import matplotlib.pyplot as plt

epochs = [_ for _ in range(1, 201)]

tmp_tag = ['0', '1']
tmp_title = ['Minimum Complexity', 'Maximum Complexity']
tmp_name = ['Min', 'Max']

results = {}

for i in range(len(tmp_tag)):
    fig = plt.figure()
    for file in glob.glob(os.path.join("../test00/.tmp-20211004-214637-"+tmp_tag[i]+"*", "logs", "valid_console.txt")):
        try:
            f = open(file, 'r')
            list1 = f.readlines()
            valid_top1 = [float(line.split('\t')[2].split(' ')[2]) for line in list1]
            train_top1 = [float(line.split('\t')[3].split(' ')[2]) for line in list1]
        finally:
            if f:
                f.close()

        tmp_str = '-' + '-'.join(file.split('\\')[1].split('-')[4:])
        if tmp_str == '-':
            tmp_str = ''

        plt.plot(epochs, train_top1, label='train'+tmp_str, linestyle=':')
        plt.plot(epochs, valid_top1, label='valid'+tmp_str)

        results[tmp_tag[i]+'-train'+tmp_str] = [max(train_top1), min(np.argwhere(train_top1==np.max(train_top1)))[0]]
        results[tmp_tag[i]+'-valid'+tmp_str] = [max(valid_top1), min(np.argwhere(valid_top1==np.max(valid_top1)))[0]]
        assert max(train_top1) == train_top1[results[tmp_tag[i]+'-train'+tmp_str][1]]
        assert max(valid_top1) == valid_top1[results[tmp_tag[i]+'-valid'+tmp_str][1]]

    plt.legend()
    plt.title(tmp_title[i])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # 保存图片到本地
    plt.savefig(tmp_name[i]+'_plot.eps', format='eps', bbox_inches='tight')

print(results)

res_str = ''
for key in results:
    res_str += "%s,%.2f(%d)\n" % (key, results[key][0], results[key][1])

print(res_str)

stop = 1

'''
sum([s['f'] for s in subnets])
nets = [subnets[_] for _ in sort_idx]
len(nets)
F_nets = [nets[_] for _ in front]
sum([s['f'] for s in F_nets])
len([s['f'] for s in F_nets])

import matplotlib.pyplot as plt
fig = plt.figure()
plt.savefig(tmp_name[i]+'_plot.eps', format='eps', bbox_inches='tight')
'''
