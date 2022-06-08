import numpy as np
import matplotlib.pyplot as plt
import json


def show_num_by_freq_threshold(in_path):
    atts = json.load(open(in_path, 'r'))['atts']
    atts_freq = [x[1] for x in atts]
    atts_freq = np.array(atts_freq)
    max_value = np.max(atts_freq) * 1
    # xs = np.linspace(100, max_value, num=100)  # the label locations
    xs = np.linspace(1000, 5e4, num=100)  # the label locations

    ys = []
    for x in xs:
        num_atts = np.sum(atts_freq > x)
        ys.append(num_atts)
    ys = np.array(ys)

    fig, ax = plt.subplots(figsize=(15, 10))
    rects1 = ax.plot(xs, ys, label='num atts', linewidth=2, color='red')
    plt.margins(x=0)
    ax.set_xlabel('att freqency')
    ax.set_ylabel('num atts')
    ax.set_title('num atts by freq thres')
    # ax.set_xticks(x, labels, rotation=90)
    ax.legend()
    ax.grid()
    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    fig.tight_layout()
    # plt.show()
    # plt.savefig("atts_cut_scale_1.png", bbox_inches='tight', pad_inches=0)
    plt.savefig("vis_freq/atts_between_1000_5e4.png", bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    in_path = 'infos/all_gather_atts_filtered.json'
    show_num_by_freq_threshold(in_path)