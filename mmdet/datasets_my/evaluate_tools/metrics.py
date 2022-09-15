import json
import os

import torch

from .evaluator import Evaluator
from .ovad_evaluator import AttEvaluator, print_metric_table


def cal_metrics(
        attr2idx,
        dataset_name,
        prefix_path,
        pred,
        gt_label,
        is_logit=True,
        use_vaw=False,
        top_k=8,
        save_result=False
):
    if not use_vaw:
        attr_type = json.load(open(prefix_path + '/attribute_types.json'))
        attr_parent_type = json.load(open(prefix_path + '/attribute_parent_types.json'))
        attribute_head_tail = json.load(open(prefix_path + '/head_tail.json'))

        evaluator = AttEvaluator(
            attr2idx,
            attr_type=attr_type,
            attr_parent_type=attr_parent_type,
            attr_headtail=attribute_head_tail,
            att_seen_unseen={},
            dataset_name=dataset_name,
            threshold=0.5,
            top_k=top_k,
            exclude_atts=[],
        )
        # Run evaluation
        if is_logit:
            pred = pred.data.cpu().float().sigmoid().numpy()  # Nx620
        else:
            pred = pred.data.cpu().float().numpy()  # Nx620
        gt_label = gt_label.data.cpu().float().numpy()  # Nx620

        if save_result:
            output_file_fun = os.path.join("output", "{}.log".format(dataset_name))
        else:
            output_file_fun = ''

        results = evaluator.print_evaluation(
            pred=pred,
            gt_label=gt_label,
            output_file=output_file_fun,
        )
        # Print results in table
        results = print_metric_table(evaluator, results)
        return results

    if fpath_attribute_index is None:
        fpath_attribute_index = prefix_path + '/attribute_index.json'
    fpath_attribute_types = prefix_path + '/attribute_types.json'
    fpath_attribute_parent_types = prefix_path + '/attribute_parent_types.json'
    fpath_head_tail = prefix_path + '/head_tail.json'

    if is_logit:
        pred = pred.data.cpu().float().sigmoid().numpy()  # Nx620
    else:
        pred = pred.data.cpu().float().numpy()  # Nx620
    gt_label = gt_label.data.cpu().float().numpy()  # Nx620
    evaluator = Evaluator(
        fpath_attribute_index, fpath_attribute_types,
        fpath_attribute_parent_types, fpath_head_tail)
    # Compute scores.
    scores_overall, scores_per_class = evaluator.evaluate(pred, gt_label)

    output = torch.tensor(scores_overall['all']['f1'])
    if return_all:
        scores_overall_topk, scores_per_class_topk = evaluator.evaluate(pred, gt_label, threshold_type='topk')
        output = [scores_overall, scores_per_class, scores_overall_topk, scores_per_class_topk]
    if return_evaluator:
        output += [evaluator]

    return output
    # scores_overall_topk, scores_per_class_topk = evaluator.evaluate(pred, gt_label, threshold_type='topk')
    
    # CATEGORIES = ['all', 'head', 'medium', 'tail'] + \
    #     list(evaluator.attribute_parent_type.keys())

    for category in CATEGORIES:
        print(f"----------{category.upper()}----------")
        print(f"mAP: {scores_per_class[category]['ap']:.4f}")
        
        print("Per-class (threshold 0.5):")
        for metric in ['recall', 'precision', 'f1', 'bacc']:
            if metric in scores_per_class[category]:
                print(f"- {metric}: {scores_per_class[category][metric]:.4f}")
        print("Per-class (top 15):")
        for metric in ['recall', 'precision', 'f1']:
            if metric in scores_per_class_topk[category]:
                print(f"- {metric}: {scores_per_class_topk[category][metric]:.4f}")
    
        print("Overall (threshold 0.5):")
        for metric in ['recall', 'precision', 'f1', 'bacc']:
            if metric in scores_overall[category]:
                print(f"- {metric}: {scores_overall[category][metric]:.4f}")
        print("Overall (top 15):")
        for metric in ['recall', 'precision', 'f1']:
            if metric in scores_overall_topk[category]:
                print(f"- {metric}: {scores_overall_topk[category][metric]:.4f}")

    # with open(output, 'w') as f:
    #     f.write('| {:<18}| AP\t\t| Recall@K\t| B.Accuracy\t| N_Pos\t| N_Neg\t|\n'.format('Name'))
    #     f.write('-----------------------------------------------------------------------------------------------------\n')
    #     for i_class in range(evaluator.n_class):
    #         att = evaluator.idx2attr[i_class]
    #         f.write('| {:<18}| {:.4f}\t| {:.4f}\t| {:.4f}\t\t| {:<6}| {:<6}|\n'.format(
    #             att,
    #             evaluator.get_score_class(i_class).ap,
    #             evaluator.get_score_class(i_class, threshold_type='topk').get_recall(),
    #             evaluator.get_score_class(i_class).get_bacc(),
    #             evaluator.get_score_class(i_class).n_pos,
    #             evaluator.get_score_class(i_class).n_neg))

