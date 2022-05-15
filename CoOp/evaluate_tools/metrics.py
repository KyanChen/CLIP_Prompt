from .evaluator import Evaluator


def cal_metrics(prefix_path, pred, gt_label):
    fpath_attribute_index = prefix_path+'attribute_index.json'
    fpath_attribute_types = prefix_path+'attribute_types.json'
    fpath_attribute_parent_types = prefix_path+'attribute_parent_types.json'
    fpath_head_tail = prefix_path+'head_tail.json'

    import pdb
    pdb.set_trace()
    pred = pred.data.cpu().float().sigmoid().numpy() # Nx620
    gt_label = gt_label.data.cpu().float().numpy() # Nx620
    gt_label[gt_label==-1] = 2
    evaluator = Evaluator(
        fpath_attribute_index, fpath_attribute_types,
        fpath_attribute_parent_types, fpath_head_tail)
    # Compute scores.
    scores_overall, scores_per_class = evaluator.evaluate(pred, gt_label)
    
    return scores_overall['all']['ap']
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