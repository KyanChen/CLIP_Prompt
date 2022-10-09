import argparse
import os

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn.functional import cosine_embedding_loss, l1_loss, mse_loss
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch.utils import data
import matplotlib.pyplot as plt
from torch import distributed as dist


def VEL(src):  # bs seq ch
    return src[:, 1:, :] - src[:, :-1, :]


def ACC(src):  # bs seq ch
    return VEL(VEL(src))


def fft(src):
    return src.permute(0, 2, 1).reshape(-1, src.shape[1]).rfft(signal_ndim=1)


# def train(hp, logger):
#     """
#     train network
#     """
#     trainset = DancerDataset(root=hp.data_root, phase="train", clip_len=hp.clip_len, is_global=hp.is_global)
#     train_loader = data.DataLoader(trainset, batch_size=hp.batch_size, shuffle=True, drop_last=True)
#
#     valset = DancerDataset(root=hp.data_root, phase="val", clip_len=hp.clip_len, is_global=hp.is_global)
#     val_loader = data.DataLoader(valset, batch_size=hp.batch_size, shuffle=True, drop_last=False)
#
#     # global model
#     if hp.is_global:
#         poss_num = hp.points_num * 3
#         rots_num = hp.points_num * 4
#         if hp.is_pos_only:
#             rots_num = 0
#     else:
#         poss_num = 3
#         rots_num = hp.points_num * 4
#
#     model = Blender(mode=hp.mode, is_global=hp.is_global, clip_len=hp.clip_len,
#                     is_conditional=hp.is_conditional, poss_num=poss_num, rots_num=rots_num,
#                     feature_size=hp.feature_size, vocab=trainset.vocab, pos_ebd=hp.pos_ebd, cls_ebd=hp.cls_ebd,
#                     inout_net=hp.inout_net).cuda()
#
#     if len(hp.trained_model) > 0:
#         model.load_state_dict(torch.load(hp.trained_model)["model_state"])
#         logger.info("loading {} ...".format(hp.trained_model))
#
#     # ============= convert =============== #
#     parents = trainset.parents
#     offsets = torch.FloatTensor(trainset.offsets).cuda()
#     x_mean = torch.FloatTensor(trainset.x_mean).view(1, 1, -1, 3).cuda()
#     x_std = torch.FloatTensor(trainset.x_std).view(1, 1, -1, 3).cuda()
#     root_mean = x_mean[0, :, :1, :]
#     root_std = x_std[0, :, :1, :]
#
#     def split_local(data, id=None, is_tensor=False):  # batch=1
#         R = data[:, :, :3]
#         if hp.is_normalize:
#             R = R * root_std.expand_as(R) + root_mean.expand_as(R)
#         else:
#             R = R / hp.poss_scale
#         X = torch.cat([R.view(R.shape[0], R.shape[1], 1, 3), offsets.expand(R.shape[0], R.shape[1], -1, -1)], dim=2).cpu()
#         Q = data[:, :, 3:]
#         Q = Q.view(Q.shape[0], Q.shape[1], -1, 4).cpu()
#         if is_tensor:
#             return X, Q
#         if id is not None:
#             return X[id].numpy(), Q[id].numpy()
#         else:
#             return X.numpy(), Q.numpy()
#
#     def split_global(data, id=None, is_tensor=False):  # batch=1
#         GP = data[:, :, :poss_num]
#         GP = GP.view(GP.shape[0], GP.shape[1], -1, 3)
#         if hp.is_normalize:
#             GP = GP * x_std.expand_as(GP) + x_mean.expand_as(GP)
#         else:
#             GP = GP / hp.poss_scale
#         GQ = data[:, :, poss_num:]
#         GQ = GQ.view(GQ.shape[0], GQ.shape[1], -1, 4)
#         if is_tensor:
#             return GP, GQ
#         if id is not None:
#             return GP[id].cpu().numpy(), GQ[id].cpu().numpy()
#         else:
#             return GP.cpu().numpy(), GQ.cpu().numpy()
#
#     def split(data, id=None, is_tensor=False):
#         if hp.is_global:
#             return split_global(data, id, is_tensor)
#         else:
#             return split_local(data, id, is_tensor)
#     def re_scale_p(P):
#         if hp.is_normalize:
#             if hp.is_global:
#                 P = (P - x_mean.expand_as(P)) / x_std.expand_as(P)
#             else:
#                 P = (P - root_mean.expand_as(P)) / root_std.expand_as(P)
#         else:
#             P = P * hp.poss_scale
#         return P
#     # ============= convert =============== #
#     if hp.weight_decay > 0:
#         optim = torch.optim.AdamW(model.parameters(), lr=hp.l_rate, weight_decay=hp.weight_decay, amsgrad=True)
#     else:
#         optim = torch.optim.Adam(model.parameters(), lr=hp.l_rate)
#     # optim = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.9)
#     step_lr = StepLR(optim, step_size=hp.step_size, gamma=hp.gamma)
#     warm_up = GradualWarmupScheduler(optim, multiplier=1, total_epoch=50,
#                                      after_scheduler=step_lr, lower_lr=hp.l_rate/100)
#
#     logger.info(model)
#     logger.info(optim)
#     logger.info(step_lr)
#
#     l1_loss = nn.L1Loss()
#     l2_loss = nn.MSELoss()
#     core_loss = l1_loss  # mrse2
#     ce_loss = nn.CrossEntropyLoss()
#     if hp.is_FK_loss:
#         fk_loss = FKloss(parents, x_mean, x_std, offsets).cuda() # TODO for scale
#     if hp.is_T_loss:
#         t_loss = Tloss(parents, offsets)
#         ik_loss = IKloss(parents, offsets)
#     batch_num = len(train_loader)
#     logger.info("Start training ...")
#     for epoch in range(hp.n_epoch):
#         if hp.m_train:
#             model.train()
#             losses = []
#
#             for i, (bs_data) in enumerate(train_loader):
#                 if hp.is_global:
#                     P, Q, C = bs_data
#                     P, Q, C = P.cuda(), Q.cuda(), C.cuda()
#                 else:
#                     P, Q, GP, GQ, C = bs_data
#                     P, Q, GP, GQ, C = P.cuda(), Q.cuda(), GP.cuda(), GQ.cuda(), C.cuda()
#                 P = re_scale_p(P)
#                 P = P.view(P.shape[0], P.shape[1], -1)
#                 Q = Q.view(Q.shape[0], Q.shape[1], -1)
#
#                 motions = torch.cat([P, Q], dim=2)
#
#                 optim.zero_grad()
#                 out_motion, mask, interp = model(C, motions)
#                 if hp.is_mask_only:
#                     mask_training =  (mask <= 1)
#                 else:
#                     mask_training =  (mask >= 0)
#                 loss_pos = core_loss(out_motion[:, mask_training, :poss_num], motions[:,mask_training, :poss_num])
#                 loss_quat = core_loss(out_motion[:, mask_training, poss_num:], motions[:,mask_training, poss_num:])
#                 # loss_cls = torch.FloatTensor([0])  # nn.functional.cross_entropy(out_class, C)
#                 loss_overall = loss_quat + hp.pos_alpha * loss_pos  # + 0 * loss_cls
#                 if hp.is_FK_loss:
#                     loss_fk = fk_loss(out_motion[:, mask_training, :], GP[:, mask_training, :])
#                     loss_overall += 0.1 * loss_fk
#                 if hp.is_T_loss:
#                     # loss_tp = t_loss(split(out_motion[:, mask_training, :], is_tensor=True)[0])
#                     loss_tp = ik_loss(*split(out_motion[:, mask_training, :], is_tensor=True))
#                     loss_overall += 0.01 * loss_tp
#                 loss_overall.backward()
#                 # total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, 'inf')
#                 optim.step()
#
#                 loss_info = [loss_pos.item(), loss_quat.item(), loss_overall.item()]
#                 losses.append(loss_info)
#
#                 if (i + 1) % hp.train_interval == 0:
#                     info = "loss_pos: {:.4f} loss_quat: {:.4f} loss_overall: {:.4f}".format(
#                         *loss_info
#                     )
#                     if hp.is_FK_loss:
#                         info += " loss_fk: {:.4f}".format(loss_fk.item())
#                     if hp.is_T_loss:
#                         info += " loss_tp: {:.4f}".format(loss_tp.item())
#                     # info += " total_norm: {:.4f}".format(total_norm.item())
#                     logger.info("Epoch [{}/{}][{}/{}][lr: {:.6f}] {}".format(
#                                 epoch + 1, hp.n_epoch, i, batch_num,
#                                 optim.param_groups[0]['lr'], info))
#                 # break
#
#         if hp.m_eval:
#             model.eval()
#             losses_val = []
#             if hp.mode == "inbetween":
#                 mask_start = n_past = 10
#                 n_future = 10
#                 mask_len = n_trans = 30
#             elif hp.mode == "blend":
#                 mask_start = n_past = 16
#                 n_future = 16
#                 mask_len = n_trans = 32
#             else:
#                 raise "ERROR IN val len"
#             bm = benchmark(hp.is_global, hp.points_num, trainset.x_mean, trainset.x_std, trainset.offsets, trainset.parents,
#                            n_past=n_past, n_future=n_future, n_trans=n_trans)
#             for i, (bs_data_val) in enumerate(val_loader):
#                 if hp.is_global:
#                     P_val, Q_val, C_val = bs_data_val
#                     P_val, Q_val, C_val = P_val.cuda(), Q_val.cuda(), C_val.cuda()
#                 else:
#                     P_val, Q_val, GP_val, GQ_val, C_val = bs_data_val
#                     P_val, Q_val, GP_val, GQ_val, C_val = P_val.cuda(), Q_val.cuda(), GP_val.cuda(), GQ_val.cuda(), C_val.cuda()
#
#                 P_val = re_scale_p(P_val)
#                 P_val = P_val.view(P_val.shape[0], P_val.shape[1], -1)
#                 Q_val = Q_val.view(Q_val.shape[0], Q_val.shape[1], -1)
#
#                 motions_val = torch.cat([P_val, Q_val], dim=2)
#
#                 with torch.no_grad():
#                     out_val, mask_val, interp_val = model(C_val, motions_val, mask_len=n_trans, mask_start=n_past)
#
#                     loss_mask_root_val = core_loss(out_val[:, mask_val == 1, :poss_num], motions_val[:, mask_val == 1, :poss_num])
#                     loss_mask_quat_val = core_loss(out_val[:, mask_val == 1, poss_num:], motions_val[:, mask_val == 1, poss_num:])
#
#                     loss_baseline_root = core_loss(interp_val[:, mask_val == 1, :poss_num], motions_val[:, mask_val == 1, :poss_num])
#                     loss_baseline_quat = core_loss(interp_val[:, mask_val == 1, poss_num:], motions_val[:, mask_val == 1, poss_num:])
#
#                 bm.collect(split(motions_val), split(out_val))
#                 losses_val.append([loss_mask_root_val.item(), loss_mask_quat_val.item(),
#                                    loss_baseline_root.item(), loss_baseline_quat.item(),
#                                 ])
#                 # break
#             score, info = bm.get_score()
#             logger.info(info)
#
#             if (epoch + 1) % 10 == 0:
#                 att = []
#                 for i, block in enumerate(model.translator.blocks):
#                     att.append(block.attn.scores[0, 0, :, :].detach().cpu().numpy())
#                 cmap = plt.get_cmap('jet')
#                 path = hp.vis_path + "vis_{}_val_{:06d}.png".format(hp.mode, epoch + 1)
#                 plt.imsave(path, cmap(np.hstack(att)))
#
#         if hp.visdom:
#             progress = epoch # + i/batch_num
#             loss_mean_train = torch.FloatTensor(losses).mean(dim=0).numpy().tolist()
#             loss_mean_val = torch.FloatTensor(losses_val).mean(dim=0).numpy().tolist()
#             hp.update_loss(progress, loss_mean_train + loss_mean_val)
#             hp.update_score(progress, list(score.values()))
#
#             with torch.no_grad():
#                 cvs_train = plot_trace(out_motion[:, :, :3], interp[:, :, :3], P[:, :, :3])
#                 cvs_val = plot_trace(out_val[:, :, :3], interp_val[:, :, :3], P_val[:, :, :3])
#                 cvs = np.hstack([cvs_train, cvs_val])
#             hp.update_trace(cvs.transpose([2, 0, 1]))
#
#         """
#         if (epoch + 1) % hp.vis_interval == 0 or epoch == 0:
#             with torch.no_grad():
#                 draw_bvh_pair(
#                     [split_global(out_motion, 0)[0], None],
#                     [split_global(interp, 0)[0], None],
#                     [split_global(motions, 0)[0], None],
#                     hp.vis_path + "vis_{}_train_{:06d}.gif".format(hp.mode, epoch + 1),
#                     mask=mask.detach().cpu().numpy().tolist(),
#                     parents=parents,
#                     duration=1 / 30
#                 )
#                 draw_bvh_pair(
#                     [split_global(out_val, 0)[0], None],
#                     [split_global(interp_val, 0)[0], None],
#                     [split_global(motions_val, 0)[0], None],
#                     hp.vis_path + "vis_{}_val_{:06d}.gif".format(hp.mode, epoch + 1),
#                     mask=mask_val.detach().cpu().numpy().tolist(),
#                     parents=parents,
#                     duration=1 / 30
#                 )
#         """ # TODO
#         warm_up.step()
#
#         if True and ((epoch + 1) % hp.snapshot == 0):
#             state = {
#                 'epoch': epoch + 1,
#                 'model_state': model.state_dict(),
#                 'optim_state': optim.state_dict(),
#             }
#             model_name = "{}_{}_{}.pkl".format(hp.name, hp.mode, epoch + 1)
#             logger.info("save model to %s" % (model_name))
#             torch.save(state, hp.weight_path + model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    
    distributed = True
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend='nccl')
    print('loading model!')
    
    # # init hypers
    # hp = Hyperparameters()
    # # logger
    # logger = hp.get_logger()
    # # start
    # train(hp, logger)
