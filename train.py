import collections
import math
import statistics
from functools import reduce
import numpy as np
import torch
import torch.nn as nn
from apex import amp
from torch import distributed
from torch.nn import functional as F
import os
from utils import get_regularizer
from utils.loss import (NCA, BCESigmoid, BCEWithLogitsLossWithIgnoreIndex,
                        ExcludedKnowledgeDistillationLoss, FocalLoss,
                        FocalLossNew, IcarlLoss, KnowledgeDistillationLoss,
                        UnbiasedCrossEntropy,
                        UnbiasedKnowledgeDistillationLoss, UnbiasedNCA,
                        soft_crossentropy)


class Trainer:

    def __init__(self, model, model_old, device, opts, trainer_state=None, classes=None, step=0):
        self.model_old = model_old
        self.model = model
        self.device = device
        self.step = step

        if opts.dataset == "cityscapes_domain":
            self.old_classes = opts.num_classes
            self.nb_classes = opts.num_classes
            self.nb_current_classes = opts.num_classes
            self.nb_new_classes = opts.num_classes
        elif classes is not None:
            new_classes = classes[-1]
            tot_classes = reduce(lambda a, b: a + b, classes)
            self.old_classes = tot_classes - new_classes
            self.nb_classes = opts.num_classes
            self.nb_current_classes = tot_classes
            self.nb_new_classes = new_classes
        else:
            self.old_classes = 0
            self.nb_classes = None

        # Select the Loss Type
        reduction = 'none'

        self.bce = opts.bce or opts.icarl
        if self.bce:
            self.criterion = BCEWithLogitsLossWithIgnoreIndex(reduction=reduction)
        elif opts.unce and self.old_classes != 0:
            self.criterion = UnbiasedCrossEntropy(
                old_cl=self.old_classes, ignore_index=255, reduction=reduction
            )
        elif opts.nca and self.old_classes != 0:
            self.criterion = UnbiasedNCA(
                old_cl=self.old_classes,
                ignore_index=255,
                reduction=reduction,
                scale=model.module.scalar,
                margin=opts.nca_margin
            )
        elif opts.nca:
            self.criterion = NCA(
                scale=model.module.scalar,
                margin=opts.nca_margin,
                ignore_index=255,
                reduction=reduction
            )
        elif opts.focal_loss:
            self.criterion = FocalLoss(ignore_index=255, reduction=reduction, alpha=opts.alpha, gamma=opts.focal_loss_gamma)
        elif opts.focal_loss_new:
            self.criterion = FocalLossNew(ignore_index=255, reduction=reduction, index=self.old_classes, alpha=opts.alpha, gamma=opts.focal_loss_gamma)
        else:
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction=reduction)
        self.adaptivealpha=opts.adaptivealpha
        self.adaptivebeta=opts.adaptivebeta
        self.adaptivegamma=opts.adaptivegamma

        # ILTSS
        self.lde = opts.loss_de
        self.lde_flag = self.lde > 0. and model_old is not None
        self.lde_loss = nn.MSELoss()

        self.lkd = opts.loss_kd
        self.lkd_mask = opts.kd_mask
        self.kd_mask_adaptative_factor = opts.kd_mask_adaptative_factor
        self.lkd_flag = self.lkd > 0. and model_old is not None
        self.kd_need_labels = False
        if opts.unkd:
            self.lkd_loss = UnbiasedKnowledgeDistillationLoss(reduction="none", alpha=opts.alpha)
        elif opts.kd_bce_sig:
            self.lkd_loss = BCESigmoid(reduction="none", alpha=opts.alpha, shape=opts.kd_bce_sig_shape)
        elif opts.exkd_gt and self.old_classes > 0 and self.step > 0:
            self.lkd_loss = ExcludedKnowledgeDistillationLoss(
                reduction='none', index_new=self.old_classes, new_reduction="gt",
                initial_nb_classes=opts.inital_nb_classes,
                temperature_semiold=opts.temperature_semiold
            )
            self.kd_need_labels = True
        elif opts.exkd_sum and self.old_classes > 0 and self.step > 0:
            self.lkd_loss = ExcludedKnowledgeDistillationLoss(
                reduction='none', index_new=self.old_classes, new_reduction="sum",
                initial_nb_classes=opts.inital_nb_classes,
                temperature_semiold=opts.temperature_semiold
            )
            self.kd_need_labels = True
        else:
            self.lkd_loss = KnowledgeDistillationLoss(alpha=opts.alpha)

        # ICARL
        self.icarl_combined = False
        self.icarl_only_dist = False
        if opts.icarl:
            self.icarl_combined = not opts.icarl_disjoint and model_old is not None
            self.icarl_only_dist = opts.icarl_disjoint and model_old is not None
            if self.icarl_combined:
                self.licarl = nn.BCEWithLogitsLoss(reduction='mean')
                self.icarl = opts.icarl_importance
            elif self.icarl_only_dist:
                self.licarl = IcarlLoss(reduction='mean', bkg=opts.icarl_bkg)
        self.icarl_dist_flag = self.icarl_only_dist or self.icarl_combined

        # Regularization
        regularizer_state = trainer_state['regularizer'] if trainer_state is not None else None
        self.regularizer = get_regularizer(model, model_old, device, opts, regularizer_state)
        self.regularizer_flag = self.regularizer is not None
        self.reg_importance = opts.reg_importance

        self.ret_intermediate = self.lde or (opts.pod is not None)

        self.pseudo_labeling = opts.pseudo
        self.threshold = opts.threshold
        self.step_threshold = opts.step_threshold
        self.ce_on_pseudo = opts.ce_on_pseudo
        self.pseudo_nb_bins = opts.pseudo_nb_bins
        self.pseudo_soft = opts.pseudo_soft
        self.pseudo_soft_factor = opts.pseudo_soft_factor
        self.pseudo_ablation = opts.pseudo_ablation
        self.classif_adaptive_factor = opts.classif_adaptive_factor
        self.classif_adaptive_min_factor = opts.classif_adaptive_min_factor

        self.kd_new = opts.kd_new
        self.pod = opts.pod
        self.pod_options = opts.pod_options if opts.pod_options is not None else {}
        self.pod_factor = opts.pod_factor
        self.pod_prepro = opts.pod_prepro
        self.use_pod_schedule = not opts.no_pod_schedule
        self.pod_deeplab_mask = opts.pod_deeplab_mask
        self.pod_deeplab_mask_factor = opts.pod_deeplab_mask_factor
        self.pod_apply = opts.pod_apply
        self.pod_interpolate_last = opts.pod_interpolate_last
        self.deeplab_mask_downscale = opts.deeplab_mask_downscale
        self.spp_scales = opts.spp_scales
        self.pod_logits = opts.pod_logits
        self.pod_large_logits = opts.pod_large_logits

        self.align_weight = opts.align_weight
        self.align_weight_frequency = opts.align_weight_frequency

        self.dataset = opts.dataset

        self.entropy_min = opts.entropy_min

        self.kd_scheduling = opts.kd_scheduling

        self.sample_weights_new = opts.sample_weights_new

        self.temperature_apply = opts.temperature_apply
        self.temperature = opts.temperature
        self.mmm=opts.mmm
        self.pod = opts.pod
        self.dis = opts.dis
        self.sigmoid=opts.sigmoid
        self.aaaaaa=opts.aaaaaa
        self.pooling_without_zero=opts.pooling_without_zero
        self.yess=opts.yess
        self.ppp=opts.ppp
        # CIL
        self.ce_on_new = opts.ce_on_new

    def before(self, train_loader, logger):
        if self.pseudo_labeling is None:
            return
        if self.pseudo_labeling.split("_")[0] == "median" and self.step > 0:
            logger.info("Find median score")
            self.thresholds, _ = self.find_median(train_loader, self.device, logger)
        elif self.pseudo_labeling.split("_")[0] == "entropy" and self.step > 0:
            logger.info("Find median score")
            self.thresholds, self.max_entropy = self.find_median(
                train_loader, self.device, logger, mode="entropy"
            )

    def train(self, cur_epoch, optim, train_loader, scheduler=None, print_int=10, logger=None,initial_task_loss=None):
        """Train and return epoch loss"""
        logger.info(f"Pseudo labeling is: {self.pseudo_labeling}")
        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

        device = self.device
        model = self.model
        criterion = self.criterion

        model.module.in_eval = False
        if self.model_old is not None:
            self.model_old.in_eval = False

        epoch_loss = 0.0
        pod=0.0
        reg_loss = 0.0
        waiji=0.0
        pod0=0.0
        pod1=0.0
        pod2=0.0
        pod3=0.0
        pod4=0.0
        pod5=0.0
        epoch_loss_before = 0.0
        reg_loss_before=0.0
        pod_loss_before = 0.0
        waijibefore=0.0
        pod0before=0.0
        pod1before=0.0
        pod2before=0.0
        pod3before=0.0
        pod4before=0.0
        pod5before=0.0                
        interval_loss = 0.0
        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)
        pod_loss = torch.tensor(0.)
        loss_entmin = torch.tensor(0.)

        sample_weights = None

        train_loader.sampler.set_epoch(cur_epoch)

        model.train()
        for cur_step, (images, labels) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            masked_old_images=images.clone()       
            masked_new_images=images.clone()       

            original_labels = labels.clone()

            if (
                self.lde_flag or self.lkd_flag or self.icarl_dist_flag or self.pod is not None or
                self.pseudo_labeling is not None
            ) and self.model_old is not None:
                with torch.no_grad():
                    outputs_old, features_old = self.model_old(
                        images, ret_intermediate=self.ret_intermediate
                    )

            classif_adaptive_factor = 1.0
            if self.step > 0:
                mask_background = labels < self.old_classes

                if self.pseudo_labeling == "naive":
                    labels[mask_background] = outputs_old.argmax(dim=1)[mask_background]
                elif self.pseudo_labeling is not None and self.pseudo_labeling.startswith(
                    "threshold_"
                ):
                    threshold = float(self.pseudo_labeling.split("_")[1])
                    probs = torch.softmax(outputs_old, dim=1)
                    pseudo_labels = probs.argmax(dim=1)
                    pseudo_labels[probs.max(dim=1)[0] < threshold] = 255
                    labels[mask_background] = pseudo_labels[mask_background]
                elif self.pseudo_labeling == "confidence":
                    probs_old = torch.softmax(outputs_old, dim=1)
                    labels[mask_background] = probs_old.argmax(dim=1)[mask_background]
                    sample_weights = torch.ones_like(labels).to(device, dtype=torch.float32)
                    sample_weights[mask_background] = probs_old.max(dim=1)[0][mask_background]
                elif self.pseudo_labeling == "median":
                    probs = torch.softmax(outputs_old, dim=1)
                    max_probs, pseudo_labels = probs.max(dim=1)
                    pseudo_labels[max_probs < self.thresholds[pseudo_labels]] = 255
                    labels[mask_background] = pseudo_labels[mask_background]
                elif self.pseudo_labeling == "entropy":
                    probs = torch.softmax(outputs_old, dim=1)
                    max_probs, pseudo_labels = probs.max(dim=1)

                    mask_valid_pseudo = (entropy(probs) /
                                         self.max_entropy) < self.thresholds[pseudo_labels]


                    if self.pseudo_soft is None:
                        # All old labels that are NOT confident enough to be used as pseudo labels:
                        labels[~mask_valid_pseudo & mask_background] = 255

                        if self.pseudo_ablation is None:
                            # All old labels that are confident enough to be used as pseudo labels:
                            labels[mask_valid_pseudo & mask_background] = pseudo_labels[mask_valid_pseudo &
                                                                                        mask_background]
                        elif self.pseudo_ablation == "corrected_errors":
                            pass  # If used jointly with data_masking=current+old, the labels already
                                  # contrain the GT, thus all potentials errors were corrected.
                        elif self.pseudo_ablation == "removed_errors":
                            pseudo_error_mask = labels != pseudo_labels
                            kept_pseudo_labels = mask_valid_pseudo & mask_background & ~pseudo_error_mask
                            removed_pseudo_labels = mask_valid_pseudo & mask_background & pseudo_error_mask

                            labels[kept_pseudo_labels] = pseudo_labels[kept_pseudo_labels]
                            labels[removed_pseudo_labels] = 255
                        else:
                            raise ValueError(f"Unknown type of pseudo_ablation={self.pseudo_ablation}")
                    elif self.pseudo_soft == "soft_uncertain":
                        labels[mask_valid_pseudo & mask_background] = pseudo_labels[mask_valid_pseudo &
                                                                                    mask_background]

                    if self.classif_adaptive_factor:
                        # Number of old/bg pixels that are certain
                        num = (mask_valid_pseudo & mask_background).float().sum(dim=(1,2))
                        # Number of old/bg pixels
                        den =  mask_background.float().sum(dim=(1,2))
                        # If all old/bg pixels are certain the factor is 1 (loss not changed)
                        # Else the factor is < 1, i.e. the loss is reduced to avoid
                        # giving too much importance to new pixels
                        classif_adaptive_factor = num / (den + 1e-6)
                        classif_adaptive_factor = classif_adaptive_factor[:, None, None]

                        if self.classif_adaptive_min_factor:
                            classif_adaptive_factor = classif_adaptive_factor.clamp(min=self.classif_adaptive_min_factor)
                mask_new_for_feature=(labels>=self.old_classes)&(labels!=255)#新类为1
                mask_old_for_feature =(labels < self.old_classes) #旧类为1
                mask_old_for_feature=torch.stack([mask_old_for_feature, mask_old_for_feature, mask_old_for_feature], dim=1)
                mask_new_for_feature=torch.stack([mask_new_for_feature, mask_new_for_feature, mask_new_for_feature], dim=1)
                masked_old_images[mask_old_for_feature]=0.
                masked_new_images[mask_new_for_feature]=0.
                if self.model_old is not None:
                    with torch.no_grad():
                        outputs_old, features_old = self.model_old(
                            images, ret_intermediate=self.ret_intermediate
                        )
                        masked_old_outputs, masked_old_features = self.model_old(masked_old_images, ret_intermediate=self.ret_intermediate)   
                        masked_new_outputs, masked_new_features = self.model_old(masked_new_images, ret_intermediate=self.ret_intermediate)
            bg=labels==0
            bg=bg.unsqueeze(1).repeat(1,256,1,1)
            
            optim.zero_grad()
            outputs, features = model(images, ret_intermediate=self.ret_intermediate)
            # large_pre_logits=features["large_pre_logits"]
            # for lb,fea in zip(bg,large_pre_logits):
            #     fea_bg = []

            #     for f in fea:
            #         f=f[lb].unsqueeze(0)
            #         fea_bg.append(f)
            #     fea_bg=torch.cat(fea_bg,dim=0)
            #     k_means = KMeans(n_clusters=2, random_state=10)
                
                
            #     fea_bg=fea_bg.cpu().detach().numpy()
            #     k_means.fit(fea_bg)
            #     centers = k_means.cluster_centers_
            #     print(centers)             
            tasks_loss=[]
            # xxx BCE / Cross Entropy Loss
            if self.pseudo_soft is not None:
                loss = soft_crossentropy(
                    outputs,
                    labels,
                    outputs_old,
                    mask_valid_pseudo,
                    mask_background,
                    self.pseudo_soft,
                    pseudo_soft_factor=self.pseudo_soft_factor
                )
            elif not self.icarl_only_dist:
                if self.ce_on_pseudo and self.step > 0:
                    assert self.pseudo_labeling is not None
                    assert self.pseudo_labeling == "entropy"
                    # Apply UNCE on:
                    #   - all new classes (foreground)
                    #   - old classes (background) that were not selected for pseudo
                    loss_not_pseudo = criterion(
                        outputs,
                        original_labels,
                        mask=mask_background & mask_valid_pseudo  # what to ignore
                    )

                    # Apply CE on:
                    # - old classes that were selected for pseudo
                    _labels = original_labels.clone()
                    _labels[~(mask_background & mask_valid_pseudo)] = 255
                    _labels[mask_background & mask_valid_pseudo] = pseudo_labels[mask_background &
                                                                                 mask_valid_pseudo]
                    loss_pseudo = F.cross_entropy(
                        outputs, _labels, ignore_index=255, reduction="none"
                    )
                    # Each loss complete the others as they are pixel-exclusive
                    loss = loss_pseudo + loss_not_pseudo
                elif self.ce_on_new:
                    _labels = labels.clone()
                    _labels[_labels == 0] = 255
                    loss = criterion(outputs, _labels)  # B x H x W
                else:
                    loss = criterion(outputs, labels)  # B x H x W
            else:
                loss = self.licarl(outputs, labels, torch.sigmoid(outputs_old))

            if self.sample_weights_new is not None:
                sample_weights = torch.ones_like(original_labels).to(device, dtype=torch.float32)
                sample_weights[original_labels >= 0] = self.sample_weights_new

            if sample_weights is not None:
                loss = loss * sample_weights
            epoch_loss_before+=loss.mean().item()
            loss = classif_adaptive_factor * loss *self.ppp
            loss = loss.mean()  # scalar
            tasks_loss.append(loss)

            if self.icarl_combined:
                # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                n_cl_old = outputs_old.shape[1]
                # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                l_icarl = self.icarl * n_cl_old * self.licarl(
                    outputs.narrow(1, 0, n_cl_old), torch.sigmoid(outputs_old)
                )

            # xxx ILTSS (distillation on features or logits)
            if self.lde_flag:
                lde = self.lde * self.lde_loss(features['body'], features_old['body'])

            if self.lkd_flag:
                # resize new output to remove new logits and keep only the old ones
                if self.lkd_mask is not None and self.lkd_mask == "oldbackground":
                    kd_mask = labels < self.old_classes
                elif self.lkd_mask is not None and self.lkd_mask == "new":
                    kd_mask = labels >= self.old_classes
                else:
                    kd_mask = None

                if self.temperature_apply is not None:
                    temp_mask = torch.ones_like(labels).to(outputs.device).to(outputs.dtype)

                    if self.temperature_apply == "all":
                        temp_mask = temp_mask / self.temperature
                    elif self.temperature_apply == "old":
                        mask_bg = labels < self.old_classes
                        temp_mask[mask_bg] = temp_mask[mask_bg] / self.temperature
                    elif self.temperature_apply == "new":
                        mask_fg = labels >= self.old_classes
                        temp_mask[mask_fg] = temp_mask[mask_fg] / self.temperature
                    temp_mask = temp_mask[:, None]
                else:
                    temp_mask = 1.0

                if self.kd_need_labels:
                    lkd = self.lkd * self.lkd_loss(
                        outputs * temp_mask, outputs_old * temp_mask, labels, mask=kd_mask
                    )
                else:
                    lkd = self.lkd * self.lkd_loss(
                        outputs * temp_mask, outputs_old * temp_mask, mask=kd_mask
                    )

                if self.kd_new:  # WTF?
                    mask_bg = labels == 0
                    lkd = lkd[mask_bg]

                if kd_mask is not None and self.kd_mask_adaptative_factor:
                    lkd = lkd.mean(dim=(1, 2)) * kd_mask.float().mean(dim=(1, 2))
                lkd = torch.mean(lkd)
                
            if self.pod is not None and self.step > 0:
                attentions_old = features_old["attentions"]
                attentions_new = features["attentions"]
                attention_old_masked = masked_old_features["attentions"]
                attention_new_masked = masked_new_features["attentions"]

                if self.pod_logits:
                    attentions_old.append(features_old["sem_logits_small"])
                    attentions_new.append(features["sem_logits_small"])
                    attention_old_masked.append(masked_old_features["sem_logits_small"])
                    attention_new_masked.append(masked_new_features["sem_logits_small"])

                elif self.pod_large_logits:
                    attentions_old.append(outputs_old)
                    attentions_new.append(outputs)
                    attention_old_masked.append(masked_old_outputs)
                    attention_new_masked.append(masked_new_outputs)

                pod_loss,pod_loss_before,waijilossbefore,waijiloss,ppppbefore,ppppafter = features_distillation(
                    attentions_old,
                    attentions_new,
                    attention_old_masked,
                    attention_new_masked,
                    aaaaaa=self.aaaaaa,
                    mmm=self.mmm,
                    collapse_channels=self.pod,
                    labels=labels,
                    index_new_class=self.old_classes,
                    pod_apply=self.pod_apply,
                    pod_deeplab_mask=self.pod_deeplab_mask,
                    pod_deeplab_mask_factor=self.pod_deeplab_mask_factor,
                    interpolate_last=self.pod_interpolate_last,
                    pod_factor=self.pod_factor,
                    prepro=self.pod_prepro,
                    deeplabmask_upscale=not self.deeplab_mask_downscale,
                    spp_scales=self.spp_scales,
                    pod_options=self.pod_options,
                    outputs_old=outputs_old,
                    masked_old_outputs=masked_old_outputs,
                    use_pod_schedule=self.use_pod_schedule,
                    nb_current_classes=self.nb_current_classes,
                    nb_new_classes=self.nb_new_classes,
                    dis=self.dis,
                    sigmoid=self.sigmoid,
                    yess=self.yess,
                )
                tasks_loss.append(waijiloss)
                tasks_loss.append(lkd)

                for i in ppppafter:
                    tasks_loss.append(i)

                tasks_loss=torch.stack(tasks_loss)
            if cur_epoch == 0:
            # set L(0)
                initial_task_loss = tasks_loss.data.cpu().numpy()
            if self.entropy_min > 0. and self.step > 0:
                mask_new = labels > 0
                entropies = entropy(torch.softmax(outputs, dim=1))
                entropies[mask_new] = 0.
                pixel_amount = (~mask_new).float().sum(dim=(1, 2))
                loss_entmin = (entropies.sum(dim=(1, 2)) / pixel_amount).mean()

            if self.kd_scheduling:
                lkd = lkd * math.sqrt(self.nb_current_classes / self.nb_new_classes)

            # xxx first backprop of previous loss (compute the gradients for regularization methods)
            loss_tot =torch.sum(torch.mul(model.module.head.lossweights, tasks_loss))
            # 损失加入到列表中
            # print(torch.mul(model.module.head.lossweights, tasks_loss).shape,tasks_loss.shape,model.module.head.lossweights.shape)
            with amp.scale_loss(loss_tot, optim) as scaled_loss:
                scaled_loss.backward(retain_graph=True)

            # xxx Regularizer (EWC, RW, PI)


            # set the gradients of w_i(t) to zero because these gradients have to be updated using the GradNorm loss
            #print('Before turning to 0: {}'.format(model.module.head.lossweights.grad))
            #print('Turning to 0: {}'.format(model.module.head.lossweights.grad))
            model.module.head.lossweights.grad.data = model.module.head.lossweights.grad.data * 0.0
        # switch for each weighting algorithm:
        # --> grad norm
        
        # get layer of shared weights
            W = model.module.get_last_shared_layer()

            W0 = model.module.get_layer0()
            W1 = model.module.get_layer1()
            W2 = model.module.get_layer2()
            W3 = model.module.get_layer3()
        # get the gradient norms for each of the tasks
        # G^{(i)}_w(t) 
            norms = []
            featurenorms = []
            gygw = torch.autograd.grad(tasks_loss[0], W.parameters(), retain_graph=True)
            # compute the norm
            norms.append(torch.mean(torch.mul(model.module.head.lossweights[0], gygw[0])))




            
                # get the gradient of this task loss with respect to the shared parameters
            gygw = torch.autograd.grad(tasks_loss[1], W.parameters(), retain_graph=True)

            # compute the norm
            norms.append(torch.mean(torch.mul(model.module.head.lossweights[1], gygw[0])))
            gygw = torch.autograd.grad(tasks_loss[2], W.parameters(), retain_graph=True)
            # compute the norm
            norms.append(torch.mean(torch.mul(model.module.head.lossweights[2], gygw[0])))
            gygw = torch.autograd.grad(tasks_loss[3], W0.parameters(), retain_graph=True)
            # compute the norm

            featurenorms.append(torch.mean(torch.mul(model.module.head.lossweights[3], gygw[0])))
            gygw = torch.autograd.grad(tasks_loss[4], W1.parameters(), retain_graph=True)

            # compute the norm
            featurenorms.append(torch.mean(torch.mul(model.module.head.lossweights[4], gygw[0])))
            gygw = torch.autograd.grad(tasks_loss[5], W2.parameters(), retain_graph=True)

            # compute the norm
            featurenorms.append(torch.mean(torch.mul(model.module.head.lossweights[5], gygw[0])))

            gygw = torch.autograd.grad(tasks_loss[6], W3.parameters(), retain_graph=True)

            # compute the norm
            norms.append(torch.mean(torch.mul(model.module.head.lossweights[6], gygw[0])))  
            gygw = torch.autograd.grad(tasks_loss[7], W.parameters(), retain_graph=True)
            # compute the norm


            norms.append(torch.mean(torch.mul(model.module.head.lossweights[7], gygw[0])))

            gygw = torch.autograd.grad(tasks_loss[8], W.parameters(), retain_graph=True)
            # compute the norm
            norms.append(torch.mean(torch.mul(model.module.head.lossweights[8], gygw[0])))
            norms = torch.stack(norms)
            
            #print('G_w(t): {}'.format(norms))
                # get the gradient of this task loss with respect to the shared parameters
                    



            featurenorms = torch.stack(featurenorms)

            # compute the inverse training rate r_i(t) 
            # \curl{L}_i 
            if torch.cuda.is_available():
                loss_ratio = tasks_loss.data.cpu().numpy() / initial_task_loss
            else:
                loss_ratio = tasks_loss.data.numpy() / initial_task_loss
            # r_i(t)
            #print('r_i(t): {}'.format(inverse_train_rate))
            inverse_train_rate1=[]
            inverse_train_rate1.append(loss_ratio[0])
            inverse_train_rate1.append(loss_ratio[1])
            inverse_train_rate1.append(loss_ratio[2])
            inverse_train_rate1.append(loss_ratio[6])

            inverse_train_rate1.append(loss_ratio[7])

            inverse_train_rate1.append(loss_ratio[8])                

            inverse_train_rate1=np.array(inverse_train_rate1)
            inverse_train_rate1=inverse_train_rate1 / np.mean(inverse_train_rate1)
            inverse_train_rate2=[]

            inverse_train_rate2.append(loss_ratio[3])
            inverse_train_rate2.append(loss_ratio[4])
            inverse_train_rate2.append(loss_ratio[5])
            inverse_train_rate2=np.array(inverse_train_rate2)
            inverse_train_rate2=inverse_train_rate2 / np.mean(inverse_train_rate2)

            # compute the mean norm \tilde{G}_w(t) 
            mean_norm = np.mean(norms.data.cpu().numpy())
            mean_featurenorms = np.mean(featurenorms.data.cpu().numpy())


            #print('tilde G_w(t): {}'.format(mean_norm))

            layerweight=[]
            for i in range (4):
                if 5/4*model.module.head.lossweights[i+3]-model.module.head.lossweights[i+4]>0:
                    layerweight.append(5/4*model.module.head.lossweights[i+3]-model.module.head.lossweights[i+4])
                else:
                    layerweight.append(torch.tensor(0.).cuda())
            layerweight = torch.stack(layerweight)

            # compute the GradNorm loss 
            # this term has to remain constant
            constant_term = torch.tensor(mean_norm * (inverse_train_rate1 **self.adaptivealpha), requires_grad=False).cuda()
            constant_featureterm = torch.tensor(mean_featurenorms * (inverse_train_rate2 **self.adaptivealpha), requires_grad=False).cuda()

            #print('Constant term: {}'.format(constant_term))
            # this is the GradNorm loss itself
            grad_norm_loss1=self.adaptivegamma*torch.abs(model.module.head.lossweights[0]-self.nb_new_classes/self.old_classes*model.module.head.lossweights[2])
            grad_norm_loss2=self.adaptivebeta*torch.sum(layerweight)
            # print(grad_norm_loss2,grad_norm_loss1, torch.sum(torch.abs(norms - constant_term)))

            grad_norm_loss = 10*torch.sum(torch.abs(norms - constant_term))+10*torch.sum(torch.abs(featurenorms - constant_featureterm))+grad_norm_loss1+grad_norm_loss2
            # grad_norm_loss1+grad_norm_loss2
            #print('GradNorm loss {}'.format(grad_norm_loss))
            # compute the gradient for the weights
            
            model.module.head.lossweights.grad = torch.autograd.grad(grad_norm_loss, model.module.head.lossweights)[0]
            # print(model.module.head.lossweights.grad)
            optim.step()
            # print(model.module.head.lossweights)
            if scheduler is not None:
                scheduler.step()
            epoch_loss += loss.item()
            reg_loss += l_reg.item() if l_reg != 0. else 0.
            reg_loss += lkd.item() + lde.item() + l_icarl.item()
            interval_loss += loss.item() + lkd.item() + lde.item() + l_icarl.item() + pod_loss.item(
            ) + loss_entmin.item()
            pod+= pod_loss.item()
            waiji+= waijiloss.item()
            reg_loss_before+= lkd.item() /self.lkd
            waijibefore+=waijilossbefore.item()
            pod_loss_before+=pod_loss_before.item()
            pod0+=ppppafter[0].item()
            pod1+=ppppafter[1].item()
            pod2+=ppppafter[2].item()
            pod3+=ppppafter[3].item()
            pod4+=ppppafter[4].item()
            pod5+=ppppafter[5].item()
            pod0before+=ppppbefore[0].item()
            pod1before+=ppppbefore[1].item()
            pod2before+=ppppbefore[2].item()
            pod3before+=ppppbefore[3].item()
            pod4before+=ppppbefore[4].item()
            pod5before+=ppppbefore[5].item()             
            interval_loss += l_reg.item() if l_reg != 0. else 0.

            if (cur_step + 1) % print_int == 0:
                interval_loss = interval_loss / print_int
                logger.info(
                    f"Epoch {cur_epoch}, Batch {cur_step + 1}/{len(train_loader)},"
                    f" Loss={interval_loss}"
                )
                logger.info(
                    f"Loss made of: CE {loss}, LKD {lkd}, LDE {lde}, LReg {l_reg}, POD {pod_loss} EntMin {loss_entmin}"
                )
                # visualization
                if logger is not None:
                    x = cur_epoch * len(train_loader) + cur_step + 1
                    logger.add_scalar('Loss', interval_loss, x)
                interval_loss = 0.0

        # collect statistics from multiple processes
        waijibefore=torch.tensor(waijibefore).to(self.device)
        pod0before= torch.tensor(pod0before).to(self.device)        
        pod1before=torch.tensor(pod1before).to(self.device)
        pod2before= torch.tensor(pod2before).to(self.device)        
        pod3before=torch.tensor(pod3before).to(self.device)
        pod4before= torch.tensor(pod4before).to(self.device)
        pod5before= torch.tensor(pod5before).to(self.device)
        reg_loss_before=torch.tensor(reg_loss_before).to(self.device)
        epoch_loss_before= torch.tensor(epoch_loss_before).to(self.device)
        epoch_loss = torch.tensor(epoch_loss).to(self.device)
        reg_loss = torch.tensor(reg_loss).to(self.device)
        pod = torch.tensor(pod).to(self.device)
        waiji = torch.tensor(waiji).to(self.device)
        pod0 = torch.tensor(pod0).to(self.device)
        pod1 = torch.tensor(pod1).to(self.device)
        pod2 = torch.tensor(pod2).to(self.device)
        pod3 = torch.tensor(pod3).to(self.device)
        pod4 = torch.tensor(pod4).to(self.device)
        pod5 =torch.tensor(pod5).to(self.device)
        pod_loss_before=torch.tensor(pod_loss_before).to(self.device)
        torch.distributed.reduce(epoch_loss_before, dst=0)
        torch.distributed.reduce(epoch_loss, dst=0)
        torch.distributed.reduce(reg_loss_before, dst=0)
        torch.distributed.reduce(reg_loss, dst=0)
        torch.distributed.reduce(pod_loss_before, dst=0)
        torch.distributed.reduce(pod, dst=0)
        torch.distributed.reduce(waijibefore, dst=0)
        torch.distributed.reduce(pod0before, dst=0)
        torch.distributed.reduce(pod1before, dst=0)
        torch.distributed.reduce(pod2before, dst=0)
        torch.distributed.reduce(pod3before, dst=0)
        torch.distributed.reduce(pod4before, dst=0)
        torch.distributed.reduce(pod5before, dst=0)
        torch.distributed.reduce(waiji, dst=0)
        torch.distributed.reduce(pod0, dst=0)
        torch.distributed.reduce(pod1, dst=0)
        torch.distributed.reduce(pod2, dst=0)
        torch.distributed.reduce(pod3, dst=0)
        torch.distributed.reduce(pod4, dst=0)
        torch.distributed.reduce(pod5, dst=0)
        if distributed.get_rank() == 0:
            epoch_loss_before = epoch_loss_before / distributed.get_world_size() / len(train_loader)
            reg_loss_before = reg_loss_before / distributed.get_world_size() / len(train_loader)
            pod_loss_before = pod_loss_before / distributed.get_world_size() / len(train_loader)
            pod0before = pod0before / distributed.get_world_size() / len(train_loader)
            pod1before = pod1before / distributed.get_world_size() / len(train_loader)
            pod2before = pod2before / distributed.get_world_size() / len(train_loader)
            pod3before = pod3before / distributed.get_world_size() / len(train_loader)
            pod4before = pod4before / distributed.get_world_size() / len(train_loader)
            pod5before = pod5before / distributed.get_world_size() / len(train_loader)
            waijibefore = waijibefore / distributed.get_world_size() / len(train_loader)
            epoch_loss = epoch_loss / distributed.get_world_size() / len(train_loader)
            reg_loss = reg_loss / distributed.get_world_size() / len(train_loader)
            pod = pod / distributed.get_world_size() / len(train_loader)
            pod0 = pod0 / distributed.get_world_size() / len(train_loader)
            pod1 = pod1 / distributed.get_world_size() / len(train_loader)
            pod2 = pod2 / distributed.get_world_size() / len(train_loader)
            pod3 = pod3 / distributed.get_world_size() / len(train_loader)
            pod4 = pod4 / distributed.get_world_size() / len(train_loader)
            pod5 = pod5 / distributed.get_world_size() / len(train_loader)
            waiji = waiji / distributed.get_world_size() / len(train_loader)
            
        logger.info(f"Epoch {cur_epoch}, Class Loss={epoch_loss}, Reg Loss={reg_loss}")
        # if cur_epoch == 0:
        # # set L(0)
        #     initial_task_loss= []
        #     initial_task_loss.append(epoch_loss_before)##可以改
        #     initial_task_loss.append(waijibefore)
        #     initial_task_loss.append(pod0before)
        #     initial_task_loss.append(pod1before)
        #     initial_task_loss.append(pod2before)
        #     initial_task_loss.append(pod3before)
        #     initial_task_loss.append(pod4before)
        #     initial_task_loss.append(pod5before)
        #     initial_task_loss.append(reg_loss_before)

        #     initial_task_loss=torch.stack(initial_task_loss)
        #     initial_task_loss = initial_task_loss.data.cpu().numpy()
        return (epoch_loss, reg_loss,pod,pod0,pod1,pod2,pod3,pod4,pod5,waiji,epoch_loss_before,reg_loss_before,pod_loss_before,pod0before,pod1before,pod2before,pod3before,pod4before,pod5before,waijibefore,initial_task_loss)
    def L2norm(self, A):
        A=A.squeeze(-1)
        A=A.squeeze(-1)
        AT=A.transpose(0,1)
        ATA=AT.matmul(A)
        eig,_=torch.eig(ATA)
        maxeig=torch.max(eig)
        return maxeig**1/2
    def find_median(self, train_loader, device, logger, mode="probability"):
        """Find the median prediction score per class with the old model.

        Computing the median naively uses a lot of memory, to allievate it, instead
        we put the prediction scores into a histogram bins and approximate the median.

        https://math.stackexchange.com/questions/2591946/how-to-find-median-from-a-histogram
        """
        if mode == "entropy":
            max_value = torch.log(torch.tensor(self.nb_current_classes).float().to(device))
            nb_bins = 100
        else:
            max_value = 1.0
            nb_bins = 20  # Bins of 0.05 on a range [0, 1]
        if self.pseudo_nb_bins is not None:
            nb_bins = self.pseudo_nb_bins

        histograms = torch.zeros(self.nb_current_classes, nb_bins).long().to(self.device)

        for cur_step, (images, labels) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs_old, features_old = self.model_old(images, ret_intermediate=False)

            mask_bg = labels == 0
            probas = torch.softmax(outputs_old, dim=1)
            max_probas, pseudo_labels = probas.max(dim=1)

            if mode == "entropy":
                values_to_bins = entropy(probas)[mask_bg].view(-1) / max_value
            else:
                values_to_bins = max_probas[mask_bg].view(-1)

            x_coords = pseudo_labels[mask_bg].view(-1)
            y_coords = torch.clamp((values_to_bins * nb_bins).long(), max=nb_bins - 1)

            histograms.index_put_(
                (x_coords, y_coords),
                torch.LongTensor([1]).expand_as(x_coords).to(histograms.device),
                accumulate=True
            )

            if cur_step % 10 == 0:
                logger.info(f"Median computing {cur_step}/{len(train_loader)}.")

        thresholds = torch.zeros(self.nb_current_classes, dtype=torch.float32).to(
            self.device
        )  # zeros or ones? If old_model never predict a class it may be important

        logger.info("Approximating median")
        for c in range(self.nb_current_classes):
            total = histograms[c].sum()
            if total <= 0.:
                continue

            half = total / 2
            running_sum = 0.
            for lower_border in range(nb_bins):
                lower_border = lower_border / nb_bins
                bin_index = int(lower_border * nb_bins)
                if half >= running_sum and half <= (running_sum + histograms[c, bin_index]):
                    break
                running_sum += lower_border * nb_bins

            median = lower_border + ((half - running_sum) /
                                     histograms[c, bin_index].sum()) * (1 / nb_bins)

            thresholds[c] = median

        base_threshold = self.threshold
        if "_" in mode:
            mode, base_threshold = mode.split("_")
            base_threshold = float(base_threshold)
        if self.step_threshold is not None:
            self.threshold += self.step * self.step_threshold

        if mode == "entropy":
            for c in range(len(thresholds)):
                thresholds[c] = max(thresholds[c], base_threshold)
        else:
            for c in range(len(thresholds)):
                thresholds[c] = min(thresholds[c], base_threshold)
        logger.info(f"Finished computing median {thresholds}")
        return thresholds.to(device), max_value

    def validate(self, loader, metrics, ret_samples_ids=None, logger=None, end_task=False):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        criterion = self.criterion
        model.eval()

        model.module.in_eval = True
        if self.model_old is not None:
            self.model_old.in_eval = True

        class_loss = 0.0
        reg_loss = 0.0
        lkd = torch.tensor(0.)
        lde = torch.tensor(0.)
        l_icarl = torch.tensor(0.)
        l_reg = torch.tensor(0.)

        if self.step > 0 and self.align_weight_frequency == "epoch":
            model.module.align_weight(self.align_weight)
        elif self.step > 0 and self.align_weight_frequency == "task" and end_task:
            model.module.align_weight(self.align_weight)

        ret_samples = []
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)

                if (
                    self.lde_flag or self.lkd_flag or self.icarl_dist_flag
                ) and self.model_old is not None:
                    with torch.no_grad():
                        outputs_old, features_old = self.model_old(images, ret_intermediate=True)

                outputs, features = model(images, ret_intermediate=True)

                # xxx BCE / Cross Entropy Loss
                if not self.icarl_only_dist:
                    loss = criterion(outputs, labels)  # B x H x W
                else:
                    loss = self.licarl(outputs, labels, torch.sigmoid(outputs_old))

                loss = loss.mean()  # scalar

                if self.icarl_combined:
                    # tensor.narrow( dim, start, end) -> slice tensor from start to end in the specified dim
                    n_cl_old = outputs_old.shape[1]
                    # use n_cl_old to sum the contribution of each class, and not to average them (as done in our BCE).
                    l_icarl = self.icarl * n_cl_old * self.licarl(
                        outputs.narrow(1, 0, n_cl_old), torch.sigmoid(outputs_old)
                    )

                # xxx ILTSS (distillation on features or logits)
                if self.lde_flag:
                    lde = self.lde_loss(features['body'], features_old['body'])

                if self.lkd_flag and not self.kd_need_labels:
                    lkd = self.lkd_loss(outputs, outputs_old).mean()

                # xxx Regularizer (EWC, RW, PI)
                if self.regularizer_flag:
                    l_reg = self.regularizer.penalty()

                class_loss += loss.item()
                reg_loss += l_reg.item() if l_reg != 0. else 0.
                reg_loss += lkd.item() + lde.item() + l_icarl.item()

                _, prediction = outputs.max(dim=1)

                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)

                if ret_samples_ids is not None and i in ret_samples_ids:  # get samples
                    ret_samples.append((images[0].detach().cpu().numpy(), labels[0], prediction[0]))

            # collect statistics from multiple processes
            metrics.synch(device)
            score = metrics.get_results()

            class_loss = torch.tensor(class_loss).to(self.device)
            reg_loss = torch.tensor(reg_loss).to(self.device)

            torch.distributed.reduce(class_loss, dst=0)
            torch.distributed.reduce(reg_loss, dst=0)

            if distributed.get_rank() == 0:
                class_loss = class_loss / distributed.get_world_size() / len(loader)
                reg_loss = reg_loss / distributed.get_world_size() / len(loader)

            if logger is not None:
                logger.info(
                    f"Validation, Class Loss={class_loss}, Reg Loss={reg_loss} (without scaling)"
                )

        return (class_loss, reg_loss), score, ret_samples

    def state_dict(self):
        state = {"regularizer": self.regularizer.state_dict() if self.regularizer_flag else None}

        return state

    def load_state_dict(self, state):
        if state["regularizer"] is not None and self.regularizer is not None:
            self.regularizer.load_state_dict(state["regularizer"])


def entropy(probabilities):
    """Computes the entropy per pixel.

    # References:
        * ESL: Entropy-guided Self-supervised Learning for Domain Adaptation in Semantic Segmentation
          Saporta et al.
          CVPR Workshop 2020

    :param probabilities: Tensor of shape (b, c, w, h).
    :return: One entropy per pixel, shape (b, w, h)
    """
    factor = 1 / math.log(probabilities.shape[1] + 1e-8)
    return -factor * torch.mean(probabilities * torch.log(probabilities + 1e-8), dim=1)


def features_distillation(
    list_attentions_a,
    list_attentions_b,
    list_attentions_c,
    list_attentions_d,
    aaaaaa,
    mmm,
    collapse_channels="spatial",
    normalize=True,
    labels=None,
    index_new_class=None,
    pod_apply="all",
    pod_deeplab_mask=False,
    pod_deeplab_mask_factor=None,
    interpolate_last=False,
    pod_factor=1.,
    prepro="pow",
    deeplabmask_upscale=True,
    spp_scales=[1, 2, 4],
    pod_options=None,
    outputs_old=None,
    masked_old_outputs=None,
    use_pod_schedule=True,
    nb_current_classes=-1,
    nb_new_classes=-1,
    dis=None,
    sigmoid=False,
    yess=False,
):
    """A mega-function comprising several features-based distillation.

    :param list_attentions_a: A list of attention maps, each of shape (b, n, w, h).
    :param list_attentions_b: A list of attention maps, each of shape (b, n, w, h).
    :param collapse_channels: How to pool the channels.
    :param memory_flags: Integer flags denoting exemplars.
    :param only_old: Only apply loss to exemplars.
    :return: A float scalar loss.
    """
    device = list_attentions_a[0].device

    assert len(list_attentions_a) == len(list_attentions_b)

    if pod_deeplab_mask_factor is None:
        pod_deeplab_mask_factor = pod_factor

    #if collapse_channels in ("spatial_tuple", "spp", "spp_noNorm", "spatial_noNorm"):
    normalize = False

    apply_mask = "background"
    upscale_mask_topk = 1
    mask_position = "last"  # Others choices "all" "backbone"
    use_adaptative_factor = False
    mix_new_old = None
    ppppbefore=[]
    ppppafter=[]
    qq = torch.tensor(0.).to(list_attentions_a[0].device)

    loss = torch.tensor(0.).to(list_attentions_a[0].device)
    for ii, (a, b, c,d) in enumerate(zip(list_attentions_a, list_attentions_b, list_attentions_c,list_attentions_d)):
        adaptative_pod_factor = 1.0
        difference_function = "frobenius"
        pool = True
        use_adaptative_factor = False
        handle_extra_channels = "sum"
        normalize_per_scale = False
        b1,c1,h1,w1=a.shape

        if pod_options and pod_options.get("switch"):
            if ii < len(list_attentions_a) - 1:
                if "before" in pod_options["switch"]:
                    collapse_channels = pod_options["switch"]["before"].get(
                        "type", collapse_channels
                    )
                    pod_factor = pod_options["switch"]["before"].get("factor", pod_factor)
                    normalize = pod_options["switch"]["before"].get("norm", False)
                    prepro = pod_options["switch"]["before"].get("prepro", prepro)
                    use_adaptative_factor = pod_options["switch"]["before"].get(
                        "use_adaptative_factor", use_adaptative_factor
                    )
            else:
                if "after" in pod_options["switch"]:
                    collapse_channels = pod_options["switch"]["after"].get(
                        "type", collapse_channels
                    )
                    pod_factor = pod_options["switch"]["after"].get("factor", pod_factor)
                    normalize = pod_options["switch"]["after"].get("norm", False)
                    prepro = pod_options["switch"]["after"].get("prepro", prepro)

                    apply_mask = pod_options["switch"]["after"].get("apply_mask", apply_mask)
                    upscale_mask_topk = pod_options["switch"]["after"].get(
                        "upscale_mask_topk", upscale_mask_topk
                    )
                    use_adaptative_factor = pod_options["switch"]["after"].get(
                        "use_adaptative_factor", use_adaptative_factor
                    )
                    mix_new_old = pod_options["switch"]["after"].get("mix_new_old", mix_new_old)

                    handle_extra_channels = pod_options["switch"]["after"].get(
                        "extra_channels", handle_extra_channels
                    )
                    spp_scales = pod_options["switch"]["after"].get(
                        "spp_scales", spp_scales
                    )
                    use_pod_schedule = pod_options["switch"]["after"].get(
                        "use_pod_schedule", use_pod_schedule
                    )

            mask_position = pod_options["switch"].get("mask_position", mask_position)
            normalize_per_scale = pod_options["switch"].get(
                "normalize_per_scale", normalize_per_scale
            )
            pool = pod_options.get("pool", pool)

        if a.shape[1] != b.shape[1]:
            assert ii == len(list_attentions_a) - 1
            assert a.shape[0] == b.shape[0]
            assert a.shape[2] == b.shape[2]
            assert a.shape[3] == b.shape[3]

            assert handle_extra_channels in ("trim", "sum"), handle_extra_channels

            if handle_extra_channels == "sum":
                _b = torch.zeros_like(a).to(a.dtype).to(a.device)
                _b[:, 0] = b[:, 0] + b[:, index_new_class:].sum(dim=1)
                _b[:, 1:] = b[:, 1:index_new_class]
                b = _b
            elif handle_extra_channels == "trim":
                b = b[:, :index_new_class]

        # shape of (b, n, w, h)
        assert a.shape == b.shape, (a.shape, b.shape)

        if dis == "L1_mean":
            dis_old = torch.abs(c - a).mean(dim=1)
            dis_new = torch.abs(d - a).mean(dim=1)
        elif dis== "mean_L1":
            dis_old = torch.abs(c.mean(dim=1)-a.mean(dim=1)).float()
            dis_new = torch.abs(d.mean(dim=1)-a.mean(dim=1)).float()
            ch_old = torch.abs(c.mean(dim=(2,3))-a.mean(dim=(2,3))).float()
            ch_new = torch.abs(d.mean(dim=(2,3))-a.mean(dim=(2,3))).float()
        elif dis== "L2":
            dis_old=F.normalize(c - a, p=2, dim=1)
            dis_new=F.normalize(d - a, p=2, dim=1)
            ch_old=F.normalize(c - a, p=2, dim=(2,3))
            ch_new=F.normalize(d - a, p=2, dim=(2,3))               
        elif dis== "cos":
            dis_old=torch.cosine_similarity(a, c, dim=1).float()
            dis_new=torch.cosine_similarity(a, d, dim=1).float()               
            ch_old = torch.abs(c.mean(dim=(2,3))-a.mean(dim=(2,3))).float()
            ch_new = torch.abs(d.mean(dim=(2,3))-a.mean(dim=(2,3))).float()
        mask=[]
        ch=[]
        for i,(old,new, x,old_ch,new_ch) in enumerate(zip(dis_old,dis_new, labels,ch_old,ch_new)):
            if sigmoid=="old":
                ma=torch.sigmoid(old)             
            elif sigmoid=="new":
                                     
                ma=1-torch.sigmoid(new)             
            elif sigmoid=="none":                     
                old_labels=(x < index_new_class)
                new_labels=(x >= index_new_class)& (x!=255)
                h,w=x.shape
                num_old_labels_pixel=torch.sum(old_labels).item()
                num_new_labels_pixel=torch.sum(new_labels).item()
                ratio_old=num_old_labels_pixel/h/w
                ratio_new=num_new_labels_pixel/h/w
                value_old = torch.quantile(old, q=ratio_old)#old是mask掉旧类的特征 从小到大排序，q=1是最大，0.1是偏小
                value_new = torch.quantile(new, q=ratio_new)
                ma_old=torch.where((old <  value_old.item()), 0, 1)
                ma_new=torch.where((new >  value_new.item()), 0, 1)
                ma=(ma_old&ma_new).float()

                # value_new = torch.quantile(new, q=ratio_old)
                # ma=torch.where((new >  value_new.item()), 0, 1).float()                
            elif sigmoid=="newnone":
                old_labels=(x < index_new_class)

                new_labels=(x >= index_new_class)& (x!=255)
                h,w=x.shape
                num_old_labels_pixel=torch.sum(old_labels).item()

                num_new_labels_pixel=torch.sum(new_labels).item()
                ratio_new=num_new_labels_pixel/h/w
                ratio_old=num_old_labels_pixel/h/w

                value_new = torch.quantile(new, q=ratio_old)
                ma=torch.where((new >  value_new.item()), 0, 1).float()     
            elif sigmoid=="gt":
                new_labels=(x >= index_new_class)& (x!=255)
                ma=new_labels.float()
            elif sigmoid=="whc":                     
                old_labels=(x < index_new_class)
                new_labels=(x >= index_new_class)& (x!=255)
                h,w=x.shape
                num_old_labels_pixel=torch.sum(old_labels).item()
                num_new_labels_pixel=torch.sum(new_labels).item()
                ratio_old=num_old_labels_pixel/h/w
                ratio_new=num_new_labels_pixel/h/w
                if dis== "cos":
                    value_old = torch.quantile(old, q=ratio_old)#old是mask掉旧类的特征 从小到大排序，q=1是最大，0.1是偏小
                    value_new = torch.quantile(new, q=ratio_new)
                    ma_old=torch.where((old <  value_old.item()), 0, 1)
                    ma_new=torch.where((new >  value_new.item()), 0, 1)
                else:
                    value_old = torch.quantile(old, q=ratio_new)#old是mask掉旧类的特征 从小到大排序，q=1是最大，0.1是偏小
                    value_new = torch.quantile(new, q=ratio_old)
                    ma_old=torch.where((old >  value_old.item()), 0, 1)
                    ma_new=torch.where((new <  value_new.item()), 0, 1)
                ma=(ma_old|ma_new).float()
                # print(ratio_old,ratio_new,ma_old.sum()/ma_old.shape[0]/ma_old.shape[0],ma_new.sum()/ma_old.shape[0]/ma_old.shape[0],ma.sum()/ma_old.shape[0]/ma_old.shape[0],(ma_old|ma_new).float().sum()/ma_old.shape[0]/ma_old.shape[0])

                # value_new = torch.quantile(new, q=ratio_old)
                # ma=torch.where((new >  value_new.item()), 0, 1).float()     
                
                ch_old_val = torch.quantile(old_ch, q=ratio_old)
                ch_new_val = torch.quantile(new_ch, q=ratio_new)
                ma_ch_old=torch.where((old_ch <  ch_old_val.item()), 0, 1)
                ma_ch_new=torch.where((new_ch >  ch_new_val.item()), 0, 1)
                ma_ch=(ma_ch_old|ma_ch_new).float()
                              
            ma=ma.unsqueeze(0)
            mask.append(ma)
            if sigmoid=="whc":
                ma_ch=ma_ch.unsqueeze(0)
                ch.append(ma_ch)
        mask=torch.cat(mask,dim=0)
        mask=mask.unsqueeze(1)

        if sigmoid=="gt":
            out_size = a.shape[-2:]
            mask=F.interpolate(mask,out_size, mode="bilinear", align_corners=False)       
         
        mask=mask.repeat(1,c1,1,1)
        if sigmoid=="whc" and ii< (len(list_attentions_a) - 1):
            ch=torch.cat(ch,dim=0)
            ch=ch.unsqueeze(2)
            ch=ch.unsqueeze(3)
            ch=ch.repeat(1,1,h1,w1)
            mask=mask*ch

        # for (d, ma, x) in zip(dis, mask, labels):
        #     old_labels=x < index_new_class
        #     h,w=x.shape
        #     num_old_labels_pixel=torch.sum(old_labels).item()
        #     ratio=num_old_labels_pixel/h/w
        #     value = torch.quantile(d, q=ratio)#从小到大排序，找到0.8那个点的值 （batchsize，1）
        #     ma = torch.where((dis > value.item()), 0, 1).float()
        
        #                 
            # print("d: "+str(d.shape),"ma: "+str(ma.shape)+"x: "+str(x.shape)+"value: "+str(value))

        # a = F.normalize(a, dim=1, p=2)
        # b = F.normalize(b, dim=1, p=2)
        # bbbb=mask.repeat(1,3,1,1)
        # t,y,u,p=bbbb.shape
        # if u==128:
        #     for i,pic,aaa in zip(images,masked_new_images,bbbb):
        #         toPIL = transforms.ToPILImage() #这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
        #         pic = toPIL(pic)
        #         aaa=torch.where((aaa==1), 256, 0)            
        #         aaa=toPIL(aaa)
        #         i=toPIL(i)
                
        #         pic.save('/home/wuyiwen/CVPR2021_PLOP-main/images/maskedimages'+str(i)+str(list_attentions_a[0].device)+'.jpg')
        #         i.save('/home/wuyiwen/CVPR2021_PLOP-main/images/images'+str(i)+str(list_attentions_a[0].device)+'.jpg')
        #         aaa.save('/home/wuyiwen/CVPR2021_PLOP-main/images/mask'+str(i)+str(list_attentions_a[0].device)+'.jpg')

        if ii >1:
            a= a*mask
            b= b*mask



        if not pod_deeplab_mask and use_adaptative_factor:
            adaptative_pod_factor = (labels == 0).float().mean()


        if collapse_channels == "spatial":
            a_h = a.sum(dim=3).view(a.shape[0], -1)
            b_h = b.sum(dim=3).view(b.shape[0], -1)
            a_w = a.sum(dim=2).view(a.shape[0], -1)
            b_w = b.sum(dim=2).view(b.shape[0], -1)
            a = torch.cat([a_h, a_w], dim=-1)
            b = torch.cat([b_h, b_w], dim=-1)
        elif collapse_channels == "global":
            a = _global_pod(a, spp_scales, normalize=False)
            b = _global_pod(b, spp_scales, normalize=False)
        elif collapse_channels == "mulglobal":
            a = torch.pow(a, 2)
            b = torch.pow(b, 2)





            a = mulglobal(a,scale=[4,8,12,16,20,24])
            b = mulglobal(b,scale=[4,8,12,16,20,24])            
        elif collapse_channels == "local":
            if(ii==len(list_attentions_a) - 2):# and yess:
                cc= second_order_pooling(a,mmm)
                dd = second_order_pooling(b,mmm)#4 8                       
            a = torch.pow(a, 2)
            b = torch.pow(b, 2)            
 



            if pod_deeplab_mask and (
                (ii == len(list_attentions_a) - 1 and mask_position == "last") or
                mask_position == "all"
            ):
                if pod_deeplab_mask_factor == 0.:
                    continue

                pod_factor = pod_deeplab_mask_factor

                if apply_mask == "background":
                    mask = labels < index_new_class
                elif apply_mask == "old":
                    pseudo_labels = labels.clone()
                    mask_background = labels == 0
                    pseudo_labels[mask_background] = outputs_old.argmax(dim=1)[mask_background]

                    mask = (labels < index_new_class) & (0 < pseudo_labels)
                else:
                    raise NotImplementedError(f"Unknown apply_mask={apply_mask}.")

                if deeplabmask_upscale:
                    a = F.interpolate(
                        torch.topk(a, k=upscale_mask_topk, dim=1)[0],
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                    b = F.interpolate(
                        torch.topk(b, k=upscale_mask_topk, dim=1)[0],
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )
                else:
                    mask = F.interpolate(mask[:, None].float(), size=a.shape[-2:]).bool()[:, 0]

                if use_adaptative_factor:
                    print("aaaa")
                    adaptative_pod_factor = mask.float().mean(dim=(1, 2))

                a = _local_pod_masked(
                    a, mask, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
                )
                b = _local_pod_masked(
                    b, mask, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
                )
            else:
                a = _local_pod(
                    a, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
                )
                b = _local_pod(
                    b, spp_scales, normalize=False, normalize_per_scale=normalize_per_scale
                )
        else:
            raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

        if ii == len(list_attentions_a) - 1 and pod_options is not None:
            if "difference_function" in pod_options:
                difference_function = pod_options["difference_function"]
        elif pod_options is not None:
            if "difference_function_all" in pod_options:
                difference_function = pod_options["difference_function_all"]

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        if difference_function == "frobenius":
            if isinstance(a, list):
                for ii,(aa, bb) in enumerate(zip(a, b)) :
                    if ii==0:
                        layer_loss = torch.frobenius_norm(aa - bb, dim=-1)/6.
                    else:
                        layer_loss+=torch.frobenius_norm(aa - bb, dim=-1)/6.
            else:
                layer_loss = torch.frobenius_norm(a - b, dim=-1)
                if(ii==len(list_attentions_a) - 2):# and yess:
                    layer_loss1= torch.frobenius_norm(cc - dd, dim=-1)
        elif difference_function == "frobenius_mix":
            layer_loss_old = torch.frobenius_norm(a[0] - b[0], dim=-1)
            layer_loss_new = torch.frobenius_norm(a[1] - b[1], dim=-1)

            layer_loss = mix_new_old * layer_loss_old + (1 - mix_new_old) * layer_loss_new
        elif difference_function == "l1":
            if isinstance(a, list):
                layer_loss = torch.tensor(
                    [torch.norm(aa - bb, p=1, dim=-1) for aa, bb in zip(a, b)]
                ).to(device)
            else:
                layer_loss = torch.norm(a - b, p=1, dim=-1)
        elif difference_function == "kl":
            d1, d2, d3 = a.shape
            a = (a.view(d1 * d2, d3) + 1e-8).log()
            b = b.view(d1 * d2, d3) + 1e-8

            layer_loss = F.kl_div(a, b, reduction="none").view(d1, d2, d3).sum(dim=(1, 2))
        elif difference_function == "bce":
            d1, d2, d3 = a.shape
            layer_loss = bce(a.view(d1 * d2, d3), b.view(d1 * d2, d3)).view(d1, d2,
                                                                            d3).mean(dim=(1, 2))
        else:
            raise NotImplementedError(f"Unknown difference_function={difference_function}")
        ####    #####
        before = torch.mean(layer_loss)

        ppppbefore.append(before)
        qq+= before

        ####    #####
        
        layer_loss = torch.mean(adaptative_pod_factor * layer_loss)
        ppppafter.append(layer_loss*pod_factor)
        if(ii==len(list_attentions_a) - 2):# and yess:
            layerloss1before=torch.mean(layer_loss1)
            qq+= layerloss1before*len(list_attentions_a)
            layer_loss1 = torch.mean(adaptative_pod_factor*layer_loss1*len(list_attentions_a))
            layer_loss = pod_factor * layer_loss+layer_loss1*aaaaaa
        else:
            layer_loss = pod_factor * layer_loss

        if pod_factor <= 0.:
            continue

        if use_pod_schedule:
            layer_loss = layer_loss * math.sqrt(nb_current_classes / nb_new_classes)
        loss += layer_loss

    return loss / len(list_attentions_a),qq/ len(list_attentions_a),layerloss1before,layer_loss1*aaaaaa/ len(list_attentions_a),ppppbefore,ppppafter


def bce(x, y):
    return -(y * torch.log(x + 1e-6) + (1 - y) * torch.log((1 - x) + 1e-6))


def _local_pod(x, spp_scales=[1, 2, 4], normalize=False, normalize_per_scale=False):
    b = x.shape[0]
    w = x.shape[-1]
    emb = []

    for scale_index, scale in enumerate(spp_scales):
        k = w // scale

        nb_regions = scale**2

        for i in range(scale):
            for j in range(scale):
                tensor = x[..., i * k:(i + 1) * k, j * k:(j + 1) * k]

                horizontal_pool = tensor.mean(dim=3).view(b, -1)
                vertical_pool = tensor.mean(dim=2).view(b, -1)

                if normalize_per_scale is True:
                    horizontal_pool = horizontal_pool / nb_regions
                    vertical_pool = vertical_pool / nb_regions
                elif normalize_per_scale == "spm":
                    if scale_index == 0:
                        factor = 2 ** (len(spp_scales) - 1)
                    else:
                        factor = 2 ** (len(spp_scales) - scale_index)
                    horizontal_pool = horizontal_pool / factor
                    vertical_pool = vertical_pool / factor

                if normalize:
                    horizontal_pool = F.normalize(horizontal_pool, dim=1, p=2)
                    vertical_pool = F.normalize(vertical_pool, dim=1, p=2)

                emb.append(horizontal_pool)
                emb.append(vertical_pool)

    return torch.cat(emb, dim=1)


def _local_pod_masked(
    x, mask, spp_scales=[1, 2, 4], normalize=False, normalize_per_scale=False
):
    b = x.shape[0]
    c = x.shape[1]
    w = x.shape[-1]
    emb = []

    mask = mask[:, None].repeat(1, c, 1, 1)
    x[mask] = 0.

    for scale in spp_scales:
        k = w // scale

        nb_regions = scale**2

        for i in range(scale):
            for j in range(scale):
                tensor = x[..., i * k:(i + 1) * k, j * k:(j + 1) * k]

                horizontal_pool = tensor.mean(dim=3).view(b, -1)
                vertical_pool = tensor.mean(dim=2).view(b, -1)

                if normalize_per_scale is True:
                    horizontal_pool = horizontal_pool / nb_regions
                    vertical_pool = vertical_pool / nb_regions
                elif normalize_per_scale == "spm":
                    if scale_index == 0:
                        factor = 2 ** (len(spp_scales) - 1)
                    else:
                        factor = 2 ** (len(spp_scales) - scale_index)
                if normalize:
                    horizontal_pool = F.normalize(horizontal_pool, dim=1, p=2)
                    vertical_pool = F.normalize(vertical_pool, dim=1, p=2)

                emb.append(horizontal_pool)
                emb.append(vertical_pool)

    return torch.cat(emb, dim=1)


def _global_pod(x, spp_scales=[2, 4, 8], normalize=False):
    b = x.shape[0]
    w = x.shape[-1]

    emb = []
    for scale in spp_scales:
        tensor = F.avg_pool2d(x, kernel_size=w // scale)
        horizontal_pool = tensor.sum(dim=2).view(b, -1)
        vertical_pool = tensor.sum(dim=3).view(b, -1)

        if normalize:
            horizontal_pool = F.normalize(horizontal_pool, dim=1, p=2)
            vertical_pool = F.normalize(vertical_pool, dim=1, p=2)

        tensor_pool = torch.cat([horizontal_pool, vertical_pool], dim=-1)

        emb.append(tensor_pool)

    return torch.cat(emb, dim=1)



def second_order_pooling(x,scale=8):
    k = x.shape[1] // scale
    emb = []
    for j in range(scale):
        tensor = x[:,j * k:(j + 1) * k,:,:]
        tensor=F.avg_pool2d(tensor, 5, stride=2, padding=5//2)
        ba,c,h,w=tensor.shape

        tensor=tensor.permute(0,2,3,1).contiguous().view(-1,c)
        tensor1= tensor.unsqueeze(2)
        tensor2=tensor.unsqueeze(1)

        tensor=torch.matmul(tensor1,tensor2).view(ba,-1,h,w)

        tensor=tensor.contiguous().view(ba, -1)
        # sign=torch.sign(tensor)
        # tensor=torch.abs(tensor)**0.5*sign #0.3 0.5 0.7
        # tensor = nn.functional.normalize(tensor,dim=1)
        emb.append(tensor)
    return torch.cat(emb, dim=1)
def mulglobal(x,scale=[4, 8, 12,16,20,24]):
    
    emb = []
    for i in scale:
        x_pooling = F.avg_pool2d(x, i, stride=1, padding=i//2).view(x.shape[0], -1)
        emb.append(x_pooling)
    return torch.cat(emb, dim=1)