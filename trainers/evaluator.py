# encoding: utf-8
import numpy as np
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt

from trainers.re_ranking import re_ranking as re_ranking_func

class ResNetEvaluator:
    def __init__(self, model):
        self.model = model

    def save_incorrect_pairs(self, distmat, queryloader, galleryloader, 
        g_pids, q_pids, g_camids, q_camids, savefig):
        os.makedirs(savefig, exist_ok=True)
        self.model.eval()
        m = distmat.shape[0]
        indices = np.argsort(distmat, axis=1)
        for i in range(m):
            for j in range(10):
                index = indices[i][j]
                if g_camids[index] == q_camids[i] and g_pids[index] == q_pids[i]:
                    continue
                else:
                    break
            if g_pids[index] == q_pids[i]:
                continue
            fig, axes =plt.subplots(1, 11, figsize=(12, 8))
            img = queryloader.dataset.dataset[i][0]
            img = Image.open(img).convert('RGB')
            axes[0].set_title(q_pids[i])
            axes[0].imshow(img)
            axes[0].set_axis_off()
            for j in range(10):
                gallery_index = indices[i][j]
                img = galleryloader.dataset.dataset[gallery_index][0]
                img = Image.open(img).convert('RGB')
                axes[j+1].set_title(g_pids[gallery_index])
                axes[j+1].set_axis_off()
                axes[j+1].imshow(img)
            fig.savefig(os.path.join(savefig, '%d.png' %q_pids[i]))
            plt.close(fig)

    def evaluate(self, queryloader, galleryloader, queryFliploader, galleryFliploader, 
        ranks=[1, 2, 4, 5,8, 10, 16, 20], eval_flip=False, re_ranking=False, savefig=False):
        self.model.eval()
        qf, q_pids, q_camids = [], [], []
        for inputs0, inputs1 in zip(queryloader, queryFliploader):
            inputs, pids, camids = self._parse_data(inputs0)
            feature0 = self._forward(inputs)
            if eval_flip:
                inputs, pids, camids = self._parse_data(inputs1)
                feature1 = self._forward(inputs)
                qf.append((feature0 + feature1) / 2.0)
            else:
                qf.append(feature0)

            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = torch.Tensor(q_pids)
        q_camids = torch.Tensor(q_camids)

        print("Extracted features for query set: {} x {}".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for inputs0, inputs1 in zip(galleryloader, galleryFliploader):
            inputs, pids, camids = self._parse_data(inputs0)
            feature0 = self._forward(inputs)
            if eval_flip:
                inputs, pids, camids = self._parse_data(inputs1)
                feature1 = self._forward(inputs)
                gf.append((feature0 + feature1) / 2.0)
            else:
                gf.append(feature0)
                
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = torch.Tensor(g_pids)
        g_camids = torch.Tensor(g_camids)

        print("Extracted features for gallery set: {} x {}".format(gf.size(0), gf.size(1)))

        print("Computing distance matrix")

        m, n = qf.size(0), gf.size(0)
        q_g_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
            torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        q_g_dist.addmm_(1, -2, qf, gf.t())

        if re_ranking:
            q_q_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m).t()
            q_q_dist.addmm_(1, -2, qf, qf.t())

            g_g_dist = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n) + \
                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n).t()
            g_g_dist.addmm_(1, -2, gf, gf.t())

            q_g_dist = q_g_dist.numpy()
            q_g_dist[q_g_dist < 0] = 0
            q_g_dist = np.sqrt(q_g_dist)

            q_q_dist = q_q_dist.numpy()
            q_q_dist[q_q_dist < 0] = 0
            q_q_dist = np.sqrt(q_q_dist)

            g_g_dist = g_g_dist.numpy()
            g_g_dist[g_g_dist < 0] = 0
            g_g_dist = np.sqrt(g_g_dist)

            distmat = torch.Tensor(re_ranking_func(q_g_dist, q_q_dist, g_g_dist))
        else:
            distmat = q_g_dist 

        if savefig:
            print("Saving fingure")
            self.save_incorrect_pairs(distmat.numpy(), queryloader, galleryloader, 
                g_pids.numpy(), q_pids.numpy(), g_camids.numpy(), q_camids.numpy(), savefig)

        print("Computing CMC and mAP")
        cmc, mAP = self.eval_func_gpu(distmat, q_pids, g_pids, q_camids, g_camids)

        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")

        return cmc[0]

    def _parse_data(self, inputs):
        imgs, pids, camids = inputs
        return imgs.cuda(), pids, camids

    def _forward(self, inputs):
        with torch.no_grad():
            feature = self.model(inputs)
        return feature.cpu()

    def eval_func_gpu(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        num_q, num_g = distmat.size()
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        _, indices = torch.sort(distmat, dim=1)
        matches = g_pids[indices] == q_pids.view([num_q, -1]) 
        keep = ~((g_pids[indices] == q_pids.view([num_q, -1])) & (g_camids[indices]  == q_camids.view([num_q, -1])))
        #keep = g_camids[indices]  != q_camids.view([num_q, -1])

        results = []
        num_rel = []
        for i in range(num_q):
            m = matches[i][keep[i]]
            if m.any():
                num_rel.append(m.sum())
                results.append(m[:max_rank].unsqueeze(0))
        matches = torch.cat(results, dim=0).float()
        num_rel = torch.Tensor(num_rel)

        cmc = matches.cumsum(dim=1)
        cmc[cmc > 1] = 1
        all_cmc = cmc.sum(dim=0) / cmc.size(0)

        pos = torch.Tensor(range(1, max_rank+1))
        temp_cmc = matches.cumsum(dim=1) / pos * matches
        AP = temp_cmc.sum(dim=1) / num_rel
        mAP = AP.sum() / AP.size(0)
        return all_cmc.numpy(), mAP.item()

    def eval_func(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        """Evaluation with market1501 metric
            Key: for each query identity, its gallery images from the same camera view are discarded.
            """
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        num_valid_q = 0.  # number of valid query
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        return all_cmc, mAP


class ResNetEvaluatorRBDLF:
    def __init__(self, model):
        self.model = model

    def save_incorrect_pairs(self, distmat, queryloader, galleryloader,
                             g_pids, q_pids, g_camids, q_camids, savefig):
        os.makedirs(savefig, exist_ok=True)
        self.model.eval()
        m = distmat.shape[0]
        indices = np.argsort(distmat, axis=1)
        for i in range(m):
            for j in range(10):
                index = indices[i][j]
                if g_camids[index] == q_camids[i] and g_pids[index] == q_pids[i]:
                    continue
                else:
                    break
            if g_pids[index] == q_pids[i]:
                continue
            fig, axes = plt.subplots(1, 11, figsize=(12, 8))
            img = queryloader.dataset.dataset[i][0]
            img = Image.open(img).convert('RGB')
            axes[0].set_title(q_pids[i])
            axes[0].imshow(img)
            axes[0].set_axis_off()
            for j in range(10):
                gallery_index = indices[i][j]
                img = galleryloader.dataset.dataset[gallery_index][0]
                img = Image.open(img).convert('RGB')
                axes[j + 1].set_title(g_pids[gallery_index])
                axes[j + 1].set_axis_off()
                axes[j + 1].imshow(img)
            fig.savefig(os.path.join(savefig, '%d.png' % q_pids[i]))
            plt.close(fig)

    def evaluate(self, queryloader, galleryloader, queryFliploader, galleryFliploader,
                 ranks=[1, 2, 4, 5, 8, 10, 16, 20], eval_flip=False, re_ranking=False, savefig=False):
        self.model.eval()
        qf, q_pids, q_camids = [], [], []
        qf_global = []
        qf_BFE = []
        for inputs0, inputs1 in zip(queryloader, queryFliploader):
            inputs, pids, camids = self._parse_data(inputs0)
            feature0 = self._forward(inputs)
            # global feature and bfe feature
            qf_global0 = feature0[0].cpu()
            qf_BFE0 = feature0[1].cpu()
            feature0 = torch.cat(feature0, 1).cpu()
            if eval_flip:
                inputs, pids, camids = self._parse_data(inputs1)
                feature1 = self._forward(inputs)
                qf.append((feature0 + feature1) / 2.0)
            else:
                qf.append(feature0)
                qf_global.append(qf_global0)
                qf_BFE.append(qf_BFE0)

            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        qf_global=torch.cat(qf_global, 0)
        qf_BFE=torch.cat(qf_BFE, 0)
        q_pids = torch.Tensor(q_pids)
        q_camids = torch.Tensor(q_camids)

        print("Extracted features for query set: {} x {}".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        gf_global = []
        gf_BFE = []
        for inputs0, inputs1 in zip(galleryloader, galleryFliploader):
            inputs, pids, camids = self._parse_data(inputs0)
            feature0 = self._forward(inputs)
            gf_global0=feature0[0].cpu()
            gf_BFE0=feature0[1].cpu()
            feature0 = torch.cat(feature0, 1).cpu()
            if eval_flip:
                inputs, pids, camids = self._parse_data(inputs1)
                feature1 = self._forward(inputs)
                gf.append((feature0 + feature1) / 2.0)
            else:
                gf.append(feature0)
                gf_global.append(gf_global0)
                gf_BFE.append(gf_BFE0)

            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        gf_global = torch.cat(gf_global, 0)
        gf_BFE = torch.cat(gf_BFE, 0)

        g_pids = torch.Tensor(g_pids)
        g_camids = torch.Tensor(g_camids)

        print("Extracted features for gallery set: {} x {}".format(gf.size(0), gf.size(1)))

        print("Computing distance matrix")

        m, n = qf.size(0), gf.size(0)
        q_g_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        q_g_dist.addmm_(1, -2, qf, gf.t())

        if re_ranking:
            q_q_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                       torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m).t()
            q_q_dist.addmm_(1, -2, qf, qf.t())

            g_g_dist = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n) + \
                       torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n).t()
            g_g_dist.addmm_(1, -2, gf, gf.t())

            q_g_dist = q_g_dist.numpy()
            q_g_dist[q_g_dist < 0] = 0
            q_g_dist = np.sqrt(q_g_dist)

            q_q_dist = q_q_dist.numpy()
            q_q_dist[q_q_dist < 0] = 0
            q_q_dist = np.sqrt(q_q_dist)

            g_g_dist = g_g_dist.numpy()
            g_g_dist[g_g_dist < 0] = 0
            g_g_dist = np.sqrt(g_g_dist)

            distmat = torch.Tensor(re_ranking_func(q_g_dist, q_q_dist, g_g_dist))
        else:
            distmat = q_g_dist

        if savefig:
            print("Saving fingure")
            self.save_incorrect_pairs(distmat.numpy(), queryloader, galleryloader,
                                      g_pids.numpy(), q_pids.numpy(), g_camids.numpy(), q_camids.numpy(), savefig)

        print("Computing CMC and mAP")
        cmc, mAP = self.eval_func_gpu(distmat, q_pids, g_pids, q_camids, g_camids)

        # global distmat
        q_g_dist_global=torch.pow(qf_global, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gf_global, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        q_g_dist_global.addmm_(1, -2, qf_global, gf_global.t())
        distmat_global=q_g_dist_global
        cmc_global, mAP_global = self.eval_func_gpu(distmat_global, q_pids, g_pids, q_camids, g_camids)

        # bfe distmat
        q_g_dist_bfe=torch.pow(qf_BFE, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gf_BFE, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        q_g_dist_bfe.addmm_(1, -2, qf_BFE, gf_BFE.t())
        distmat_bfe=q_g_dist_bfe
        cmc_bfe,mAP_bfe=self.eval_func_gpu(distmat_bfe, q_pids, g_pids, q_camids, g_camids)


        print("Combined Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")

        print("Global Results ----------")
        print("mAP: {:.1%}".format(mAP_global))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc_global[r - 1]))
        print("------------------")

        print("BFE Results ----------")
        print("mAP: {:.1%}".format(mAP_bfe))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc_bfe[r - 1]))
        print("------------------")

        return cmc[0]

    def _parse_data(self, inputs):
        imgs, pids, camids = inputs
        return imgs.cuda(), pids, camids

    def _forward(self, inputs):
        with torch.no_grad():
            feature = self.model(inputs)
        return feature#.cpu()

    def eval_func_gpu(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        num_q, num_g = distmat.size()
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        _, indices = torch.sort(distmat, dim=1)
        matches = g_pids[indices] == q_pids.view([num_q, -1])
        keep = ~((g_pids[indices] == q_pids.view([num_q, -1])) & (g_camids[indices] == q_camids.view([num_q, -1])))
        # keep = g_camids[indices]  != q_camids.view([num_q, -1])

        results = []
        num_rel = []
        for i in range(num_q):
            m = matches[i][keep[i]]
            if m.any():
                num_rel.append(m.sum())
                results.append(m[:max_rank].unsqueeze(0))
        matches = torch.cat(results, dim=0).float()
        num_rel = torch.Tensor(num_rel)

        cmc = matches.cumsum(dim=1)
        cmc[cmc > 1] = 1
        all_cmc = cmc.sum(dim=0) / cmc.size(0)

        pos = torch.Tensor(range(1, max_rank + 1))
        temp_cmc = matches.cumsum(dim=1) / pos * matches
        AP = temp_cmc.sum(dim=1) / num_rel
        mAP = AP.sum() / AP.size(0)
        return all_cmc.numpy(), mAP.item()

    def eval_func(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        """Evaluation with market1501 metric
            Key: for each query identity, its gallery images from the same camera view are discarded.
            """
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        num_valid_q = 0.  # number of valid query
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        return all_cmc, mAP


class ResNetEvaluatorDistiller:
    def __init__(self, model):
        self.model = model

    def save_incorrect_pairs(self, distmat, queryloader, galleryloader,
                             g_pids, q_pids, g_camids, q_camids, savefig):
        os.makedirs(savefig, exist_ok=True)
        self.model.eval()
        m = distmat.shape[0]
        indices = np.argsort(distmat, axis=1)
        for i in range(m):
            for j in range(10):
                index = indices[i][j]
                if g_camids[index] == q_camids[i] and g_pids[index] == q_pids[i]:
                    continue
                else:
                    break
            if g_pids[index] == q_pids[i]:
                continue
            fig, axes = plt.subplots(1, 11, figsize=(12, 8))
            img = queryloader.dataset.dataset[i][0]
            img = Image.open(img).convert('RGB')
            axes[0].set_title(q_pids[i])
            axes[0].imshow(img)
            axes[0].set_axis_off()
            for j in range(10):
                gallery_index = indices[i][j]
                img = galleryloader.dataset.dataset[gallery_index][0]
                img = Image.open(img).convert('RGB')
                axes[j + 1].set_title(g_pids[gallery_index])
                axes[j + 1].set_axis_off()
                axes[j + 1].imshow(img)
            fig.savefig(os.path.join(savefig, '%d.png' % q_pids[i]))
            plt.close(fig)

    def evaluate(self, queryloader, galleryloader, queryFliploader, galleryFliploader,
                 ranks=[1, 2, 4, 5, 8, 10, 16, 20], eval_flip=False, re_ranking=False, savefig=False):
        self.model.eval()
        qf, q_pids, q_camids = [], [], []
        qf_global = []
        qf_BFE = []
        # Added
        qf_hybrid=[]
        qf_RBDLF=[]

        qf_all=[]
        for inputs0, inputs1 in zip(queryloader, queryFliploader):
            inputs, pids, camids = self._parse_data(inputs0)
            feature0,feature01 = self._forward(inputs) #feature0:BDBNet features,#feature01:ResNet50-RBDLF feature
            # global feature and bfe feature
            qf_global0 = feature0[0].cpu()
            qf_BFE0 = feature0[1]#.cpu()
            qf_RBDLF0=feature01.cpu()

            feature0 = torch.cat(feature0, 1).cpu()#BDBNet features
            feature01 = torch.cat((feature01,qf_BFE0),1).cpu()#hybrid features

            qf_BFE0=qf_BFE0.cpu()

            qf_all0=torch.cat((feature0,feature01),1)

            if eval_flip:
                inputs, pids, camids = self._parse_data(inputs1)
                feature1 = self._forward(inputs)
                qf.append((feature0 + feature1) / 2.0)
            else:
                qf.append(feature0)
                qf_global.append(qf_global0)
                qf_BFE.append(qf_BFE0)
                qf_hybrid.append(feature01)
                qf_RBDLF.append(qf_RBDLF0)
                qf_all.append(qf_all0)

            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        qf_global=torch.cat(qf_global, 0)
        qf_BFE=torch.cat(qf_BFE, 0)
        qf_hybrid = torch.cat(qf_hybrid, 0)
        qf_RBDLF = torch.cat(qf_RBDLF, 0)
        qf_all = torch.cat(qf_all, 0)


        q_pids = torch.Tensor(q_pids)
        q_camids = torch.Tensor(q_camids)

        print("Extracted features for query set: {} x {}".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        gf_global = []
        gf_BFE = []
        # Added
        gf_hybrid = []
        gf_RBDLF = []

        gf_all=[]
        for inputs0, inputs1 in zip(galleryloader, galleryFliploader):
            inputs, pids, camids = self._parse_data(inputs0)
            feature0,feature01 = self._forward(inputs)
            gf_global0=feature0[0].cpu()
            gf_BFE0=feature0[1]#.cpu()
            gf_RBDLF0=feature01.cpu()

            feature0 = torch.cat(feature0, 1).cpu()
            feature01 = torch.cat((feature01,gf_BFE0),1).cpu()#hybrid features

            gf_BFE0=gf_BFE0.cpu()

            gf_all0=torch.cat((feature0,feature01),1)

            if eval_flip:
                inputs, pids, camids = self._parse_data(inputs1)
                feature1 = self._forward(inputs)
                gf.append((feature0 + feature1) / 2.0)
            else:
                gf.append(feature0)
                gf_global.append(gf_global0)
                gf_BFE.append(gf_BFE0)
                gf_hybrid.append(feature01)
                gf_RBDLF.append(gf_RBDLF0)
                gf_all.append(gf_all0)

            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        gf_global = torch.cat(gf_global, 0)
        gf_BFE = torch.cat(gf_BFE, 0)
        gf_hybrid = torch.cat(gf_hybrid, 0)
        gf_RBDLF = torch.cat(gf_RBDLF, 0)
        gf_all = torch.cat(gf_all, 0)


        g_pids = torch.Tensor(g_pids)
        g_camids = torch.Tensor(g_camids)

        print("Extracted features for gallery set: {} x {}".format(gf.size(0), gf.size(1)))

        print("Computing distance matrix")

        m, n = qf.size(0), gf.size(0)
        q_g_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        q_g_dist.addmm_(1, -2, qf, gf.t())

        if re_ranking:
            q_q_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                       torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m).t()
            q_q_dist.addmm_(1, -2, qf, qf.t())

            g_g_dist = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n) + \
                       torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n).t()
            g_g_dist.addmm_(1, -2, gf, gf.t())

            q_g_dist = q_g_dist.numpy()
            q_g_dist[q_g_dist < 0] = 0
            q_g_dist = np.sqrt(q_g_dist)

            q_q_dist = q_q_dist.numpy()
            q_q_dist[q_q_dist < 0] = 0
            q_q_dist = np.sqrt(q_q_dist)

            g_g_dist = g_g_dist.numpy()
            g_g_dist[g_g_dist < 0] = 0
            g_g_dist = np.sqrt(g_g_dist)

            distmat = torch.Tensor(re_ranking_func(q_g_dist, q_q_dist, g_g_dist))
        else:
            distmat = q_g_dist

        if savefig:
            print("Saving fingure")
            self.save_incorrect_pairs(distmat.numpy(), queryloader, galleryloader,
                                      g_pids.numpy(), q_pids.numpy(), g_camids.numpy(), q_camids.numpy(), savefig)

        print("Computing CMC and mAP")
        cmc, mAP = self.eval_func_gpu(distmat, q_pids, g_pids, q_camids, g_camids)

        # global distmat
        q_g_dist_global=torch.pow(qf_global, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gf_global, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        q_g_dist_global.addmm_(1, -2, qf_global, gf_global.t())
        distmat_global=q_g_dist_global
        cmc_global, mAP_global = self.eval_func_gpu(distmat_global, q_pids, g_pids, q_camids, g_camids)

        # bfe distmat
        q_g_dist_bfe=torch.pow(qf_BFE, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gf_BFE, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        q_g_dist_bfe.addmm_(1, -2, qf_BFE, gf_BFE.t())
        distmat_bfe=q_g_dist_bfe
        cmc_bfe,mAP_bfe=self.eval_func_gpu(distmat_bfe, q_pids, g_pids, q_camids, g_camids)

        # resnet50-RBDLF
        q_g_dist_resnet50_RBDLF=torch.pow(qf_RBDLF, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gf_RBDLF, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        q_g_dist_resnet50_RBDLF.addmm_(1, -2, qf_RBDLF, gf_RBDLF.t())
        distmat_resnet50_RBDLF=q_g_dist_resnet50_RBDLF
        cmc_resnet50_RBDLF, mAP_resnet50_RBDLF = self.eval_func_gpu(distmat_resnet50_RBDLF, q_pids, g_pids, q_camids, g_camids)

        # Hybrid results
        q_g_dist_hybrid=torch.pow(qf_hybrid, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gf_hybrid, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        q_g_dist_hybrid.addmm_(1, -2, qf_hybrid, gf_hybrid.t())
        distmat_hybrid=q_g_dist_hybrid
        cmc_hybrid, mAP_hybrid = self.eval_func_gpu(distmat_hybrid, q_pids, g_pids, q_camids, g_camids)

        # All features
        q_g_dist_all=torch.pow(qf_all, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gf_all, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        q_g_dist_all.addmm_(1, -2, qf_all, gf_all.t())
        distmat_all=q_g_dist_all
        cmc_all, mAP_all = self.eval_func_gpu(distmat_all, q_pids, g_pids, q_camids, g_camids)

        print("BDBNet Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")

        print("BDBNet Global Results ----------")
        print("mAP: {:.1%}".format(mAP_global))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc_global[r - 1]))
        print("------------------")

        print("BDBNet BFE Results ----------")
        print("mAP: {:.1%}".format(mAP_bfe))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc_bfe[r - 1]))
        print("------------------")

        print("ResNet50-RBDLF Results ----------")
        print("mAP: {:.1%}".format(mAP_resnet50_RBDLF))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc_resnet50_RBDLF[r - 1]))
        print("------------------")

        print("Hybrid Results ----------")
        print("mAP: {:.1%}".format(mAP_hybrid))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc_hybrid[r - 1]))
        print("------------------")

        print("All Results ----------")
        print("mAP: {:.1%}".format(mAP_all))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc_all[r - 1]))
        print("------------------")

        return cmc[0]

    def _parse_data(self, inputs):
        imgs, pids, camids = inputs
        return imgs.cuda(), pids, camids

    def _forward(self, inputs):
        with torch.no_grad():
            feature = self.model(inputs)
        return feature#.cpu()

    def eval_func_gpu(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        num_q, num_g = distmat.size()
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        _, indices = torch.sort(distmat, dim=1)
        matches = g_pids[indices] == q_pids.view([num_q, -1])
        keep = ~((g_pids[indices] == q_pids.view([num_q, -1])) & (g_camids[indices] == q_camids.view([num_q, -1])))
        # keep = g_camids[indices]  != q_camids.view([num_q, -1])

        results = []
        num_rel = []
        for i in range(num_q):
            m = matches[i][keep[i]]
            if m.any():
                num_rel.append(m.sum())
                results.append(m[:max_rank].unsqueeze(0))
        matches = torch.cat(results, dim=0).float()
        num_rel = torch.Tensor(num_rel)

        cmc = matches.cumsum(dim=1)
        cmc[cmc > 1] = 1
        all_cmc = cmc.sum(dim=0) / cmc.size(0)

        pos = torch.Tensor(range(1, max_rank + 1))
        temp_cmc = matches.cumsum(dim=1) / pos * matches
        AP = temp_cmc.sum(dim=1) / num_rel
        mAP = AP.sum() / AP.size(0)
        return all_cmc.numpy(), mAP.item()

    def eval_func(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        """Evaluation with market1501 metric
            Key: for each query identity, its gallery images from the same camera view are discarded.
            """
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        num_valid_q = 0.  # number of valid query
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        return all_cmc, mAP
