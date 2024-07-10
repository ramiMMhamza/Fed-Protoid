# Ge et al. Self-paced Contrastive Learning with Hybrid Memory for Domain Adaptive Object Re-ID.  # noqa
# Written by Yixiao Ge.

import torch
import torch.nn.functional as F
from torch import autograd, nn

from ...utils.dist_utils import all_gather_tensor

try:
    # PyTorch >= 1.6 supports mixed precision training
    from torch.cuda.amp import custom_fwd, custom_bwd
    class HM(autograd.Function):

        @staticmethod
        @custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, inputs, indexes, features, features_protos, momentum):
            ctx.features = features
            ctx.momentum = momentum
            if features_protos is not None:
                all_features = torch.cat([ctx.features,features_protos])
                ctx.all_features = all_features
                outputs = inputs.mm(all_features.t())
            else:
                ctx.all_features = None
                outputs = inputs.mm(ctx.features.t())
            all_inputs = all_gather_tensor(inputs)
            all_indexes = all_gather_tensor(indexes)
            ctx.save_for_backward(all_inputs, all_indexes)
            return outputs

        @staticmethod
        @custom_bwd
        def backward(ctx, grad_outputs):
            inputs, indexes = ctx.saved_tensors
            grad_inputs = None
            if ctx.needs_input_grad[0]:
                if ctx.all_features is not None:
                    grad_inputs = grad_outputs.mm(ctx.all_features)
                else:
                    grad_inputs = grad_outputs.mm(ctx.features)

            # momentum update
            for x, y in zip(inputs, indexes):
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

            return grad_inputs, None, None, None, None
except:
    class HM(autograd.Function):

        @staticmethod
        def forward(ctx, inputs, indexes, features, features_protos, momentum):
            ctx.features = features
            ctx.momentum = momentum
            # outputs = inputs.mm(ctx.features.t())
            if features_protos is not None:
                all_features = torch.cat([ctx.features,features_protos])
                ctx.all_features = all_features
                outputs = inputs.mm(all_features.t())
            else:
                ctx.all_features = None
                outputs = inputs.mm(ctx.features.t())
            all_inputs = all_gather_tensor(inputs)
            all_indexes = all_gather_tensor(indexes)
            ctx.save_for_backward(all_inputs, all_indexes)
            return outputs

        @staticmethod
        def backward(ctx, grad_outputs):
            inputs, indexes = ctx.saved_tensors
            grad_inputs = None
            if ctx.needs_input_grad[0]:
                if ctx.all_features is not None:
                    grad_inputs = grad_outputs.mm(ctx.all_features)
                else:
                    grad_inputs = grad_outputs.mm(ctx.features)

            # momentum update
            for x, y in zip(inputs, indexes):
                ctx.features[y] = ctx.momentum * ctx.features[y] + (1.0 - ctx.momentum) * x
                ctx.features[y] /= ctx.features[y].norm()

            return grad_inputs, None, None, None, None


def hm(inputs, indexes, features, features_protos, momentum=0.5):
    return HM.apply(
        inputs, indexes, features, features_protos, torch.Tensor([momentum]).to(inputs.device)
    )

class HybridMemory(nn.Module):
    def __init__(self, num_features, num_memory, num_memory_target, num_memory_source, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_memory = num_memory
        self.num_memory_target = num_memory_target
        self.num_memory_source = num_memory_source

        self.momentum = momentum
        self.temp = temp

        self.register_buffer("features", torch.zeros(num_memory, num_features))
        self.register_buffer("memory_features_protos", torch.zeros(num_memory, num_features))
        self.register_buffer("features_target", torch.zeros(num_memory_target, num_features))
        self.register_buffer("features_source", torch.zeros(num_memory_source, num_features))
        self.register_buffer("labels", torch.zeros(num_memory).long())
        self.register_buffer("labels_protos", torch.zeros(num_memory_target).long())
        self.register_buffer("labels_source", torch.zeros(num_memory_source).long())

    @torch.no_grad()
    def _update_feature(self, features,memory_features_protos=None):
        features = F.normalize(features, p=2, dim=1)
        self.features.data.copy_(features.float().to(self.features.device))
        if memory_features_protos is not None:
            memory_features_protos = F.normalize(memory_features_protos, p=2, dim=1)
            self.memory_features_protos = memory_features_protos.float().to(self.memory_features_protos.device)
        else:
            self.memory_features_protos = memory_features_protos

    @torch.no_grad()
    def _update_label(self, labels, labels_protos=None):
        # self.labels.data.copy_(labels.long().to(self.labels.device))
        if labels_protos is not None:
            labels = torch.cat([labels, torch.tensor(labels_protos)])
            self.labels = labels.long().to(self.labels.device)
            self.labels_protos = labels_protos.long().to(self.labels_protos.device)
        else:
            self.labels = labels.long().to(self.labels.device)
            self.labels_protos = labels_protos
        # self.labels_target.data.copy_(labels_target.long().to(self.labels_target.device))
        # self.labels_source.data.copy_(labels_source.long().to(self.labels_source.device))

    def forward(self, results, indexes):
        inputs = results["feat"]
        inputs = F.normalize(inputs, p=2, dim=1)
        # inputs: B*2048, features: L*2048
        inputs = hm(inputs, indexes, self.features,self.memory_features_protos, self.momentum)
        B = inputs.size(0)
        inputs /= self.temp
        # inputs[int(B/2):,:] /= self.temp #Taus 0.05
        # inputs[:int(B/2),:] /= 0.1 #Taut
        def masked_softmax(vec, mask, coef,  dim=1, epsilon=1e-6,):
            exps = torch.exp(vec)
            if coef is not None :
                masked_exps = (exps * mask.float().clone()).clone()
                masked_exps_weighted = masked_exps*coef
                # masked_exps_target = (exps*coef_target.t()).clone()
                # masked_exps_source = (exps*coef_source.t()).clone()
                masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
                
                return masked_exps_weighted/masked_sums
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return masked_exps / masked_sums #masked_exps / masked_sums

        # len_ = int(len(results["feat"])/2) 
        # nt = self.labels_target.max()+1 #self.num_memory_target
        # indexes_target, indexes_source = indexes[:len_], indexes[len_:]
        targets = self.labels[indexes].clone()
        # targets_target = self.labels_target[indexes_target].clone()
        # targets_source = self.labels_source[indexes_source-indexes_source.min()].clone()
        # if self.labels_protos is not None:
        #     labels_protos = 
        labels = self.labels.clone()
        sim = torch.zeros(labels.max() + 1, B).float().cuda()
        sim.index_add_(0, labels, inputs.t().contiguous())
        nums = torch.zeros(labels.max() + 1, 1).float().cuda()
        # nums.index_add_(0, labels, torch.ones(self.num_memory, 1).float().cuda())
        nums.index_add_(0, labels, torch.ones(len(labels), 1).float().cuda())
        
        mask = (nums > 0).float()
        # torch.save(results["feat"], "data_exp/data/features.pt")
        # torch.save(self.features, "data_exp/data/all_features.pt")
        # torch.save(indexes, "data_exp/data/indexes.pt")
        # torch.save(mask, "data_exp/data/mask.pt")
        # torch.save(nums, "data_exp/data/nums.pt")
        # torch.save(labels, "data_exp/data/labels2.pt")
        # torch.save(targets, "data_exp/data/targets.pt")
        # torch.save(inputs, "data_exp/data/inputs.pt")
        # torch.save(sim, "data_exp/data/sim2.pt")

        # sys.exit()
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim)
        mask = mask.expand_as(sim)
        ################### For weighted Loss ###################
        a, b = sim.size()
        coef = torch.zeros((a,b)).cuda()
        coef[:,:int(b/2)]= 3/4 #2/3 #Ct
        coef[:,int(b/2):]= 1/4 #1/3 #Cs
        # test_target = torch.zeros((a,b)).cuda()
        # test_target[:nt,:int(b/2)]= 1
        # test_source = torch.zeros((a,b)).cuda()
        # test_source[:,:int(b/2)]= 0
        # test_source[nt:,int(b/2):]= 1
        ########################################################
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous(), coef = None) # coef.t().contiguous() , coef_target = test_target, coef_source = test_source)
        # masked_sim = masked_sim_target+masked_sim_source
        # return F.nll_loss(torch.log(masked_sim_target + 1e-6), targets_target) + F.nll_loss(torch.log(masked_sim_source + 1e-6), targets_source) #
        return F.nll_loss(torch.log(masked_sim + 1e-6), targets)
#     def forward(self, results, indexes):
# #         # return F.nll_loss(torch.log(masked_sim + 1e-6), targets)
#         # len_ = int(len(results["feat"])/2) 
#         # inputs_target, inputs_source = results["feat"][:len_], results["feat"][len_:]
#         # indexes_target, indexes_source = indexes[:len_], indexes[len_:]
#         # inputs_target = F.normalize(inputs_target, p=2, dim=1)
#         # inputs_source = F.normalize(inputs_source, p=2, dim=1)
#         # # inputs: B*2048, features: L*2048
#         # inputs_target = hm(inputs_target, indexes_target, self.features, self.momentum)
#         # inputs_target /= self.temp
#         # inputs_source = hm(inputs_source, indexes_source, self.features_source, self.momentum)
#         # inputs_source /= self.temp
#         # B_target = inputs_target.size(0)
#         # B_source = inputs_source.size(0)

#         def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
#             exps = torch.exp(vec)
#             masked_exps = exps * mask.float().clone()
#             masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
#             return masked_exps / masked_sums

#         # targets_target = self.labels_target[indexes_target].clone()
#         # labels_target = self.labels_target.clone()
#         # targets_source = self.labels_source[indexes_source] #.clone()
#         # labels_source = self.labels_source #.clone()

#         # sim_target = torch.zeros(int(labels_target.max()) + 1, B_target).float().cuda()
#         # sim_target.index_add_(0, labels_target, inputs_target.t().contiguous())
#         # nums_target = torch.zeros(int(labels_target.max()) + 1, 1).float().cuda()
#         # nums_target.index_add_(0, labels_target, torch.ones(self.num_memory_target, 1).float().cuda())
#         # mask_target = (nums_target > 0).float()
#         # sim_target /= (mask_target * nums_target + (1 - mask_target)).clone().expand_as(sim_target)
#         # mask_target = mask_target.expand_as(sim_target)
#         # masked_sim_target = masked_softmax(sim_target.t().contiguous(), mask_target.t().contiguous())

#         # sim_source = torch.zeros(int(self.labels_source.max()) + 1, B_source).float().cuda()
#         # sim_source.index_add_(0, self.labels_source, inputs_source.t().contiguous())
#         # nums_source = torch.zeros(int(self.labels_source.max()) + 1, 1).float().cuda()
#         # nums_source.index_add_(0, self.labels_source, torch.ones(self.num_memory_source, 1).float().cuda())
#         # mask_source = (nums_source > 0).float()
#         # sim_source /= (mask_source * nums_source + (1 - mask_source)).clone().expand_as(sim_source)
#         # mask_source = mask_source.expand_as(sim_source)
#         # masked_sim_source = masked_softmax(sim_source.t().contiguous(), mask_source.t().contiguous())
        
        
#         inputs = results #"feat"]
#         inputs = F.normalize(inputs, p=2, dim=1)

#         # inputs: B*2048, features: L*2048
#         inputs = hm(inputs, indexes, self.features, self.momentum)
#         inputs /= self.temp
#         B = inputs.size(0)
#         targets = self.labels[indexes].clone()
#         labels = self.labels.clone()

#         sim = torch.zeros(labels.max() + 1, B).float().cuda()
#         sim.index_add_(0, labels, inputs.t().contiguous())
#         nums = torch.zeros(labels.max() + 1, 1).float().cuda()
#         nums.index_add_(0, labels, torch.ones(self.num_memory, 1).float().cuda())
#         mask = (nums > 0).float()
#         sim /= (mask * nums + (1 - mask)).clone().expand_as(sim)
#         mask = mask.expand_as(sim)
#         masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())
#         return F.nll_loss(torch.log(masked_sim+ 1e-6), targets) #F.nll_loss(torch.log(masked_sim_target + 1e-6), targets_target)#, F.nll_loss(torch.log(masked_sim_source + 1e-6), self.labels_source[indexes_source])
# # # # F.nll_loss(torch.log(masked_sim_target + 1e-6), targets_target)#, 