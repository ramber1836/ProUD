from turtle import forward
import paddle.nn as nn
import paddle
import paddle.nn.functional as F

class MyModel(nn.Layer):
    def __init__(self, args):
        super().__init__()
        self.feat_size = args.feat_size
        nn.initializer.set_global_initializer(nn.initializer.XavierUniform())
        self.based_embed = nn.Embedding(args.item_size, args.dim)
        self.dropout = nn.Dropout(p=args.dropout)
        self.dense_one = nn.Linear(2 * args.dim, args.dim)
        self.dense_two = nn.Linear(args.dim, args.dim)
        self.dense_three = nn.Linear(args.dim, 1, bias_attr=False)
        self.args = args
    
    def _sparsemax(self, logits):
        obs = logits.shape[0]
        dims = logits.shape[1]

        z = logits

        z_sorted, _ = paddle.topk(z, k=dims)

        z_cumsum = paddle.cumsum(z_sorted, axis=1)
        k = paddle.arange(1, dims+1)
        z_check = 1 + k * z_sorted > z_cumsum
        k_z = paddle.sum(z_check, axis=1)

        k_z_safe = paddle.maximum(k_z, paddle.to_tensor(1))
        indices = paddle.stack([paddle.arange(0, obs), k_z_safe-1], axis=1)
        tau_sum = paddle.gather_nd(z_cumsum, indices)
        tau_z = (tau_sum - 1) / k_z

        p = paddle.maximum(paddle.to_tensor(0, dtype=z.dtype), z - tau_z.unsqueeze(1))
        p_safe = paddle.where(
            paddle.logical_or(
                paddle.equal(k_z, 0), paddle.isnan(z_cumsum[:, -1])).unsqueeze(1).expand([-1, logits.shape[-1]]),
            paddle.full([obs, dims], paddle.to_tensor(float("nan"), dtype=logits.dtype)),
            p)
        
        return p_safe

    def forward(self, feat, label):
        # feat: B x L
        feat_size = feat.shape[1]
        feat_lookup = self.based_embed(feat) # B x L x D
        mask = paddle.not_equal(feat, paddle.to_tensor(0).cast(feat.dtype)) # B x L
        mask_BF1 = mask.unsqueeze(2) # B x L x 1
        mask_BFF = mask.unsqueeze(1).expand([-1, feat_size, -1]) # B x L x L

        expand_left = feat_lookup.unsqueeze(2).expand([-1, -1, feat_size, -1]) # B x L x L x D
        expand_right = feat_lookup.unsqueeze(1).expand([-1, feat_size, -1, -1]) # B x L x L x D
        norm_left = paddle.maximum(paddle.norm(expand_left, p=2, axis=-1), paddle.to_tensor(1e-6)) # B x L x L
        proj_weight = paddle.sum(expand_left * expand_right, axis=-1) / norm_left # B x L x L
        proj_weight = F.softmax(proj_weight, axis=-1) * mask_BFF # B x L x L
        proj_weight /= paddle.sum(proj_weight, axis=-1, keepdim=True) # B x L x L
        proj_weight = proj_weight.unsqueeze(3) # B x L x L x 1
        feat_weighted = paddle.sum(expand_right * proj_weight, axis=-2) # B x L x D

        feat_likelihood = paddle.concat([feat_lookup, feat_weighted], axis=-1) # B x L x 2D
        likelihood = self.dropout(F.relu(self.dense_one(feat_likelihood)))
        likelihood = self.dropout(F.relu(self.dense_two(likelihood)))
        likelihood = self.dense_three(likelihood) # B x L x 1

        likelihood_for_reg = likelihood * mask_BF1
        likelihood += (mask_BF1 - 1.) * 1e4
        likelihood = self._sparsemax(likelihood.squeeze(2)).unsqueeze(2)
        direction = paddle.sum(feat_lookup * likelihood, axis=-2) # B x D
        norm_direction = paddle.maximum(paddle.norm(direction, p=2, axis=-1), paddle.to_tensor(1e-6)) # B
        feat_sum = paddle.sum(feat_lookup * mask_BF1, axis=-2) # B x D
        
        pred_proj = paddle.sum(feat_sum * direction, axis=-1) / norm_direction # B

        # train
        label_flt = 2. * label.cast(dtype=paddle.float32) - 1
        proj_loss = paddle.mean(-paddle.log(F.sigmoid(label_flt * pred_proj)))
        l1_loss = self.args.likelihood_reg * paddle.sum(likelihood_for_reg ** 2) / 2

        loss = proj_loss + l1_loss
        final_pred = 1. / (1. + paddle.exp(- pred_proj))
        return loss, final_pred
        


