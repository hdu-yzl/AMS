"""Mask-based multi-modal, multi-domain CTR models for the Amazon dataset.

The Amazon benchmark contains three domains. Each sample carries an ID feature
(user/item embeddings), a text feature and an image feature. A lightweight
hypernetwork (the ``hy_*`` modules) predicts, per domain, a binary mask that
decides whether the text and/or image modality should be kept for the final
prediction. Four fusion variants (DNN1-DNN4) are provided.
"""

import torch
import torch.nn as nn

from modules.layers import MultiLayerPerceptron, FeatureEmbedding


class LBSign(torch.autograd.Function):
    """Straight-through sign estimator.

    Forward returns ``sign(x)`` (a hard 0/1 style gate after the ReLU), while the
    backward pass simply clamps the incoming gradient to ``[-1, 1]`` so that the
    non-differentiable sign operation can still be trained end-to-end.
    """

    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clamp_(-1, 1)


class CrossModalAttention(nn.Module):
    """Enhance text/image representations with ID-guided cross attention.

    Inputs:
        id_x:    (batch, proj_dim) ID representation, used as the query.
        text_x:  (batch, proj_dim) text representation.
        image_x: (batch, proj_dim) image representation.

    Returns:
        enh_text, enh_image: (batch, proj_dim) attended representations.
    """

    def __init__(self, input_dim, num_heads=1, dropout=0.1):
        super().__init__()
        self.text2id = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.image2id = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, id_x, text_x, image_x):
        q = id_x.unsqueeze(1)  # (B, 1, D)
        k_text = text_x.unsqueeze(1)  # (B, 1, D)
        k_image = image_x.unsqueeze(1)  # (B, 1, D)

        enh_text, _ = self.text2id(q, k_text, k_text)  # (B, 1, D)
        enh_image, _ = self.image2id(q, k_image, k_image)  # (B, 1, D)

        return enh_text.squeeze(1), enh_image.squeeze(1)  # (B, D)


class BasicModel(torch.nn.Module):
    """Shared backbone: feature projections and the per-domain mask network."""

    def __init__(self, opt):
        super(BasicModel, self).__init__()
        self.latent_dim = opt["latent_dim"]
        self.feature_num = opt["feat_num"]
        self.id_field_num = opt["id_field_num"]
        self.image_dim = opt["image_dim"]
        self.text_dim = opt["text_dim"]
        self.projection_dim = opt["projection_dim"]
        self.embed_dims = opt["mlp_dims"]
        self.dropout = opt["mlp_dropout"]
        self.use_bn = opt["use_bn"]

        self.embedding = FeatureEmbedding(self.feature_num + 1, self.latent_dim)
        self.hy_d_embedding = FeatureEmbedding(5, self.projection_dim)
        self.sign = LBSign.apply
        self.temp = 1  # temperature, annealed during training
        self.warm_step = 0
        self.domain_num = 3
        self.dnn_dim = self.id_field_num * self.latent_dim

        # Project every modality into a shared projection space.
        self.id_projection = torch.nn.Linear(self.dnn_dim, self.projection_dim)
        self.text_projection = torch.nn.Linear(self.text_dim, self.projection_dim)
        self.image_projection = torch.nn.Linear(self.image_dim, self.projection_dim)
        self.cross_attn = CrossModalAttention(input_dim=self.projection_dim, num_heads=1)

        # Mask hypernetwork: shared trunk + per-domain text/image output heads.
        self.hy_network = MultiLayerPerceptron(
            self.projection_dim * 4, self.embed_dims, self.dropout,
            output_layer=False, use_bn=self.use_bn,
        )
        self.hy_text_outputs = nn.ModuleList(
            [nn.Linear(self.embed_dims[-1], 1) for _ in range(self.domain_num)]
        )
        self.hy_image_outputs = nn.ModuleList(
            [nn.Linear(self.embed_dims[-1], 1) for _ in range(self.domain_num)]
        )

    def forward(self, id_feat, text, image, d, step=0):
        raise NotImplementedError

    def compute_mask(self, id_p, text_p, image_p, d):
        """Predict a (batch, 2) binary mask for the [text, image] modalities."""
        enh_text, enh_image = self.cross_attn(id_p, text_p, image_p)
        d_embedding = self.hy_d_embedding(d.type(torch.long).squeeze(1))
        x_fusion = torch.cat((id_p, enh_text, enh_image, d_embedding), dim=-1)
        hy_output = self.hy_network(x_fusion)

        # Route every sample to the output head of its own domain.
        p1 = self.hy_text_outputs[0](hy_output) * torch.eq(d, 0).type(torch.long)
        p2 = self.hy_image_outputs[0](hy_output) * torch.eq(d, 0).type(torch.long)
        for i in range(1, self.domain_num):
            p1 += self.hy_text_outputs[i](hy_output) * torch.eq(d, i).type(torch.long)
            p2 += self.hy_image_outputs[i](hy_output) * torch.eq(d, i).type(torch.long)

        p1 = torch.sigmoid(self.temp * p1)
        p2 = torch.sigmoid(self.temp * p2)
        p = torch.cat([p1, p2], dim=-1)

        # Straight-through binarization around the threshold ``thre``.
        mask = self.sign(torch.relu(p - self.thre))
        return mask


class DNN1(BasicModel):
    """Fusion by concatenation of (id, masked text, masked image)."""

    def __init__(self, opt):
        super(DNN1, self).__init__(opt)
        dropout = 0
        use_bn = False
        self.dnn_dim = self.id_field_num * self.latent_dim
        self.dnn1 = MultiLayerPerceptron(self.projection_dim * 3, self.embed_dims, dropout, use_bn=use_bn)
        self.dnn2 = MultiLayerPerceptron(self.projection_dim * 3, self.embed_dims, dropout, use_bn=use_bn)
        self.dnn3 = MultiLayerPerceptron(self.projection_dim * 3, self.embed_dims, dropout, use_bn=use_bn)

    def forward(self, id_feat, text, image, d, step=0):
        x_embedding = self.embedding(id_feat)
        x_ = x_embedding.view(-1, self.dnn_dim)
        id_p = self.id_projection(x_)
        text_p = self.text_projection(text)
        image_p = self.image_projection(image)

        # The mask network is trained on detached features so it does not
        # back-propagate into the prediction backbone.
        mask = self.compute_mask(id_p.detach(), text_p.detach(), image_p.detach(), d)
        if step < self.warm_step:
            mask = torch.zeros_like(mask).detach()

        mask_text = mask[:, 0].unsqueeze(-1)
        mask_image = mask[:, 1].unsqueeze(-1)

        x_dnn = torch.cat((id_p, text_p * mask_text, image_p * mask_image), dim=1)

        logit1 = self.dnn1(x_dnn)
        logit2 = self.dnn2(x_dnn)
        logit3 = self.dnn3(x_dnn)
        return logit1, logit2, logit3, mask


class DNN2(BasicModel):
    """Fusion by per-modality gated attention."""

    def __init__(self, opt):
        super(DNN2, self).__init__(opt)
        dropout = 0
        use_bn = False
        self.dnn_dim = self.id_field_num * self.latent_dim
        self.dnn1 = MultiLayerPerceptron(self.projection_dim, self.embed_dims, dropout, use_bn=use_bn)
        self.dnn2 = MultiLayerPerceptron(self.projection_dim, self.embed_dims, dropout, use_bn=use_bn)
        self.dnn3 = MultiLayerPerceptron(self.projection_dim, self.embed_dims, dropout, use_bn=use_bn)

        self.W = nn.ParameterDict(
            {m: nn.Parameter(torch.empty(self.projection_dim, self.projection_dim)) for m in ["id", "text", "image"]}
        )
        self.b = nn.ParameterDict(
            {m: nn.Parameter(torch.empty(self.projection_dim)) for m in ["id", "text", "image"]}
        )
        for m in ["id", "text", "image"]:
            nn.init.xavier_uniform_(self.W[m])
            nn.init.zeros_(self.b[m])

    def forward(self, id_feat, text, image, d, step=0):
        x_embedding = self.embedding(id_feat)
        x_ = x_embedding.view(-1, self.dnn_dim)
        id_p = self.id_projection(x_)
        text_p = self.text_projection(text)
        image_p = self.image_projection(image)

        mask = self.compute_mask(id_p.detach(), text_p.detach(), image_p.detach(), d)
        if step < self.warm_step:
            mask = torch.zeros_like(mask).detach()

        mask_text = mask[:, 0].unsqueeze(-1)
        mask_image = mask[:, 1].unsqueeze(-1)
        text_m = text_p * mask_text
        image_m = image_p * mask_image

        alpha_id = torch.tanh(id_p @ self.W["id"] + self.b["id"])
        alpha_text = torch.tanh(text_m @ self.W["text"] + self.b["text"])
        alpha_img = torch.tanh(image_m @ self.W["image"] + self.b["image"])
        x_dnn = alpha_id * id_p + alpha_text * text_m + alpha_img * image_m

        logit1 = self.dnn1(x_dnn)
        logit2 = self.dnn2(x_dnn)
        logit3 = self.dnn3(x_dnn)
        return logit1, logit2, logit3, mask


class DNN3(BasicModel):
    """Fusion by a softmax gate over (id, masked text, masked image)."""

    def __init__(self, opt):
        super(DNN3, self).__init__(opt)
        dropout = 0
        use_bn = False
        self.dnn_dim = self.id_field_num * self.latent_dim
        self.dnn1 = MultiLayerPerceptron(self.projection_dim, self.embed_dims, dropout, use_bn=use_bn)
        self.dnn2 = MultiLayerPerceptron(self.projection_dim, self.embed_dims, dropout, use_bn=use_bn)
        self.dnn3 = MultiLayerPerceptron(self.projection_dim, self.embed_dims, dropout, use_bn=use_bn)
        self.gate = torch.nn.Linear(self.projection_dim * 3, 3)

    def forward(self, id_feat, text, image, d, step=0):
        x_embedding = self.embedding(id_feat)
        x_ = x_embedding.view(-1, self.dnn_dim)
        id_p = self.id_projection(x_)
        text_p = self.text_projection(text)
        image_p = self.image_projection(image)

        mask = self.compute_mask(id_p.detach(), text_p.detach(), image_p.detach(), d)
        if step < self.warm_step:
            mask = torch.zeros_like(mask).detach()

        mask_text = mask[:, 0].unsqueeze(-1)
        mask_image = mask[:, 1].unsqueeze(-1)
        text_m = text_p * mask_text
        image_m = image_p * mask_image

        x_cat = torch.cat((id_p, text_m, image_m), dim=1)
        gate = torch.softmax(self.gate(x_cat), dim=-1)
        x_dnn = (
            id_p * gate[:, 0].unsqueeze(-1)
            + text_m * gate[:, 1].unsqueeze(-1)
            + image_m * gate[:, 2].unsqueeze(-1)
        )

        logit1 = self.dnn1(x_dnn)
        logit2 = self.dnn2(x_dnn)
        logit3 = self.dnn3(x_dnn)
        return logit1, logit2, logit3, mask


class DNN4(BasicModel):
    """Fusion by a Transformer encoder over the three modality tokens."""

    def __init__(self, opt):
        super(DNN4, self).__init__(opt)
        dropout = 0
        use_bn = False
        self.dnn_dim = self.id_field_num * self.latent_dim
        self.dnn1 = MultiLayerPerceptron(self.projection_dim, self.embed_dims, dropout, use_bn=use_bn)
        self.dnn2 = MultiLayerPerceptron(self.projection_dim, self.embed_dims, dropout, use_bn=use_bn)
        self.dnn3 = MultiLayerPerceptron(self.projection_dim, self.embed_dims, dropout, use_bn=use_bn)
        self.transformer = nn.TransformerEncoderLayer(d_model=self.projection_dim, nhead=4, batch_first=True)

    def forward(self, id_feat, text, image, d, step=0):
        x_embedding = self.embedding(id_feat)
        x_ = x_embedding.view(-1, self.dnn_dim)
        id_p = self.id_projection(x_)
        text_p = self.text_projection(text)
        image_p = self.image_projection(image)

        mask = self.compute_mask(id_p.detach(), text_p.detach(), image_p.detach(), d)
        if step < self.warm_step:
            mask = torch.zeros_like(mask).detach()

        mask_text = mask[:, 0].unsqueeze(-1)
        mask_image = mask[:, 1].unsqueeze(-1)
        text_m = text_p * mask_text
        image_m = image_p * mask_image

        tokens = torch.stack([id_p, text_m, image_m], dim=1)  # (batch, 3, dim)
        fused = self.transformer(tokens)  # (batch, 3, dim)
        x_dnn = fused.mean(dim=1)

        logit1 = self.dnn1(x_dnn)
        logit2 = self.dnn2(x_dnn)
        logit3 = self.dnn3(x_dnn)
        return logit1, logit2, logit3, mask


def getModel(model: str, opt):
    """Factory that maps a model name to its implementation."""
    model = model.lower()
    if model == "dnn1":
        return DNN1(opt)
    elif model == "dnn2":
        return DNN2(opt)
    elif model == "dnn3":
        return DNN3(opt)
    elif model == "dnn4":
        return DNN4(opt)
    else:
        raise ValueError("Invalid model type: {}".format(model))


def getOptim(network, optim, lr, m_lr, l2):
    """Build two optimizers: one for the backbone, one for the mask network.

    Parameters whose name contains ``hy`` belong to the mask hypernetwork and
    use the (typically smaller) mask learning rate ``m_lr``.
    """
    weight_params = map(
        lambda a: a[1],
        filter(lambda p: p[1].requires_grad and "hy" not in p[0], network.named_parameters()),
    )
    mask_params = map(
        lambda a: a[1],
        filter(lambda p: p[1].requires_grad and "hy" in p[0], network.named_parameters()),
    )

    optim = optim.lower()
    if optim == "sgd":
        return [torch.optim.SGD(weight_params, lr=lr, weight_decay=l2), torch.optim.SGD(mask_params, lr=m_lr)]
    elif optim == "adam":
        return [torch.optim.Adam(weight_params, lr=lr, weight_decay=l2), torch.optim.Adam(mask_params, lr=m_lr)]
    else:
        raise ValueError("Invalid optimizer type: {}".format(optim))
