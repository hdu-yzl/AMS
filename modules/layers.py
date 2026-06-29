"""Reusable neural network building blocks (embeddings, MLP, FM, cross net)."""

import torch
import torch.nn.functional as F
import torch.nn as nn


class MAF(torch.nn.Module):
    def __init__(self, projection_dim):
        super(MAF, self).__init__()
        self.projection_dim = projection_dim

        self.W = nn.ParameterDict({
            m: nn.Parameter(torch.empty(self.projection_dim, self.projection_dim))
            for m in ['id', 'text', 'image', 'audio']
        })
        self.b = nn.ParameterDict({
            m: nn.Parameter(torch.empty(self.projection_dim))
            for m in ['id', 'text', 'image', 'audio']
        })

        for m in ['id', 'text', 'image', 'audio']:
            nn.init.xavier_uniform_(self.W[m])
            nn.init.zeros_(self.b[m])

    def forward(self, id, text, image, audio=None):
        alpha_id = torch.sigmoid(id @ self.W['id'] + self.b['id'])
        alpha_text = torch.sigmoid(text @ self.W['text'] + self.b['text'])
        alpha_img = torch.sigmoid(image @ self.W['image'] + self.b['image'])

        x_fusion = alpha_id * id + alpha_text * text + alpha_img * image
        if audio:
            alpha_audio = torch.sigmoid(audio @ self.W['audio'] + self.b['audio'])
            x_fusion += alpha_audio * audio

        return x_fusion

class FeatureEmbedding(torch.nn.Module):
    def __init__(self, feature_num, latent_dim, initializer = torch.nn.init.xavier_uniform_):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.zeros(feature_num, latent_dim))
        initializer(self.embedding)
    
    def forward(self, x):
        """
        :param x: tensor of size (batch_size, num_fields)
        :return: tensor of size (batch_size, num_fields, embedding_dim)
        """
        return F.embedding(x, self.embedding)

class FeaturesLinear(torch.nn.Module):
    def __init__(self, feature_num, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(feature_num, output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :return : tensor of size (batch_size, 1)
        """
        return torch.sum(torch.squeeze(self.fc(x)), dim=1, keepdim=True) + self.bias

class FactorizationMachine(torch.nn.Module):
    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        :return : tensor of size (batch_size, 1) if reduce_sum
                  tensor of size (batch_size, embed_dim) else   
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, mlp_dims, dropout, output_layer=True, use_bn=False, use_ln=False):
        super().__init__()
        layers = list()
        for mlp_dim in mlp_dims:
            layers.append(torch.nn.Linear(input_dim, mlp_dim))
            if use_bn:
                layers.append(torch.nn.BatchNorm1d(mlp_dim))
            if use_ln:
                layers.append(torch.nn.LayerNorm(mlp_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = mlp_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        :return : tensor of size (batch_size, mlp_dims[-1])
        """
        return self.mlp(x)

class Linear_softmax(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout, use_bn=False, use_ln=False):
        super().__init__()
        layers = list()
        layers.append(torch.nn.Linear(input_dim, output_dim))
        if use_bn:
            layers.append(torch.nn.BatchNorm1d(output_dim))
        if use_ln:
            layers.append(torch.nn.LayerNorm(output_dim))
        layers.append(torch.nn.Softmax(dim=1))
        layers.append(torch.nn.Dropout(p=dropout))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, input_dim, output_dim)``
        :return : tensor of size (batch_size, output_dim)
        """
        return self.mlp(x)


class CrossNetwork(torch.nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x


class InnerProduct(torch.nn.Module):
    def __init__(self, field_num):
        super().__init__()
        self.rows = []
        self.cols = []
        for row in range(field_num):
            for col in range(row+1, field_num):
                self.rows.append(row)
                self.cols.append(col)
        self.rows = torch.tensor(self.rows)
        self.cols = torch.tensor(self.cols)

    def forward(self, x):
        """
        :param x: Float tensor of size (batch_size, field_num, embedding_dim)
        :return: (batch_size, field_num*(field_num-1)/2)
        """
        batch_size = x.shape[0]
        trans_x = torch.transpose(x, 1, 2)

        self.rows = self.rows.to(trans_x.device)
        self.cols = self.cols.to(trans_x.device)

        gather_rows = torch.gather(trans_x, 2, self.rows.expand(batch_size, trans_x.shape[1], self.rows.shape[0]))
        gather_cols = torch.gather(trans_x, 2, self.cols.expand(batch_size, trans_x.shape[1], self.rows.shape[0]))
        p = torch.transpose(gather_rows, 1, 2)
        q = torch.transpose(gather_cols, 1, 2)
        product_embedding = torch.mul(p, q)
        product_embedding = torch.sum(product_embedding, 2)
        return product_embedding
