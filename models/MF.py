import torch
from torch import nn


class TrendEmbedding(nn.Module):
    def __init__(self, shop_emb_size=10, trend_emb_size=10):
        super(TrendEmbedding, self).__init__()
        self.embedding_size = trend_emb_size
        self.rnn = nn.GRU(input_size=shop_emb_size, hidden_size=trend_emb_size, bidirectional=False)

    def forward(self, previous_shop_batch, hidden):
        _, trend_emb = self.rnn(previous_shop_batch, hidden)
        return trend_emb

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.embedding_size,
                           device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


class RNNMatrixFactorization(nn.Module):
    def __init__(self, n_user, n_shop, embed_size=10, hidden_feature=(16, 8)):
        super(RNNMatrixFactorization, self).__init__()
        self.embed_size = embed_size
        self.user_encoder = nn.Embedding(n_user, embed_size)
        self.shop_encoder = nn.Embedding(n_shop, embed_size)
        self.trend_encoder = TrendEmbedding(shop_emb_size=embed_size)
        self.scorer = nn.Sequential(
            nn.Linear(in_features=embed_size * 3, out_features=hidden_feature[0]),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_feature[0], out_features=hidden_feature[1]),
            nn.LeakyReLU(),
            nn.Linear(in_features=hidden_feature[1], out_features=1),
            nn.ReLU()
        )

    def forward(self, hours, user_batch, shop_batch, previous_shop_batch):
        batch_size = user_batch.shape[1]
        user_emb = self.user_encoder(user_batch).view(-1, batch_size, self.embed_size)
        shop_emb = self.shop_encoder(shop_batch).view(-1, batch_size, self.embed_size)
        previous_shop_embs = self.shop_encoder(previous_shop_batch).view(-1, batch_size, self.embed_size)

        hidden = self.trend_encoder.init_hidden(batch_size)
        trend_emb = self.trend_encoder(previous_shop_embs, hidden)
        features = torch.cat([hours, user_emb, shop_emb, trend_emb], dim=2)
        score = self.scorer(features)
        return score


if __name__ == '__main__':
    model = RNNMatrixFactorization(20, 10)
    user_batch_sample = torch.randint(0, 20, size=(1, 8, 1))  # n_user, batch_size, embedding_size
    shop_batch_sample = torch.randint(0, 10, size=(1, 8, 1))
    previous_shop_batch_sample = torch.randint(0, 10, size=(5, 8, 1))
    score = model(user_batch_sample, shop_batch_sample, previous_shop_batch_sample)
    print(score.shape)  # size = user_batch_size is the expectation