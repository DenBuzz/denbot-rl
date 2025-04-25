from typing import Any

import torch
import torch.nn as nn
from ray.rllib.core import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models.torch.torch_distributions import TorchMultiCategorical
from ray.rllib.utils import override


class DenBot(TorchRLModule, ValueFunctionAPI):
    @override(RLModule)
    def setup(self):
        super().setup()
        model_configs = self.config.model_config_dict

        reward_embedding = model_configs.get("reward_embedding", 8)
        pad_embedding = model_configs.get("pad_embedding", 8)
        unit_embedding = model_configs.get("unit_embedding", 64)

        self._reward_encoder = nn.Sequential(
            nn.Linear(
                in_features=self.observation_space["rewards"].shape[0],
                out_features=reward_embedding,
            ),
            nn.ReLU(),
        )
        self._pad_encoder = nn.Sequential(
            nn.Linear(
                in_features=self.observation_space["pads"].shape[0],
                out_features=pad_embedding,
            ),
            nn.ReLU(),
        )
        self._ball_encoder = nn.Linear(
            in_features=self.observation_space["ball"].shape[0],
            out_features=unit_embedding,
        )
        self._car_encoder = nn.Linear(
            in_features=self.observation_space["agent"].shape[0],
            out_features=unit_embedding,
        )
        self._mha = nn.MultiheadAttention(
            embed_dim=unit_embedding,
            num_heads=model_configs.get("num_heads", 8),
            batch_first=True,
        )
        self._pi = nn.Linear(
            in_features=reward_embedding + pad_embedding + 2 * unit_embedding,
            out_features=self.action_space.nvec.sum(),
        )
        self._vf = nn.Linear(
            in_features=reward_embedding + pad_embedding + 2 * unit_embedding,
            out_features=1,
        )
        self.action_dist_cls = TorchMultiCategorical.get_partial_dist_cls(input_lens=tuple(self.action_space.nvec))

    def _compute_embeddings(self, batch: dict[str, Any]) -> torch.Tensor:
        obs = batch[Columns.OBS]
        reward_embedding = self._reward_encoder(obs["rewards"])
        pad_embedding = self._pad_encoder(obs["pads"])
        ball_embedding = self._ball_encoder(obs["ball"])
        car_embedding = self._car_encoder(obs["agent"])

        qkv = torch.cat((car_embedding.unsqueeze(1), ball_embedding.unsqueeze(1)), dim=1)

        unit_embeddings, _ = self._mha(qkv, qkv, qkv)
        unit_embeddings = torch.flatten(unit_embeddings, start_dim=1)

        embeddings = torch.cat((reward_embedding, pad_embedding, unit_embeddings), dim=-1)
        return embeddings

    @override(RLModule)
    def _forward(self, batch: dict[str, Any], **kwargs) -> dict[str, Any]:
        embeddings = self._compute_embeddings(batch)
        logits = self._pi(embeddings)
        return {Columns.ACTION_DIST_INPUTS: logits}

    @override(ValueFunctionAPI)
    def compute_values(self, batch: dict[str, Any], embeddings: Any = None) -> torch.Tensor:
        if embeddings is None:
            embeddings = self._compute_embeddings(batch)
        return self._vf(embeddings).squeeze(-1)
