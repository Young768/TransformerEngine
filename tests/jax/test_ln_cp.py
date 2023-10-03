# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import pytest
from functools import partial
from jax.experimental.pjit import pjit

from utils import is_devices_enough
from transformer_engine.jax.flax import extend_logical_axis_rules
from transformer_engine.jax.sharding import get_dot_sharding_meta
from transformer_engine.jax.sharding import get_elementwise_sharding_meta
from transformer_engine.jax.sharding import get_fp8_meta_sharding_meta
from transformer_engine.jax.sharding import global_shard_guard
from transformer_engine.jax.sharding import infer_major_sharding_type
from transformer_engine.jax.sharding import is_dp_enabled, is_tp_enabled
from transformer_engine.jax.sharding import ShardingMeta, ShardingResource, ShardingType
from transformer_engine.jax.sharding import get_elementwise_sharding_meta, extend_fsdp_sharding_meta
from transformer_engine.jax.layernorm import layernorm


def _get_sharding_resource(mesh_names, sharding_type):
    dp_r = None
    tp_r = None

    if sharding_type in (ShardingType.DP, ShardingType.DP_TP_COL, ShardingType.DP_TP_ROW):
        dp_r = mesh_names[0]

    if sharding_type in (ShardingType.TP_COL, ShardingType.TP_ROW):
        tp_r = mesh_names[0]

    if sharding_type in (ShardingType.DP_TP_COL, ShardingType.DP_TP_ROW):
        tp_r = mesh_names[1]
    return ShardingResource(dp_r, tp_r)


DEVICE_COUNT = 8
MESH_CONFIG = [((8,), ("dp",), ShardingType.DP)]

SRS = [
    ShardingResource(),
    ShardingResource('data', None),
    ShardingResource(None, 'model'),
    ShardingResource('data', 'model')
]


class TestShardingMetaGenerator:

    BATCH_AXIS_NAME = 'batch'
    MODEL_AXIS_NAME = 'model'

    


    @pytest.mark.parametrize('mesh_shape,mesh_names,sharding_type', MESH_CONFIG)
    @pytest.mark.parametrize('input_shape', [(32, 128)])
    @pytest.mark.parametrize('other_shape', [(128,)])
    @pytest.mark.parametrize('batch_dim', [0])
    @pytest.mark.parametrize('zero_centered_gamma', [False])
    @pytest.mark.skipif(not is_devices_enough(DEVICE_COUNT), reason='Num of GPU is not enough')
    def test_elementwise(self, mesh_shape, mesh_names, sharding_type, input_shape, other_shape,
                         batch_dim, zero_centered_gamma):

        epsilon = 1e-6

        def func(x, gamma, beta):
            x = layernorm(x, gamma, beta, layernorm_type="layernorm", zero_centered_gamma=zero_centered_gamma,
                                    epsilon=epsilon, sharding_type=sharding_type, dp_dim_index=batch_dim)
            return jnp.mean(x)

        #def grad_f(x, gamma, beta):
        #    return jnp.mean(func(x, gamma, beta))
        test = false
        devices = np.asarray(jax.devices()[:DEVICE_COUNT]).reshape(*mesh_shape)
        with global_shard_guard(_get_sharding_resource(mesh_names, sharding_type)):
            with jax.sharding.Mesh(devices, mesh_names):
                x_ = random.normal(random.PRNGKey(1124), input_shape)
                gamma = jnp.ones(other_shape)
                beta = jnp.ones(other_shape)
                graded_f = jax.value_and_grad(func, argnums=(0, 1, 2))
                pjitter = pjit(graded_f)
                ref_hlo = pjitter.lower(x_, gamma, beta).compile().as_text()
                # This is for reference to validate the sharding is done correctly.