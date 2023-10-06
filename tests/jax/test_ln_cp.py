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
from jax.sharding import PartitionSpec, NamedSharding
P = PartitionSpec


def _get_sharding_spec(mesh_names, sharding_type):

    if sharding_type is ShardingType.DP:
        return P(mesh_names[0], None), P(None), P(None)
    elif sharding_type is ShardingType.DP_TP_COL:
        return P(mesh_names[0], mesh_names[1]), P(None), P(None)
    else:
        raise NotImplementedError


DEVICE_COUNT = 8
MESH_CONFIG = [((8,), ("dp",), ShardingType.DP, {"all-reduce":2}),]
              #((4, 2), ("dp", "tp"), ShardingType.DP_TP_COL, {"all-reduce":1}),]
              #((8,), ("tp",), ShardingType.TP_COL, {}),
              #((2, 4), ("dp", "tp"), ShardingType.DP_TP_COL, {"all-reduce":1}),


Allreduce = "all-reduce"
Other = "other"

epsilon = 1e-6

def func(x, gamma, beta):
    x = layernorm(x, gamma, beta, layernorm_type="layernorm", zero_centered_gamma=zero_centered_gamma,
                                    epsilon=epsilon, sharding_type=sharding_type, dp_dim_index=batch_dim)
    return jnp.mean(x)

def count_collective(hlo):
    tmp = hlo.splitlines()
    symb = "-start"
    result = {}
    debug = ""
    for line in tmp:
        txt = line.split()
        if len(txt) > 0 and symb in txt[0]:
            if Allreduce in txt[0]:
                if Allreduce not in result: result[Allreduce] = 0
                result[Allreduce] += 1
            else:
                if Other not in result: result[Other] = 0
                result[Other] += 1
    return result

class TestXMAPGenerator:

    @pytest.mark.parametrize('mesh_shape,mesh_names,sharding_type, collective_ref', MESH_CONFIG)
    @pytest.mark.parametrize('input_shape', [(32, 128)])
    @pytest.mark.parametrize('other_shape', [(128,)])
    @pytest.mark.parametrize('batch_dim', [0])
    @pytest.mark.parametrize('zero_centered_gamma', [False])
    @pytest.mark.skipif(not is_devices_enough(DEVICE_COUNT), reason='Num of GPU is not enough')
    def test_layernorm(self, mesh_shape, mesh_names, sharding_type, collective_ref, input_shape, other_shape,
                         batch_dim, zero_centered_gamma):

        devices = np.asarray(jax.devices()[:DEVICE_COUNT]).reshape(*mesh_shape)
        with global_shard_guard(_get_sharding_resource(mesh_names, sharding_type)):
            with jax.sharding.Mesh(devices, mesh_names):
                x_ = random.normal(random.PRNGKey(1124), input_shape)
                gamma = jnp.ones(other_shape)
                beta = jnp.ones(other_shape)
                graded_f = jax.value_and_grad(func, argnums=(0, 1, 2))
                pjitter = pjit(graded_f)
                out = {}
                hlo = pjitter.lower(x_, gamma, beta).compile().as_text()
                dic = count_collective(hlo)
                assert dic==collective_ref, f"Expected number of collective is: {dic==collective_ref=}, but got {dic=}."


class TestCPGenerator:

    @pytest.mark.parametrize('mesh_shape,mesh_names,sharding_type, collective_ref', MESH_CONFIG)
    @pytest.mark.parametrize('input_shape', [(32, 128)])
    @pytest.mark.parametrize('other_shape', [(128,)])
    @pytest.mark.parametrize('batch_dim', [0])
    @pytest.mark.parametrize('zero_centered_gamma', [False])
    @pytest.mark.skipif(not is_devices_enough(DEVICE_COUNT), reason='Num of GPU is not enough')
    def test_layernorm(self, mesh_shape, mesh_names, sharding_type, collective_ref, input_shape, other_shape,
                         batch_dim, zero_centered_gamma):

        devices = np.asarray(jax.devices()[:DEVICE_COUNT]).reshape(*mesh_shape)
        with global_shard_guard(_get_sharding_resource(mesh_names, sharding_type)):
            with jax.sharding.Mesh(devices, mesh_names):
                x_ = random.normal(random.PRNGKey(1124), input_shape)
                gamma = jnp.ones(other_shape)
                beta = jnp.ones(other_shape)
                graded_f = jax.value_and_grad(func, argnums=(0, 1, 2))
                pjitter = pjit(graded_f)
                out = {}
                hlo = pjitter.lower(x_, gamma, beta).compile().as_text()
                dic = count_collective(hlo)
                assert dic==collective_ref, f"Expected number of collective is: {dic==collective_ref=}, but got {dic=}."
        
        x_ = random.normal(random.PRNGKey(1124), input_shape)
        gamma = jnp.ones(other_shape)
        beta = jnp.ones(other_shape)
        test = False
        x_spec, gamma_spec, beta_spec = _get_sharding_spec(mesh_names, sharding_type)
        with mesh, global_shard_guard(_get_sharding_resource(mesh_names, sharding_type)):
            x_ = jax.device_put(x_, NamedSharding(mesh, x_spec))
            gamma = jax.device_put(gamma, NamedSharding(mesh, gamma_spec))
            beta = jax.device_put(beta, NamedSharding(mesh, beta_spec))

            pjitter = pjit(graded_f,
                        in_shardings=[x_spec, gamma_spec, beta_spec],
                        out_shardings=(None, x_spec, gamma_spec, beta_spec,))
            
            hlo = pjitter.lower(x_, gamma, beta).compile().as_text()
            dic = count_collective(hlo)
            print(hlo)
            test_l, test_grads = pjitter(x_, gamma, beta)
            dic = count_collective(hlo)
            assert test, f"{dic=}."
            assert dic==collective_ref, f"Expected number of collective is: {dic==collective_ref=}, but got {dic=}."