# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Positional encodings, used in `transformer.py`."""

import enum
import functools
import math
from typing import Any, Optional, Union

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


class PositionalEncodings(enum.Enum):
  """Enum for all the positional encodings implemented."""

  NONE = 0
  SIN_COS = 1
  ALIBI = 2
  RELATIVE = 3
  ROTARY = 4
  LEARNT = 5
  NOISY_SIN_COS = 6
  NOISY_RELATIVE = 7
  NOISY_LEARNT = 8
  NOISY_ROTARY = 9
  NOISY_ALIBI = 10


@chex.dataclass
class SinCosParams:
  """Parameters for the classical sin/cos positional encoding."""

  # The maximum wavelength used.
  max_time: int = 10_000


# We will use this same class for Rotary and Relative.
RotaryParams = SinCosParams
RelativeParams = SinCosParams


@chex.dataclass
class LearntParams:
  """Parameters for the classical sin/cos positional encoding."""

  # The size of the embedding matrix to use.
  max_sequence_length: int


@chex.dataclass
class NoisySinCosParams:
  """Parameters for the noisy sin/cos positional encoding."""

  # The maximum length to sample.
  noise_max_length: int
  # The maximum wavelength used.
  max_time: int = 10_000


@chex.dataclass
class NoisyRelativeParams:
  """Parameters for the noisy relative positional encoding."""

  # The maximum length to sample.
  noise_max_length: int
  # Either randomize the right side and keep the same encodings for the left
  # part, keeping the symmetry, or randomize each side independently.
  randomize_both_sides: bool = False
  # The maximum wavelength used.
  max_time: int = 10_000


@chex.dataclass
class NoisyLearntParams:
  """Parameters for the noisy relative positional encoding."""

  # The maximum length to sample.
  noise_max_length: int


@chex.dataclass
class NoisyAlibiParams:
  """Parameters for the noisy alibi positional encoding."""

  # The maximum length to sample.
  noise_max_length: int
  # Either randomize the right side and keep the same encodings for the left
  # part, maintaining symmetry, or randomize each side independently.
  randomize_both_sides: bool = False


@chex.dataclass
class NoisyRotaryParams:
  """Parameters for the noisy rotary positional encoding."""

  # The maximum length to sample.
  noise_max_length: int


PositionalEncodingsParams = Union[
    SinCosParams,
    RelativeParams,
    RotaryParams,
    LearntParams,
    NoisySinCosParams,
    NoisyAlibiParams,
    NoisyRelativeParams,
    NoisyRotaryParams,
    NoisyLearntParams,
]


POS_ENC_TABLE = {
    'NONE': PositionalEncodings.NONE,
    'SIN_COS': PositionalEncodings.SIN_COS,
    'ALIBI': PositionalEncodings.ALIBI,
    'RELATIVE': PositionalEncodings.RELATIVE,
    'ROTARY': PositionalEncodings.ROTARY,
    'LEARNT': PositionalEncodings.LEARNT,
    'NOISY_SIN_COS': PositionalEncodings.NOISY_SIN_COS,
    'NOISY_ALIBI': PositionalEncodings.NOISY_ALIBI,
    'NOISY_RELATIVE': PositionalEncodings.NOISY_RELATIVE,
    'NOISY_ROTARY': PositionalEncodings.NOISY_ROTARY,
    'NOISY_LEARNT': PositionalEncodings.NOISY_LEARNT,
}

POS_ENC_PARAMS_TABLE = {
    'NONE': SinCosParams,
    'SIN_COS': SinCosParams,
    'ALIBI': SinCosParams,
    'RELATIVE': RelativeParams,
    'ROTARY': RotaryParams,
    'LEARNT': LearntParams,
    'NOISY_SIN_COS': NoisySinCosParams,
    'NOISY_ALIBI': NoisyAlibiParams,
    'NOISY_RELATIVE': NoisyRelativeParams,
    'NOISY_ROTARY': NoisyRotaryParams,
    'NOISY_LEARNT': NoisyLearntParams,
}


def sinusoid_position_encoding(
    sequence_length: int,
    hidden_size: int,
    max_timescale: float = 1e4,
    add_negative_side: bool = False,
) -> np.ndarray:
  """Creates sinusoidal encodings from the original transformer paper.

  The returned values are, for all i < D/2:
    array[pos, i] = sin(pos / (max_timescale^(2*i / D)))
    array[pos, D/2 + i] = cos(pos / (max_timescale^(2*i / D)))

  Args:
    sequence_length: Sequence length.
    hidden_size: Dimension of the positional encoding vectors, D. Should be
      even.
    max_timescale: Maximum timescale for the frequency.
    add_negative_side: Whether to also include the positional encodings for
      negative positions.

  Returns:
    An array of shape [L, D] if add_negative_side is False, else [2 * L, D].
  """
  if hidden_size % 2 != 0:
    raise ValueError(
        'The feature dimension should be even for sin/cos positional encodings.'
    )
  freqs = np.arange(0, hidden_size, 2)
  inv_freq = max_timescale ** (-freqs / hidden_size)
  pos_seq = np.arange(
      start=-sequence_length if add_negative_side else 0, stop=sequence_length
  )
  sinusoid_inp = np.einsum('i,j->ij', pos_seq, inv_freq)
  return np.concatenate([np.sin(sinusoid_inp), np.cos(sinusoid_inp)], axis=-1)


def noisy_fixed_positional_encodings(
    fixed_positional_encodings: chex.Array,
    sequence_length: int,
    rng: Optional[chex.PRNGKey] = None,
) -> chex.Array:
  """Generates noisy positional encodings from fixed positional encodings.

  Randomly samples and orders sequence_length positional encodings from a wider
  range [0, noise_max_length) rather than just [0, sequence_length).
  The user provides the full_encodings, which should span the entire range
  [0, noise_max_length).

  Args:
    fixed_positional_encodings: A tensor of shape (noise_max_length,
      embedding_size). This is from what the encodings will be sampled.
    sequence_length: The length of the output sequence.
    rng: Optional rng to use rather than hk.next_rng_key().

  Returns:
    A tensor of size [sequence_length, embedding_size].
  """
  noise_max_length, _ = fixed_positional_encodings.shape
  indexes = jrandom.choice(
      rng if rng is not None else hk.next_rng_key(),
      jnp.arange(noise_max_length),
      shape=(sequence_length,),
      replace=False,
  )
  indexes = jnp.sort(indexes)
  encodings = fixed_positional_encodings[indexes]
  return encodings


def _rel_shift_inner(logits: jax.Array, attention_length: int) -> jax.Array:
  """Shifts the relative logits.

  This is a more general than the original Transformer-XL implementation as
  inputs may also see the future. (The implementation does not rely on a
  causal mask removing the upper-right triangle.)

  Given attention length 3 and inputs:
      [[-3, -2, -1, 0, 1, 2],
       [-3, -2, -1, 0, 1, 2],
       [-3, -2, -1, 0, 1, 2]]

  The shifted output is:
      [[0, 1, 2],
       [-1, 0, 1],
       [-2, -1, 0]]

  Args:
    logits: input tensor of shape [T_q, T_v + T_q]
    attention_length: T_v `int` length of the attention, should be equal to
      memory size + sequence length.

  Returns:
    A shifted version of the input of size [T_q, T_v]. In each row, a window of
      size T_v elements is kept. The window starts at
      subsequent row.
  """
  if logits.ndim != 2:
    raise ValueError('`logits` needs to be an array of dimension 2.')
  tq, total_len = logits.shape
  assert total_len == tq + attention_length
  logits = jnp.reshape(logits, [total_len, tq])
  logits = jnp.reshape(logits, [total_len, tq])
  logits = jax.lax.slice(logits, (1, 0), logits.shape)  # logits[1:]
  logits = jnp.reshape(logits, [tq, total_len - 1])
  # Equiv to logits[:, :attention_length].
  logits = jax.lax.slice(logits, (0, 0), (tq, attention_length))
  return logits


def _rel_shift_causal(logits: jax.Array) -> jax.Array:
  """Shifts the relative logits, assuming causal attention.

  Given inputs:
      [[-4, -3, -2, -1],
       [-4, -3, -2, -1]]

  The shifted (and, later, masked) output is:
      [[-3, -2, -1,  0],
       [-4, -3, -2, -1]]

  Args:
    logits: input tensor of shape [T_q, T_v]

  Returns:
    A shifted version of the input of size [T_q, T_v].
  """
  t1, t2 = logits.shape
  # We prepend zeros on the final timescale dimension.
  to_pad = jnp.zeros_like(logits[..., :1])
  x = jnp.concatenate((to_pad, logits), axis=-1)

  # Reshape trick to  shift input.
  x = jnp.reshape(x, [t2 + 1, t1])

  # Remove extra time dimension and re-shape.
  x = jax.lax.slice(x, [1] + [0] * (x.ndim - 1), x.shape)

  return jnp.reshape(x, [t1, t2])


def relative_shift(
    logits: jax.Array, attention_length: int, causal: bool = False
) -> jax.Array:
  if causal:
    fn = _rel_shift_causal
  else:
    fn = lambda t: _rel_shift_inner(t, attention_length)
  return jax.vmap(jax.vmap(fn))(logits)


def apply_rotary_encoding(
    x: jnp.ndarray,
    position: jnp.ndarray,
    max_time: int = 10_000,
    noisy: bool = False,
    rng: Optional[chex.PRNGKey] = None,
) -> jnp.ndarray:
  """Applies RoPE positional encodings for the input.

  Paper: https://arxiv.org/abs/2104.09864

  Args:
    x: The input tensor on which RoPE will be applied. Usually it is either some
      queries q or some keys k.
    position: The positions to use. Usually it's an arange of the maximum
      length.
    max_time: Constant used to scale position by in the encodings.
    noisy: Whether to use the noisy version.
    rng: The rng key to use if the noisy version is used.

  Returns:
    A tensor with the same shape as x.
  """
  # Expand dims for positions to support inputs of shapes BTC or BTHC.
  freq_seq = jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
  freq_seq = freq_seq / (x.shape[-1] // 2)
  inv_freq = max_time**-freq_seq
  inv_freq = jnp.repeat(inv_freq, 2, 0)
  # Produce position inputs to periodic functions.
  t = position[:, :, None, None] * inv_freq[None, None, None, :]
  if noisy:
    t = noisy_fixed_positional_encodings(t[0, :, 0], x.shape[1], rng=rng)
    t = t[None, :, None, :]
  x_rot = jnp.einsum('bthd,dD->bthD', x, _rope_kernel(x.shape[-1], x.dtype))
  return x * jnp.cos(t).astype(x.dtype) + jnp.sin(t).astype(x.dtype) * x_rot


def _rope_kernel(n: int, dtype: Any) -> np.ndarray:
  """Reorders the embedding dimension of an array, to make rotation easier."""
  # We implement the equivalent of
  #   even_dims, odd_dims,  = x[..., ::2], x[..., 1::2]
  #   return jnp.stack((-odd_dims, even_dims), axis=-1).reshape(x.shape)
  # with a custom kernel for einsum. This allows the computation to execute
  # on the MXU instead of producing a slow gather.
  assert n % 2 == 0, n
  kernel = np.zeros((n, n), dtype)
  for i in range(n):
    # Swap each neighbouring pair of values.
    if i % 2 == 0:
      kernel[i, i + 1] = 1
    else:
      kernel[i, i - 1] = -1
  return kernel


def compute_attention_with_relative_encodings(
    queries: chex.Array,
    keys: chex.Array,
    max_time: int = 10_000,
    causal: bool = False,
) -> chex.Array:
  """Returns attention with relative positional encodings.

  This code strictly follows what is described in the TransformerXL paper.
  https://arxiv.org/pdf/1901.02860.pdf

  Args:
    queries: The queries used for attention. Shape (b, t, h, d).
    keys: The keys used for attention. Shape (b, T, h, d).
    max_time: Constant used to scale position by in the sin/cos encodings.
    causal: Whether to use causal attention when shifting the relative logits.

  Returns:
    The attention logits. Shape (b, h, t, T).
  """
  batch_size, k_seq_len, num_heads, num_hiddens = keys.shape
  hiddens = num_hiddens * num_heads

  # First compute the content logits.
  content_bias = hk.get_parameter(
      name='relpos_contentbias',
      shape=[num_heads, num_hiddens],
      init=hk.initializers.RandomNormal(stddev=0.02),
  )
  content_logits = jnp.einsum('bthd,bThd->bhtT', queries + content_bias, keys)

  positional_encodings = sinusoid_position_encoding(
      sequence_length=k_seq_len,
      hidden_size=hiddens,
      max_timescale=max_time,
      add_negative_side=not causal,
  )
  positional_encodings = jnp.broadcast_to(
      positional_encodings, (batch_size,) + positional_encodings.shape
  )
  relative_keys = hk.Linear(hiddens, with_bias=False)(positional_encodings)
  relative_keys = jnp.reshape(
      relative_keys, positional_encodings.shape[:-1] + (num_heads, num_hiddens)
  )

  # Then compute the relative part.
  relative_bias = hk.get_parameter(
      name='relpos_relativebias',
      shape=[num_heads, num_hiddens],
      init=hk.initializers.RandomNormal(stddev=0.02),
  )
  relative_logits = jnp.einsum(
      'bthd,bThd->bhtT', queries + relative_bias, relative_keys
  )
  # We shift the relative logits instead of the positional encoding matrix as
  # described in Appendix B of the paper (https://arxiv.org/pdf/1901.02860.pdf).
  relative_logits = relative_shift(
      relative_logits, attention_length=content_logits.shape[-1], causal=causal
  )
  assert content_logits.shape == relative_logits.shape
  return content_logits + relative_logits


def compute_attention_with_noisy_relative_encodings(
    queries: chex.Array,
    keys: chex.Array,
    noise_max_length: int,
    randomize_both_sides: bool = False,
    max_time: int = 10_000,
    causal: bool = False,
) -> chex.Array:
  """Returns attention with *noisy* relative positional encodings.

  This code follows what is described in the TransformerXL paper.
  https://arxiv.org/pdf/1901.02860.pdf
  However, in this version, the base positional encodings R (which are then
  shifted), are randomly sampled and ordered from a wider range than the
  sequence length.

  Args:
    queries: The queries used for attention. Shape (b, t, h, d).
    keys: The keys used for attention. Shape (b, T, h, d).
    noise_max_length: The maximum length used to sample the encodings.
    randomize_both_sides: Whether to sample the encodings on the left and on the
      right of the current token, or just sample from the left and take the
      inverted ones for the right part.
    max_time: Constant used to scale position by in the sin/cos encodings.
    causal: Whether to use causal attention when shifting the relative logits.

  Returns:
    The attention logits. Shape (b, h, t, T).
  """
  batch_size, k_seq_len, num_heads, num_hiddens = keys.shape
  hiddens = num_hiddens * num_heads

  # First compute the content logits.
  content_bias = hk.get_parameter(
      name='relpos_contentbias',
      shape=[num_heads, num_hiddens],
      init=hk.initializers.RandomNormal(stddev=0.02),
  )
  content_logits = jnp.einsum('bthd,bThd->bhtT', queries + content_bias, keys)

  # Select random indexes.
  # The indexes are in the range
  # [-noise_max_length + 1, noise_max_length - 1]
  right_indexes = jrandom.choice(
      hk.next_rng_key(),
      jnp.arange(1, noise_max_length),
      shape=(k_seq_len - 1,),
      replace=False,
  )
  right_indexes = jnp.sort(right_indexes)
  if randomize_both_sides:
    left_indexes = jrandom.choice(
        hk.next_rng_key(),
        jnp.arange(start=-noise_max_length + 1, stop=0),
        shape=(k_seq_len,),
        replace=False,
    )
    left_indexes = jnp.sort(left_indexes)
  else:
    left_indexes = -right_indexes[::-1]
    # The leftmost index is required by position_embedding.relative_shift.
    left_indexes = jnp.concatenate([jnp.zeros((1,)), left_indexes])
  zero_index = jnp.zeros((1,))
  indexes = jnp.concatenate([left_indexes, zero_index, right_indexes])
  # We shift the indexes to the range [0, 2*noise_max_length-1], since this
  # will be the range of the sin/cos. In this array, the value at index
  # noise_max_length is the sin/cos encoding at position 0, which is exactly
  # what we want: when doing relative attention, the token should have a fixed
  # encoding of position 0 for its own position.
  indexes += noise_max_length
  indexes = jnp.array(indexes, dtype=jnp.int32)

  positional_encodings = sinusoid_position_encoding(
      sequence_length=noise_max_length,
      hidden_size=hiddens,
      max_timescale=max_time,
  )
  positional_encodings = jnp.array(positional_encodings, dtype=jnp.float32)
  positional_encodings = positional_encodings[indexes]
  positional_encodings = jnp.broadcast_to(
      positional_encodings, (batch_size,) + positional_encodings.shape
  )
  relative_keys = hk.Linear(hiddens, with_bias=False)(positional_encodings)
  relative_keys = jnp.reshape(
      relative_keys, positional_encodings.shape[:-1] + (num_heads, num_hiddens)
  )

  # Then compute the relative part.
  relative_bias = hk.get_parameter(
      name='relpos_relativebias',
      shape=[num_heads, num_hiddens],
      init=hk.initializers.RandomNormal(stddev=0.02),
  )
  relative_logits = jnp.einsum(
      'bthd,bThd->bhtT', queries + relative_bias, relative_keys
  )
  # We shift the relative logits instead of the positional encoding matrix as
  # described in Appendix B of the paper (https://arxiv.org/pdf/1901.02860.pdf).
  relative_logits = relative_shift(
      relative_logits, attention_length=content_logits.shape[-1], causal=causal
  )
  assert content_logits.shape == relative_logits.shape
  return content_logits + relative_logits


def _get_alibi_slopes(num_heads: int) -> list[float]:
  """Returns the slopes for the different attention heads.

  While this does not exactly match the description of the [ALiBi
  paper](https://arxiv.org/pdf/2108.12409.pdf), it corresponds to the [official
  implementation](https://github.com/ofirpress/attention_with_linear_biases/blob/a06526fbfe557f9148e414b8569dcb97c7b182ba/fairseq/models/transformer.py#L742).

  Args:
    num_heads: The number of attention heads to create slopes for.
  """

  def get_slopes_power_of_2(n):
    start = 2 ** (-(2 ** -(math.log2(n) - 3)))
    ratio = start
    return [start * ratio**i for i in range(n)]

  if math.log2(num_heads).is_integer():
    return get_slopes_power_of_2(num_heads)
  else:
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    return (
        get_slopes_power_of_2(closest_power_of_2)
        + _get_alibi_slopes(2 * closest_power_of_2)[0::2][
            : num_heads - closest_power_of_2
        ]
    )


def compute_alibi_encodings_biases(
    attention_shape: tuple[int, ...]
) -> chex.Array:
  """Returns the biases following the ALiBi method.

  This code strictly follows what is described in the ALiBi paper.
  https://arxiv.org/pdf/2108.12409.pdf

  Args:
    attention_shape: The attention logits shape, without batch size, (h, t, T).

  Returns:
    The alibi biases, same shape as the input logits shape.
  """
  num_heads, q_seq_len, k_seq_len = attention_shape

  # Since we do not use causal masking, the upper triangle of the matrix has to
  # be nonzero. Therefore, we set it equal to the lower triangle, but we also
  # add a constant factor of 0.5 to the lower triangle, to (arbitrarily) break
  # the symmetry (otherwise, the model cannot distinguish left and right).
  alibi = np.zeros((q_seq_len, k_seq_len))
  alibi -= sum(np.tri(*alibi.shape, k=-i) for i in range(1, q_seq_len))
  alibi -= sum(np.tri(*alibi.T.shape, k=-i).T for i in range(1, k_seq_len))
  alibi += 0.5 * np.tri(*alibi.shape, k=-1)

  return alibi * jnp.array(_get_alibi_slopes(num_heads))[:, None, None]


def compute_noisy_alibi_encodings_biases(
    attention_shape: tuple[int, ...],
    noise_max_length: int,
    randomize_both_sides: bool = False,
) -> chex.Array:
  """Returns the biases following the ALiBi method.

  This code strictly follows what is described in the [ALiBi
  paper](https://arxiv.org/pdf/2108.12409.pdf).
  However, in this version, the biases are randomly sampled and ordered from a
  wider range than the sequence length.

  Args:
    attention_shape: The attention logits shape, without batch size, (h, t, T).
    noise_max_length: The maximum length used to sample the encodings.
    randomize_both_sides: Whether to sample the encodings on the left and on the
      right of the current token or just sample from the left and take the
      inverted ones for the right part.

  Returns:
    The alibi biases, same shape as the input logits shape.
  """
  num_heads, q_seq_len, k_seq_len = attention_shape

  sample_positions = functools.partial(
      jrandom.choice,
      a=jnp.arange(1, noise_max_length),
      replace=False,
  )

  if randomize_both_sides:
    right_positions = sample_positions(
        hk.next_rng_key(), shape=(k_seq_len - 1,)
    )
    left_positions = sample_positions(hk.next_rng_key(), shape=(q_seq_len - 1,))
    right_positions = -jnp.sort(right_positions)
    left_positions = jnp.sort(-left_positions)

  else:
    symmetric_positions = sample_positions(
        hk.next_rng_key(), shape=(max(q_seq_len, k_seq_len) - 1,)
    )
    symmetric_positions = -jnp.sort(symmetric_positions)
    right_positions = symmetric_positions[: k_seq_len - 1]
    left_positions = jnp.flip(symmetric_positions)[: q_seq_len - 1]

  # Since we do not use causal masking, the upper triangle of the matrix has to
  # be nonzero. Therefore, we set it equal to the lower triangle if
  # `randomize_both_side` is `False` and to randomly sampled positions
  # otherwise, but we also add a constant factor of 0.5 to the lower triangle,
  # to (arbitrarily) break the symmetry (otherwise, the model cannot distinguish
  # left and right).
  left_positions += 0.5

  # We add a dummy value to make the dimensions work for
  # position_embedding.relative_shift. The value will be ignored.
  left_positions = jnp.concatenate((jnp.empty((1,)), left_positions))

  positions = jnp.concatenate(
      (left_positions, jnp.zeros((1,)), right_positions)
  )
  # position_embedding.relative_shift requires a four-dimensional tensor.
  positions = jnp.tile(positions, (1, 1, q_seq_len, 1))

  alibi = relative_shift(
      positions,
      attention_length=k_seq_len,
      causal=False,
  )
  alibi = jnp.squeeze(alibi, axis=(0, 1))

  return alibi * jnp.array(_get_alibi_slopes(num_heads))[:, None, None]
