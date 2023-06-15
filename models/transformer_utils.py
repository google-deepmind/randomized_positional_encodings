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

"""Utils for the transformer architectures."""

import chex
import haiku as hk
import jax.numpy as jnp


def layer_norm(x: chex.Array) -> chex.Array:
  return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)


def chunk_sequences(sequences: chex.Array, chunk_length: int) -> chex.Array:
  """Chunks an array of sequences, on the second (time) dimension.

  Args:
    sequences: An array of sequences, of shape (B, T, F).
    chunk_length: The length of each chunk.

  Returns:
    An array of shape (B, T // chunk_length, chunk_length, F)
  Raises:
    ValueError if T is not a multiple of chunk_length.
  """
  chex.assert_rank(sequences, 3)
  batch_size, history_len, num_features = sequences.shape
  if history_len < chunk_length:
    context_length = history_len
  elif history_len % chunk_length == 0:
    context_length = chunk_length
  else:
    raise ValueError(
        'The history length should a multiple of the context length. Got'
        f' history_length={history_len} and'
        f' context_length={chunk_length}'
    )

  history_batch_size = history_len // context_length
  return jnp.reshape(
      sequences,
      (batch_size * history_batch_size, context_length, num_features),
  )


def compute_sliding_window_mask(
    sequence_length: int, attention_window: int
) -> chex.Array:
  """Returns a k-diagonal mask for a sliding window.

  Args:
    sequence_length: The length of the sequence, which will determine the shape
      of the output.
    attention_window: The size of the sliding window.

  Returns:
    A symmetric matrix of shape (sequence_length, sequence_length),
    attention_window-diagonal, with ones on the diagonal and on all the
    upper/lower diagonals up to attention_window // 2.

  Raises:
    ValueError if attention_window is <= 0.
  """
  if attention_window <= 0:
    raise ValueError(
        f'The attention window should be > 0. Got {attention_window}.'
    )

  if attention_window == 1:
    return jnp.eye(sequence_length, sequence_length)

  attention_mask = jnp.sum(
      jnp.stack(
          [
              jnp.eye(sequence_length, sequence_length, k=k, dtype=jnp.int32)
              for k in range(1, attention_window // 2 + 1)
          ]
      ),
      axis=0,
  )
  attention_mask = attention_mask + jnp.transpose(attention_mask)
  attention_mask += jnp.eye(sequence_length, sequence_length)
  return attention_mask
