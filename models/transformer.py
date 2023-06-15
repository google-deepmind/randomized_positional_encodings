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

"""Transformer model."""

import dataclasses
from typing import Any, Callable, Optional, Type, Union

from absl import logging
import chex
import haiku as hk
import jax
import jax.nn as jnn
import jax.numpy as jnp

from randomized_positional_encodings.models import positional_encodings as pos_encs_lib
from randomized_positional_encodings.models import transformer_utils


@chex.dataclass
class TransformerConfig:
  """Hyperparameters used in the Transformer architectures."""

  # The dimension of the first embedding.
  embedding_dim: int = 64
  # The number of multi-head attention layers.
  num_layers: int = 5
  # The number of heads per layer.
  num_heads: int = 8
  # The number of hidden neurons per head. If None, it is set to be equal to
  # `embedding_dim // num_heads`.
  num_hiddens_per_head: Optional[int] = None
  # The probability that each element is discarded by the dropout modules.
  # None means dropout is not used at all.
  dropout_prob: Optional[float] = 0.1
  # The parameter initialization scale for the embeddings.
  emb_init_scale: float = 0.02
  # Whether to use the embeddings rather than raw inputs.
  use_embeddings: bool = True
  # Whether to use lookup-embeddings, in which case the inputs must be ints.
  use_lookup_embeddings: bool = False
  # Input vocabulary size, not needed if use_lookup_embeddings is False.
  input_vocab_size: Optional[int] = None
  # Whether to share embeddings between the Encoder and the Decoder.
  share_embeddings: bool = False
  # The size of the sliding attention window. See MultiHeadDotProductAttention.
  attention_window: Optional[int] = None
  # The positional encoding used with default sin/cos (Vaswani et al., 2017).
  positional_encodings: pos_encs_lib.PositionalEncodings = dataclasses.field(
      default_factory=lambda: pos_encs_lib.PositionalEncodings.SIN_COS
  )
  # The parameters for the positional encodings, default sin/cos.
  positional_encodings_params: pos_encs_lib.PositionalEncodingsParams = (
      dataclasses.field(default_factory=pos_encs_lib.SinCosParams)
  )
  # How much larger the hidden layer of the feedforward network should be
  # compared to the `embedding_dim`.
  widening_factor: int = 4
  # Which activation function to use.
  activation_fn: Callable[[jax.Array], jax.Array] = jnn.relu
  # Add mask to make causal predictions. All the decoders use causal masking by
  # default, this option is only used in the encoder. This is quite unusual but
  # can still be useful in some rare cases.
  encoder_causal_masking: bool = False
  # Which token to use for the beginning of the string. None means an array
  # full of zeros will be used.
  bos_token: Optional[int] = None
  # Used by the chunked transformer.
  chunk_context_length: Optional[int] = None

  def __post_init__(self) -> None:
    """Runs after the config has been created."""
    if self.num_hiddens_per_head is None:
      self.num_hiddens_per_head = self.embedding_dim // self.num_heads

    if self.positional_encodings is None:
      self.positional_encodings = pos_encs_lib.PositionalEncodings.SIN_COS
      self.positional_encodings_params = pos_encs_lib.SinCosParams()
    elif self.positional_encodings_params is None:
      raise ValueError('No parameters for positional encodings are passed.')
    elif not isinstance(
        self.positional_encodings, pos_encs_lib.PositionalEncodings
    ) or not isinstance(
        self.positional_encodings_params, pos_encs_lib.PositionalEncodingsParams
    ):
      raise ValueError(
          "The positional encodings passed are not of the right type. You're"
          ' probably passing strings rather than actual objects.'
      )


class MultiHeadDotProductAttention(hk.Module):
  """Multi-head dot-product attention (Vaswani et al., 2017)."""

  def __init__(
      self,
      num_heads: int,
      num_hiddens_per_head: int,
      positional_encodings: Optional[pos_encs_lib.PositionalEncodings] = None,
      positional_encodings_params: Optional[
          pos_encs_lib.PositionalEncodingsParams
      ] = None,
      attention_window: Optional[int] = None,
      name: Optional[str] = None,
  ) -> None:
    """Initializes the attention module.

    Args:
      num_heads: Number of heads to use.
      num_hiddens_per_head: Number of hidden neurons per head.
      positional_encodings: Which positional encodings to use in the attention.
        None means no positional encodings are applied to keys or queries.
      positional_encodings_params: Parameters for the positional encodings.
      attention_window: Size of the attention sliding window. None means no
        sliding window is used (or equivalently, window=full_attention_length).
        We attend only on attention_window tokens around a given query token. We
        attend to tokens before AND after the query token. If attention_window
        is even, we use the value +1.
      name: Name of the module.
    """
    super().__init__(name=name)
    self._num_heads = num_heads
    self._num_hiddens_per_head = num_hiddens_per_head
    self._positional_encodings = positional_encodings
    self._attention_window = attention_window
    self._positional_encodings_params = (
        positional_encodings_params  # pytype: disable=annotation-type-mismatch
    )

  def __call__(
      self,
      inputs_q: chex.Array,
      inputs_kv: chex.Array,
      mask: Optional[chex.Array] = None,
      causal: bool = False,
  ) -> chex.Array:
    """Returns the output of the multi-head attention."""
    batch_size, sequence_length, embedding_size = inputs_q.shape

    num_hiddens = self._num_hiddens_per_head * self._num_heads
    q = hk.Linear(num_hiddens, with_bias=False)(inputs_q)
    k = hk.Linear(num_hiddens, with_bias=False)(inputs_kv)
    v = hk.Linear(num_hiddens, with_bias=False)(inputs_kv)
    # The second (sequence) dimension is undefined since it can differ between
    # queries and keys/values when decoding. Also checking that the inputs have
    # the same batch size as the reshape below does not guarantee a failure if
    # they are different.
    chex.assert_equal_shape_prefix([inputs_q, inputs_kv], prefix_len=1)
    new_shape = (batch_size, -1, self._num_heads, self._num_hiddens_per_head)
    q = jnp.reshape(q, new_shape)
    k = jnp.reshape(k, new_shape)
    v = jnp.reshape(v, new_shape)

    # Let b=batch_size, t=seq_len, h=num_heads, and d=num_hiddens_per_head.
    if self._positional_encodings == pos_encs_lib.PositionalEncodings.RELATIVE:
      # We type hint the params to match the if statement, for pytype.
      self._positional_encodings_params: pos_encs_lib.RelativeParams
      attention = pos_encs_lib.compute_attention_with_relative_encodings(
          q, k, self._positional_encodings_params.max_time, causal=causal
      )
    elif (
        self._positional_encodings
        == pos_encs_lib.PositionalEncodings.NOISY_RELATIVE
    ):
      if causal:
        raise NotImplementedError(
            'Noisy positional encodings not implemented for causal attention.'
        )
      # We type hint the params to match the if statement, for pytype.
      self._positional_encodings_params: pos_encs_lib.NoisyRelativeParams
      attention = pos_encs_lib.compute_attention_with_noisy_relative_encodings(
          q,
          k,
          max_time=self._positional_encodings_params.max_time,
          noise_max_length=self._positional_encodings_params.noise_max_length,
          randomize_both_sides=self._positional_encodings_params.randomize_both_sides,
          causal=causal,
      )
    else:
      if self._positional_encodings == pos_encs_lib.PositionalEncodings.ROTARY:
        q = pos_encs_lib.apply_rotary_encoding(
            q, position=jnp.arange(q.shape[1])[None, :]
        )
        k = pos_encs_lib.apply_rotary_encoding(
            k, position=jnp.arange(k.shape[1])[None, :]
        )
      elif (
          self._positional_encodings
          == pos_encs_lib.PositionalEncodings.NOISY_ROTARY
      ):
        # We type hint the params to match the if statement, for pytype.
        self._positional_encodings_params: pos_encs_lib.NoisyRotaryParams
        noise_max_length = self._positional_encodings_params.noise_max_length
        # WARNING: This only works with self-attention, ie q.shape==k.shape.
        rng = hk.next_rng_key()
        q = pos_encs_lib.apply_rotary_encoding(
            q,
            position=jnp.arange(noise_max_length)[None, :],
            noisy=True,
            rng=rng,
        )
        k = pos_encs_lib.apply_rotary_encoding(
            k,
            position=jnp.arange(noise_max_length)[None, :],
            noisy=True,
            rng=rng,
        )
      attention = jnp.einsum('bthd,bThd->bhtT', q, k)
    attention *= 1.0 / jnp.sqrt(self._num_hiddens_per_head)

    # ALiBi encodings are not scaled with the 1 / sqrt(d_k) factor.
    if self._positional_encodings == pos_encs_lib.PositionalEncodings.ALIBI:
      attention += pos_encs_lib.compute_alibi_encodings_biases(
          attention.shape[1:]
      )
    if (
        self._positional_encodings
        == pos_encs_lib.PositionalEncodings.NOISY_ALIBI
    ):
      # We type hint the params to match the if statement, for pytype.
      self._positional_encodings_params: pos_encs_lib.NoisyAlibiParams
      attention += pos_encs_lib.compute_noisy_alibi_encodings_biases(
          attention.shape[1:],
          noise_max_length=self._positional_encodings_params.noise_max_length,
          randomize_both_sides=self._positional_encodings_params.randomize_both_sides,
      )

    if self._attention_window is not None:
      # We compute the sliding attention by just applying a mask on the values
      # that are outside our window.
      attention_mask = transformer_utils.compute_sliding_window_mask(
          sequence_length, self._attention_window
      )
      attention = jnp.where(
          attention_mask, attention, jnp.finfo(jnp.float32).min
      )

    if mask is not None:
      attention = jnp.where(mask, attention, jnp.finfo(jnp.float32).min)

    normalized_attention = jnn.softmax(attention)

    output = jnp.einsum('bhtT,bThd->bthd', normalized_attention, v)
    output = jnp.reshape(output, (batch_size, sequence_length, num_hiddens))
    return hk.Linear(embedding_size, with_bias=False)(output)


class TransformerInit(hk.Module):
  """Helper class to avoid repeating the same __init__."""

  def __init__(self, config: TransformerConfig):
    """Initializes the module."""
    super().__init__()
    self._config = config
    if self._config.use_lookup_embeddings and self._config.bos_token is None:
      raise ValueError("Can't use lookup embeddings with a zero bos_token.")


class TransformerEmbedder(TransformerInit):
  """A module to embed sequences and add positional encodings if needed."""

  def embed_sequences(self, sequences: chex.Array) -> chex.Array:
    """Returns embedded sequences, following a linear operation or hk.Embed."""
    embs_init = hk.initializers.TruncatedNormal(
        stddev=self._config.emb_init_scale
    )
    if self._config.use_lookup_embeddings:
      embeddings_layer = hk.Embed(
          vocab_size=self._config.input_vocab_size,
          embed_dim=self._config.embedding_dim,
          lookup_style=hk.EmbedLookupStyle.ARRAY_INDEX,
          w_init=embs_init,
      )
      integer_sequences = jnp.argmax(sequences, axis=-1)
      embeddings = embeddings_layer(integer_sequences)
    else:
      embeddings_layer = hk.Linear(
          self._config.embedding_dim,
          with_bias=False,
          w_init=embs_init,
      )
      embeddings = embeddings_layer(sequences)

    embeddings *= jnp.sqrt(self._config.embedding_dim)
    return embeddings

  def add_positional_encodings(self, embeddings: chex.Array) -> chex.Array:
    """Returns new embeddings, which have been added positional encodings.

    The shape of the returned array is (B, T, E), where E is the dimension of
    the embeddings (if any are used, otherwise E = F).

    Args:
      embeddings: A batch of embeddings, of shape (B, T, F).
    """
    chex.assert_rank(embeddings, 3)

    _, sequence_length, embedding_size = embeddings.shape

    pos_enc_params = self._config.positional_encodings_params
    if (
        self._config.positional_encodings
        == pos_encs_lib.PositionalEncodings.SIN_COS
    ):
      pos_enc_params: pos_encs_lib.SinCosParams
      pos_encodings = pos_encs_lib.sinusoid_position_encoding(
          sequence_length=sequence_length,
          hidden_size=embedding_size,
          max_timescale=pos_enc_params.max_time,
      )
      h = embeddings + pos_encodings
      if self._config.dropout_prob is not None:
        h = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, h)
    elif (
        self._config.positional_encodings
        == pos_encs_lib.PositionalEncodings.NOISY_SIN_COS
    ):
      pos_enc_params: pos_encs_lib.NoisySinCosParams
      if pos_enc_params.noise_max_length > pos_enc_params.max_time:
        logging.warning(
            (
                'noise_max_length=%i is larger than max_time=%i, some '
                'positional encodings will be equal.'
            ),
            pos_enc_params.noise_max_length,
            pos_enc_params.max_time,
        )
      pos_encodings = pos_encs_lib.sinusoid_position_encoding(
          sequence_length=pos_enc_params.noise_max_length,
          hidden_size=embedding_size,
          max_timescale=pos_enc_params.max_time,
      )
      pos_encodings = jnp.array(pos_encodings)
      pos_encodings = pos_encs_lib.noisy_fixed_positional_encodings(
          pos_encodings, sequence_length
      )
      h = embeddings + pos_encodings
      if self._config.dropout_prob is not None:
        h = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, h)
    elif (
        self._config.positional_encodings
        == pos_encs_lib.PositionalEncodings.LEARNT
    ):
      pos_enc_params: pos_encs_lib.LearntParams
      pos_encodings = jnp.arange(sequence_length)
      pos_encodings = hk.Embed(
          vocab_size=pos_enc_params.max_sequence_length,
          embed_dim=embedding_size,
      )(pos_encodings)
      h = embeddings + pos_encodings
      if self._config.dropout_prob is not None:
        h = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, h)
    elif (
        self._config.positional_encodings
        == pos_encs_lib.PositionalEncodings.NOISY_LEARNT
    ):
      pos_enc_params: pos_encs_lib.NoisyLearntParams
      pos_encodings = jnp.arange(pos_enc_params.noise_max_length)
      pos_encodings = hk.Embed(
          vocab_size=pos_enc_params.noise_max_length, embed_dim=embedding_size
      )(pos_encodings)
      pos_encodings = pos_encs_lib.noisy_fixed_positional_encodings(
          pos_encodings, sequence_length
      )
      h = embeddings + pos_encodings
      if self._config.dropout_prob is not None:
        h = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, h)
    else:
      h = embeddings
    return h


class TransformerEncoder(TransformerInit):
  """Transformer Encoder (Vaswani et al., 2017)."""

  def __call__(self, inputs: jnp.ndarray) -> chex.Array:
    """Returns the transformer encoder output, shape [B, T, E]."""
    batch_size, sequence_length = inputs.shape[:2]
    # Embeds the inputs, adds positional encodings.
    embedder = TransformerEmbedder(self._config)
    embeddings = embedder.embed_sequences(inputs)
    h = embedder.add_positional_encodings(embeddings)

    # The causal mask is shared across heads.
    if self._config.encoder_causal_masking:
      causal_mask = jnp.tril(
          jnp.ones((batch_size, 1, sequence_length, sequence_length))
      )
    else:
      causal_mask = None

    for _ in range(self._config.num_layers):
      attention = MultiHeadDotProductAttention(
          num_heads=self._config.num_heads,
          num_hiddens_per_head=self._config.num_hiddens_per_head,
          positional_encodings=self._config.positional_encodings,
          positional_encodings_params=self._config.positional_encodings_params,
          attention_window=self._config.attention_window,
      )(
          inputs_q=h,
          inputs_kv=h,
          mask=causal_mask,
          causal=self._config.encoder_causal_masking,
      )
      if self._config.dropout_prob is not None:
        attention = hk.dropout(
            hk.next_rng_key(), self._config.dropout_prob, attention
        )
      attention = transformer_utils.layer_norm(h + attention)

      # Position-wise feedforward network.
      h = hk.Linear(self._config.embedding_dim * self._config.widening_factor)(
          attention
      )
      h = self._config.activation_fn(h)
      h = hk.Linear(self._config.embedding_dim)(h)

      if self._config.dropout_prob is not None:
        h = hk.dropout(hk.next_rng_key(), self._config.dropout_prob, h)
      h = transformer_utils.layer_norm(h + attention)
    return h


class ChunkedTransformerEncoder(TransformerInit):
  """A Transformer encoder that can handle large histories via chunks.

  We chunk the inputs, moving from a shape (B, T, F) to a shape (B, T/C, C, F),
  where C is the length of the chunk. Note that T must be a multiple of C for it
  to work. The chunks are then passed independently to the encoder, and all the
  outputs are then concatenated together, to return a shape (B, T, E), where E
  is the embedding_dim of the TransformerEncoder, see class above.
  """

  def __call__(self, inputs: chex.Array) -> jnp.ndarray:
    """Calls the chunked transformer encoder."""
    batch_size, history_len = inputs.shape[:2]
    inputs = transformer_utils.chunk_sequences(
        inputs, chunk_length=self._config.chunk_context_length
    )
    outputs = TransformerEncoder(self._config)(inputs=inputs)
    return jnp.reshape(outputs, (batch_size, history_len, outputs.shape[-1]))


CallableTransformer = Union[
    ChunkedTransformerEncoder,
    TransformerEncoder,
]


def make_transformer(
    output_size: int,
    transformer_module: Type[CallableTransformer],
    return_all_outputs: bool = False,
    **transformer_kwargs,
) -> Any:
  """Returns a transformer predict function."""

  if 'positional_encodings' in transformer_kwargs:
    if isinstance(transformer_kwargs['positional_encodings'], str):
      transformer_kwargs['positional_encodings_params'] = (
          pos_encs_lib.POS_ENC_PARAMS_TABLE[
              transformer_kwargs['positional_encodings']
          ](**transformer_kwargs['positional_encodings_params'])
      )
      transformer_kwargs['positional_encodings'] = pos_encs_lib.POS_ENC_TABLE[
          transformer_kwargs['positional_encodings']
      ]

  config = TransformerConfig(**transformer_kwargs)

  def transformer(*args, **kwargs) -> chex.Array:
    output = transformer_module(config=config)(*args, **kwargs)
    if not return_all_outputs:
      output = output[:, -1, :]
    return hk.Linear(output_size)(output)

  return transformer
