defmodule Laya.Head do
  @moduledoc false

  import Nx.Defn

  @epsilon 1.0e-5

  @doc "Runs Laya's published decision head on ModernBERT hidden states."
  def predict(hidden_state, attention_mask, marker_pos, marker_mask, qtype, params) do
    Nx.Defn.jit(&forward/6).(hidden_state, attention_mask, marker_pos, marker_mask, qtype, params)
  end

  defn forward(hidden_state, attention_mask, marker_pos, marker_mask, qtype, params) do
    hidden_state = Nx.as_type(hidden_state, :f32)
    type_embedding = Nx.take(params.type_embedding, qtype)
    hidden_state = hidden_state + Nx.new_axis(type_embedding, 1)

    hidden_state = transformer(hidden_state, attention_mask, params.layers.layer_0)
    hidden_state = transformer(hidden_state, attention_mask, params.layers.layer_1)

    sequence_length = Nx.axis_size(hidden_state, 1)
    hidden_size = Nx.axis_size(hidden_state, 2)
    batch_size = Nx.axis_size(hidden_state, 0)
    option_count = Nx.axis_size(marker_pos, 1)

    indices =
      marker_pos
      |> Nx.clip(0, sequence_length - 1)
      |> Nx.new_axis(-1)
      |> Nx.broadcast({batch_size, option_count, hidden_size})

    marker_states = Nx.take_along_axis(hidden_state, indices, axis: 1)

    logits =
      marker_states
      |> layer_norm(params.scorer.norm.weight, params.scorer.norm.bias)
      |> dense(params.scorer.dense.weight, params.scorer.dense.bias)
      |> Axon.Activations.gelu()
      |> dense(params.scorer.output.weight, params.scorer.output.bias)
      |> Nx.squeeze(axes: [-1])

    logits = Nx.select(Nx.equal(marker_mask, 1), logits, Nx.tensor(-1.0e4, type: :f32))

    probabilities = Axon.Activations.softmax(stop_grad(logits))
    {top2, _indices} = Nx.top_k(probabilities, k: 2)
    top1 = Nx.slice_along_axis(top2, 0, 1, axis: -1) |> Nx.squeeze(axes: [-1])
    runner_up = Nx.slice_along_axis(top2, 1, 1, axis: -1) |> Nx.squeeze(axes: [-1])

    entropy =
      probabilities
      |> Nx.clip(1.0e-9, 1.0)
      |> Nx.log()
      |> Nx.multiply(probabilities)
      |> Nx.negate()
      |> Nx.sum(axes: [-1])
      |> Nx.divide(Nx.log(Nx.max(Nx.sum(Nx.equal(marker_mask, 1), axes: [-1]), 2)))

    option_fraction =
      Nx.sum(Nx.equal(marker_mask, 1), axes: [-1]) |> Nx.max(2) |> Nx.divide(255.0)

    features = Nx.stack([top1, top1 - runner_up, entropy, option_fraction], axis: -1)

    pooled = Nx.slice_along_axis(hidden_state, 0, 1, axis: 1) |> Nx.squeeze(axes: [1])

    action_logits =
      Nx.concatenate([pooled, features], axis: -1)
      |> dense(params.action.dense.weight, params.action.dense.bias)
      |> Axon.Activations.gelu()
      |> dense(params.action.output.weight, params.action.output.bias)

    {logits, action_logits}
  end

  defnp transformer(hidden_state, attention_mask, params) do
    attended =
      self_attention(
        layer_norm(hidden_state, params.norm_1.weight, params.norm_1.bias),
        attention_mask,
        params
      )

    hidden_state = hidden_state + attended

    feed_forward =
      hidden_state
      |> layer_norm(params.norm_2.weight, params.norm_2.bias)
      |> dense(params.linear_1.weight, params.linear_1.bias)
      |> Axon.Activations.gelu()
      |> dense(params.linear_2.weight, params.linear_2.bias)

    hidden_state + feed_forward
  end

  defnp self_attention(hidden_state, attention_mask, params) do
    batch_size = Nx.axis_size(hidden_state, 0)
    sequence_length = Nx.axis_size(hidden_state, 1)
    hidden_size = Nx.axis_size(hidden_state, 2)
    num_heads = div(hidden_size, 64)
    head_size = div(hidden_size, num_heads)

    qkv = dense(hidden_state, params.in_proj.weight, params.in_proj.bias)
    qkv = Nx.reshape(qkv, {batch_size, sequence_length, 3, num_heads, head_size})

    query =
      Nx.slice_along_axis(qkv, 0, 1, axis: 2)
      |> Nx.squeeze(axes: [2])
      |> Nx.transpose(axes: [0, 2, 1, 3])

    key =
      Nx.slice_along_axis(qkv, 1, 1, axis: 2)
      |> Nx.squeeze(axes: [2])
      |> Nx.transpose(axes: [0, 2, 1, 3])

    value =
      Nx.slice_along_axis(qkv, 2, 1, axis: 2)
      |> Nx.squeeze(axes: [2])
      |> Nx.transpose(axes: [0, 2, 1, 3])

    # Laya uses `nhead = hidden_size // 64`, so each attention head is
    # always 64 wide and the PyTorch scale is exactly 1 / sqrt(64).
    scores = Nx.dot(query, [3], [0, 1], key, [3], [0, 1]) * 0.125

    padding =
      attention_mask
      |> Nx.equal(1)
      |> Nx.reshape({batch_size, 1, 1, sequence_length})
      |> Nx.broadcast({batch_size, num_heads, sequence_length, sequence_length})

    scores = Nx.select(padding, scores, Nx.tensor(-1.0e4, type: :f32))
    weights = Axon.Activations.softmax(scores)

    weights
    |> Nx.dot([3], [0, 1], value, [2], [0, 1])
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch_size, sequence_length, hidden_size})
    |> dense(params.out_proj.weight, params.out_proj.bias)
  end

  defnp dense(input, weight, bias) do
    Nx.dot(input, [-1], weight, [1]) + bias
  end

  defnp layer_norm(input, weight, bias) do
    mean = Nx.mean(input, axes: [-1], keep_axes: true)
    variance = Nx.mean(Nx.pow(input - mean, 2), axes: [-1], keep_axes: true)
    (input - mean) / Nx.sqrt(variance + @epsilon) * weight + bias
  end
end
