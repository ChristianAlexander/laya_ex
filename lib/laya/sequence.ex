defmodule Laya.Sequence do
  @moduledoc false

  alias Laya.Question
  alias Tokenizers.{Encoding, Tokenizer}

  @doc "Builds Laya's exact marker-oriented sequence for one typed question."
  def build(tokenizer, state, question, opts) do
    max_len = Keyword.fetch!(opts, :max_len)
    head_max_len = Keyword.fetch!(opts, :head_max_len)
    native = tokenizer.native_tokenizer
    special = special_ids!(native)
    options = render_options(question)
    mask_token = tokenizer.special_tokens.mask

    encode = fn text ->
      {:ok, encoding} = Tokenizer.encode(native, text, add_special_tokens: false)
      Encoding.get_ids(encoding)
    end

    head_ids =
      encode.(
        "#{question.type} question: #{String.replace(question.instructions, mask_token, " ")}"
      )

    option_ids =
      Enum.map(options, fn option ->
        [
          special.mask
          | (" " <> String.replace(option, mask_token, " ")) |> encode.() |> Enum.take(48)
        ]
      end)

    option_budget = head_max_len - (option_ids |> Enum.map(&length/1) |> Enum.sum())

    {option_ids, option_budget} =
      if option_budget < 16 do
        per_option = max(4, div(head_max_len - 16, max(1, length(option_ids))))
        option_ids = Enum.map(option_ids, &Enum.take(&1, per_option))
        {option_ids, head_max_len - (option_ids |> Enum.map(&length/1) |> Enum.sum())}
      else
        {option_ids, option_budget}
      end

    head_ids = Enum.take(head_ids, max(8, option_budget))
    prefix = [special.cls | head_ids] ++ [special.sep]

    {marked_options, _next_position} =
      Enum.map_reduce(option_ids, length(prefix), fn option, position ->
        {{position, option}, position + length(option)}
      end)

    markers = Enum.map(marked_options, &elem(&1, 0))
    prefix = prefix ++ Enum.flat_map(marked_options, &elem(&1, 1)) ++ [special.sep]
    room = max(0, max_len - length(prefix) - 1)

    state_ids =
      state |> serialize() |> String.replace(mask_token, " ") |> encode.() |> Enum.take(room)

    ids = Enum.take(prefix ++ state_ids ++ [special.sep], max_len)

    %{
      ids: ids,
      markers: Enum.filter(markers, &(&1 < max_len)),
      qtype: Question.qtype!(question.type),
      options: options
    }
  end

  def render_options(%{type: :choice, criteria: criteria}) do
    Enum.map(criteria, fn {label, criterion} ->
      if criterion in [nil, ""], do: label, else: "#{label}: #{render_criterion(criterion)}"
    end)
  end

  def render_options(%{type: :score, criteria: criteria}) do
    Enum.with_index(criteria, fn criterion, index ->
      "level #{index}: #{render_criterion(criterion)}"
    end)
  end

  def render_options(%{type: :noul, criteria: criteria}) do
    criteria = criteria || %{}
    false_criterion = Map.get(criteria, false) || Map.get(criteria, "false")
    true_criterion = Map.get(criteria, true) || Map.get(criteria, "true")

    [
      "false: " <>
        if(false_criterion in [nil, ""],
          do: "no, the statement does not hold",
          else: render_criterion(false_criterion)
        ),
      "true: " <>
        if(true_criterion in [nil, ""],
          do: "yes, the statement holds",
          else: render_criterion(true_criterion)
        )
    ]
  end

  def batch(items, pad_id) do
    max_length = items |> Enum.map(&length(&1.ids)) |> Enum.max()
    max_options = items |> Enum.map(&length(&1.markers)) |> Enum.max()

    ids = Enum.map(items, &pad(&1.ids, max_length, pad_id))

    attention_mask =
      Enum.map(items, fn item -> pad(List.duplicate(1, length(item.ids)), max_length, 0) end)

    markers = Enum.map(items, &pad(&1.markers, max_options, 0))

    marker_mask =
      Enum.map(items, fn item -> pad(List.duplicate(1, length(item.markers)), max_options, 0) end)

    %{
      input_ids: Nx.tensor(ids, type: :u32),
      attention_mask: Nx.tensor(attention_mask, type: :u32),
      marker_pos: Nx.tensor(markers, type: :s32),
      marker_mask: Nx.tensor(marker_mask, type: :u8),
      qtype: Nx.tensor(Enum.map(items, & &1.qtype), type: :s32)
    }
  end

  defp special_ids!(native) do
    vocab = Tokenizer.get_vocab(native)

    for {name, token} <- [cls: "[CLS]", sep: "[SEP]", mask: "[MASK]", pad: "[PAD]"], into: %{} do
      {name, Map.fetch!(vocab, token)}
    end
  end

  defp serialize(state) when is_binary(state), do: state
  defp serialize(state) when is_map(state) or is_list(state), do: Jason.encode!(state)

  defp serialize(state),
    do: raise(ArgumentError, "state must be a string, map, or list, got: #{inspect(state)}")

  defp render_criterion(value) when is_binary(value), do: value
  defp render_criterion(value), do: Jason.encode!(value)

  defp pad(values, target_length, value),
    do: values ++ List.duplicate(value, target_length - length(values))
end
