defmodule Laya.Model do
  @moduledoc false

  alias Bumblebee.HuggingFace.Hub
  alias Laya.{Head, Question, Sequence}

  @repo "convaiinnovations/laya"
  @checkpoints %{english: "", typed_decisions: "typed-decisions"}

  @enforce_keys [:encoder, :encoder_params, :tokenizer, :head_params, :config, :checkpoint]
  defstruct @enforce_keys

  @type t :: %__MODULE__{}

  def load(opts \\ []) do
    checkpoint = Keyword.get(opts, :checkpoint, :english)
    checkpoint = normalize_checkpoint!(checkpoint)
    repository = Keyword.get(opts, :repository, @repo)
    prefix = Map.fetch!(@checkpoints, checkpoint)

    try do
      config_path = download!(repository, join(prefix, "rl_agent_config.json"))
      config = config_path |> File.read!() |> Jason.decode!()
      encoder_path = join(prefix, "encoder")

      {:ok, spec} = Bumblebee.load_spec({:hf, repository, subdir: encoder_path})
      spec = Bumblebee.configure(spec, architecture: :base)

      unless match?(%Bumblebee.Text.ModernBert{}, spec) do
        raise ArgumentError,
              "this native runtime currently supports Laya's ModernBERT checkpoints; got #{inspect(spec.__struct__)}"
      end

      {:ok, encoder} =
        Bumblebee.load_model({:hf, repository, subdir: prefix_or_nil(prefix)},
          spec: spec,
          safetensors_reader: &read_encoder_tensors/1
        )

      {:ok, tokenizer} =
        Bumblebee.load_tokenizer({:hf, repository, subdir: join(prefix, "tokenizer")},
          type: :modernbert
        )

      head_params =
        repository
        |> download!(join(prefix, "model.safetensors"))
        |> read_head_tensors()
        |> materialize_tensors()
        |> Nx.backend_transfer(Nx.default_backend())

      {:ok,
       %__MODULE__{
         encoder: encoder.model,
         encoder_params: encoder.params,
         tokenizer: tokenizer,
         head_params: head_params,
         config: config,
         checkpoint: checkpoint
       }}
    rescue
      exception -> {:error, exception}
    end
  end

  def predict(%__MODULE__{} = model, state, questions) do
    questions = Question.normalize!(questions)
    options = sequence_options(model.config)

    items =
      Enum.map(questions, fn {id, question} ->
        item = Sequence.build(model.tokenizer, state, question, options)

        if length(item.markers) != length(item.options) do
          raise ArgumentError,
                "question #{inspect(id)} options exceed head_max_len=#{options[:head_max_len]}"
        end

        item
      end)

    pad_id =
      model.tokenizer.native_tokenizer |> Tokenizers.Tokenizer.get_vocab() |> Map.fetch!("[PAD]")

    batch = Sequence.batch(items, pad_id)

    encoder_output =
      Axon.predict(model.encoder, model.encoder_params, %{
        "input_ids" => batch.input_ids,
        "attention_mask" => batch.attention_mask
      })

    {logits, action_logits} =
      Head.predict(
        encoder_output.hidden_state,
        batch.attention_mask,
        batch.marker_pos,
        batch.marker_mask,
        batch.qtype,
        model.head_params
      )

    answers = decode_answers(questions, items, logits, action_logits, model.config)

    %{
      model: "laya-rl-agent",
      checkpoint: model.checkpoint,
      answers: answers,
      tokens: batch.attention_mask |> Nx.sum() |> Nx.to_number()
    }
  end

  defp decode_answers(questions, items, logits, action_logits, config) do
    logits = logits |> Nx.backend_transfer() |> Nx.to_list()
    action = action_logits |> Nx.backend_transfer() |> Nx.to_list()

    Enum.zip([questions, items, logits, action])
    |> Map.new(fn {{id, question}, item, row, action_row} ->
      count = length(item.markers)

      probabilities =
        row |> Enum.take(count) |> calibrated_probabilities(question.type, count, config)

      confidence = confidence(probabilities)
      action_probability = action_row |> softmax() |> hd() |> Float.round(4)

      answer =
        case question.type do
          :choice ->
            labels = Enum.map(question.criteria, &elem(&1, 0))
            {choice, _probability} = Enum.zip(labels, probabilities) |> Enum.max_by(&elem(&1, 1))

            %{
              type: :choice,
              choice: choice,
              probabilities:
                Map.new(Enum.zip(labels, Enum.map(probabilities, &Float.round(&1, 4)))),
              confidence: Float.round(confidence, 4),
              action: %{act_probability: action_probability}
            }

          :score ->
            score =
              probabilities
              |> Enum.with_index()
              |> Enum.map(fn {p, index} -> p * index end)
              |> Enum.sum()

            %{
              type: :score,
              score: Float.round(score, 4),
              legend:
                Map.new(
                  Enum.with_index(question.criteria, fn criterion, index ->
                    {Integer.to_string(index), criterion}
                  end)
                ),
              probabilities:
                Map.new(
                  Enum.with_index(probabilities, fn p, index ->
                    {Integer.to_string(index), Float.round(p, 4)}
                  end)
                ),
              confidence: Float.round(confidence, 4),
              action: %{act_probability: action_probability}
            }

          :noul ->
            p_true = Enum.at(probabilities, 1)

            %{
              type: :noul,
              noul: Float.round(p_true, 4),
              confidence: Float.round(max(p_true, 1.0 - p_true), 4),
              action: %{act_probability: action_probability}
            }
        end

      {id, answer}
    end)
  end

  defp calibrated_probabilities(logits, type, option_count, config) do
    temperature =
      Map.get(config["temperature_by_options"], temperature_bucket(type, option_count)) ||
        Enum.at(config["temperature"], Question.qtype!(type)) || 1.0

    logits
    |> Enum.map(&(&1 / max(1.0e-3, temperature)))
    |> softmax()
  end

  defp softmax(values) do
    max_value = Enum.max(values)
    values = Enum.map(values, &:math.exp(&1 - max_value))
    total = Enum.sum(values)
    Enum.map(values, &(&1 / total))
  end

  defp confidence(probabilities) do
    count = length(probabilities)

    if count < 2 do
      1.0
    else
      entropy =
        probabilities
        |> Enum.map(fn p -> -p * :math.log(max(p, 1.0e-12)) end)
        |> Enum.sum()

      min(max(1.0 - entropy / :math.log(count), 0.0), 1.0)
    end
  end

  defp read_encoder_tensors(path) do
    for {"encoder." <> name, tensor} <- Safetensors.read!(path, lazy: true), into: %{} do
      {name, tensor}
    end
  end

  defp read_head_tensors(path) do
    tensors = Safetensors.read!(path, lazy: true)

    %{
      type_embedding: tensors["type_emb.weight"],
      layers: %{
        layer_0: layer_params(tensors, 0),
        layer_1: layer_params(tensors, 1)
      },
      scorer: %{
        norm: pair(tensors, "scorer.0"),
        dense: pair(tensors, "scorer.1"),
        output: pair(tensors, "scorer.3")
      },
      action: %{
        dense: pair(tensors, "act_head.0"),
        output: pair(tensors, "act_head.2")
      }
    }
  end

  defp layer_params(tensors, index) do
    prefix = "head.layers.#{index}"

    %{
      in_proj: %{
        weight: Map.fetch!(tensors, prefix <> ".self_attn.in_proj_weight"),
        bias: Map.fetch!(tensors, prefix <> ".self_attn.in_proj_bias")
      },
      out_proj: pair(tensors, prefix <> ".self_attn.out_proj"),
      norm_1: pair(tensors, prefix <> ".norm1"),
      norm_2: pair(tensors, prefix <> ".norm2"),
      linear_1: pair(tensors, prefix <> ".linear1"),
      linear_2: pair(tensors, prefix <> ".linear2")
    }
  end

  defp pair(tensors, prefix),
    do: %{
      weight: Map.fetch!(tensors, prefix <> ".weight"),
      bias: Map.fetch!(tensors, prefix <> ".bias")
    }

  defp materialize_tensors(value) when is_struct(value), do: Nx.to_tensor(value)

  defp materialize_tensors(%{} = value),
    do: Map.new(value, fn {key, value} -> {key, materialize_tensors(value)} end)

  defp materialize_tensors(value), do: Nx.to_tensor(value)

  defp download!(repository, filename) do
    url = Hub.file_url(repository, filename, nil)
    {:ok, path} = Hub.cached_download(url)
    path
  end

  defp normalize_checkpoint!(:english), do: :english
  defp normalize_checkpoint!(:typed_decisions), do: :typed_decisions
  defp normalize_checkpoint!("english"), do: :english
  defp normalize_checkpoint!("typed-decisions"), do: :typed_decisions

  defp normalize_checkpoint!(checkpoint),
    do: raise(ArgumentError, "unknown supported checkpoint: #{inspect(checkpoint)}")

  defp sequence_options(config),
    do: [max_len: config["max_len"], head_max_len: config["head_max_len"]]

  defp prefix_or_nil(""), do: nil
  defp prefix_or_nil(prefix), do: prefix
  defp join("", filename), do: filename
  defp join(prefix, filename), do: prefix <> "/" <> filename

  defp temperature_bucket(type, option_count) do
    size =
      if option_count <= 2,
        do: "2",
        else:
          if(option_count <= 5, do: "3-5", else: if(option_count <= 10, do: "6-10", else: "11+"))

    "#{type}:#{size}"
  end
end
