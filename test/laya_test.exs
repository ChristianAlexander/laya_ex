defmodule LayaTest do
  use ExUnit.Case, async: true

  alias Laya.{Head, Question, Sequence}

  test "choice labels deliberately require an ordered criteria list" do
    assert_raise ArgumentError, ~r/map order is not a model contract/, fn ->
      Question.normalize!(%{
        department: %{type: :choice, instructions: "Where?", criteria: %{billing: "money"}}
      })
    end

    assert [
             {"department",
              %{type: :choice, criteria: [{"billing", "money"}], instructions: "Where?"}}
           ] =
             Question.normalize!(%{
               department: %{
                 type: :choice,
                 instructions: "Where?",
                 criteria: [{"billing", "money"}]
               }
             })
  end

  test "renders Laya's three decision primitives" do
    assert Sequence.render_options(%{type: :choice, criteria: [{"billing", "refunds"}]}) == [
             "billing: refunds"
           ]

    assert Sequence.render_options(%{type: :score, criteria: ["low", "high"]}) == [
             "level 0: low",
             "level 1: high"
           ]

    assert Sequence.render_options(%{type: :noul, criteria: nil}) == [
             "false: no, the statement does not hold",
             "true: yes, the statement holds"
           ]
  end

  test "Nx decision head produces marker logits and action logits" do
    hidden = Nx.divide(Nx.iota({2, 4, 64}, type: :f32), 100.0)
    attention = Nx.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], type: :u32)
    markers = Nx.tensor([[1, 2, 0], [1, 2, 0]], type: :s32)
    marker_mask = Nx.tensor([[1, 1, 0], [1, 1, 0]], type: :u8)
    qtype = Nx.tensor([0, 2], type: :s32)

    {logits, action_logits} =
      Head.predict(hidden, attention, markers, marker_mask, qtype, head_params())

    assert Nx.shape(logits) == {2, 3}
    assert Nx.shape(action_logits) == {2, 2}
    assert Enum.all?(Nx.to_flat_list(logits), &is_number/1)
    assert Enum.at(Nx.to_flat_list(logits), 2) < -9_000
  end

  defp head_params do
    zeros = fn shape -> Nx.broadcast(0.0, shape) end
    ones = fn shape -> Nx.broadcast(1.0, shape) end

    pair = fn output, input ->
      %{weight: Nx.divide(Nx.iota({output, input}, type: :f32), 10_000.0), bias: zeros.({output})}
    end

    norm = %{weight: ones.({64}), bias: zeros.({64})}

    layer = %{
      in_proj: pair.(192, 64),
      out_proj: pair.(64, 64),
      norm_1: norm,
      norm_2: norm,
      linear_1: pair.(256, 64),
      linear_2: pair.(64, 256)
    }

    %{
      type_embedding: zeros.({3, 64}),
      layers: %{layer_0: layer, layer_1: layer},
      scorer: %{norm: norm, dense: pair.(64, 64), output: pair.(1, 64)},
      action: %{dense: pair.(256, 68), output: pair.(2, 256)}
    }
  end
end
