defmodule Laya do
  @moduledoc """
  Native Elixir runtime for the Laya System-1 decision models.

  Laya uses a fine-tuned ModernBERT encoder and a custom decision head. This
  package loads the encoder through Bumblebee and executes the published head
  with Nx, so no Python process is involved at inference time.
  """

  alias Laya.Model

  @opaque model :: %Model{}

  @doc """
  Downloads (on first use) and prepares a Laya checkpoint.

  `:checkpoint` is one of `:english`, `:multilingual`, or
  `:typed_decisions`. The English checkpoint is the default.
  """
  @spec load(keyword()) :: {:ok, model()} | {:error, Exception.t()}
  def load(opts \\ []), do: Model.load(opts)

  @doc "Runs all supplied typed questions in a single Laya batch."
  @spec predict(model(), String.t() | map() | list(), map()) :: map()
  def predict(%Model{} = model, state, questions), do: Model.predict(model, state, questions)
end
