defmodule Laya.Question do
  @moduledoc false

  @qtypes %{choice: 0, score: 1, noul: 2}

  def qtype!(type) when type in [:choice, :score, :noul], do: Map.fetch!(@qtypes, type)
  def qtype!(type) when is_binary(type), do: type |> String.to_existing_atom() |> qtype!()

  def normalize!(questions) when is_map(questions) do
    questions
    |> Enum.map(fn {id, question} -> {to_string(id), normalize_question!(question)} end)
  end

  def normalize_question!(question) when is_map(question) do
    type = Map.fetch!(question, :type) |> normalize_type!()
    instructions = Map.fetch!(question, :instructions)

    unless is_binary(instructions), do: raise(ArgumentError, ":instructions must be a string")

    criteria = Map.get(question, :criteria)

    case {type, criteria} do
      {:choice, criteria} when is_list(criteria) ->
        unless Enum.all?(criteria, &match?({label, _} when is_binary(label), &1)) do
          raise ArgumentError, "choice criteria must be an ordered [{label, description}] list"
        end

      {:choice, _} ->
        raise ArgumentError,
              "choice criteria must be an ordered [{label, description}] list; map order is not a model contract"

      {:score, criteria} when is_list(criteria) ->
        :ok

      {:score, _} ->
        raise ArgumentError, "score criteria must be a list"

      {:noul, nil} ->
        :ok

      {:noul, criteria} when is_map(criteria) ->
        :ok

      {:noul, _} ->
        raise ArgumentError,
              "noul criteria must be omitted or a map with optional :false and :true keys"
    end

    %{type: type, instructions: instructions, criteria: criteria}
  end

  defp normalize_type!(type) when type in [:choice, :score, :noul], do: type

  defp normalize_type!(type) when is_binary(type),
    do: type |> String.to_existing_atom() |> normalize_type!()

  defp normalize_type!(type),
    do: raise(ArgumentError, "unknown Laya question type: #{inspect(type)}")
end
