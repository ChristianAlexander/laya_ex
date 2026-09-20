defmodule Laya.MixProject do
  use Mix.Project

  def project do
    [
      app: :laya,
      version: "0.1.0",
      elixir: "~> 1.17",
      description: "Native Nx/Bumblebee runtime for the Laya decision model.",
      package: package(),
      docs: docs(),
      source_url: "https://github.com/ChristianAlexander/laya_ex",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # ModernBERT loading, tokenizer support, and SafeTensors decoding.
      {:bumblebee, "~> 0.7.0"},
      # Used by the local benchmark only. Applications choose their own Nx backend.
      {:emlx, "~> 0.4", only: :dev},
      {:jason, "~> 1.4"},
      {:benchee, "~> 1.4", only: :dev},
      {:ex_doc, "~> 0.39", only: :dev, runtime: false}
    ]
  end

  defp package do
    [
      files: ~w(.formatter.exs .github bench lib mix.exs README.md CHANGELOG.md LICENSE NOTICE),
      licenses: ["Apache-2.0"],
      links: %{
        "GitHub" => "https://github.com/ChristianAlexander/laya_ex",
        "Laya" => "https://laya.convaiinnovations.com/",
        "Model" => "https://huggingface.co/convaiinnovations/laya"
      }
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: ["README.md", "CHANGELOG.md", "NOTICE"]
    ]
  end
end
