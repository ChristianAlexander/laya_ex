# Laya

Native Nx/Bumblebee runtime for [Laya](https://laya.convaiinnovations.com/) by
Convai Innovations. It downloads the
[official model checkpoint](https://huggingface.co/convaiinnovations/laya) on
first load.

```elixir
{:ok, model} = Laya.load()

Laya.predict(model, "Please refund the duplicate invoice.", %{
  "department" => %{
    type: :choice,
    instructions: "Which team should handle this request?",
    criteria: [
      {"billing", "invoices, payments, and refunds"},
      {"technical", "bugs and outages"},
      {"other", "everything else"}
    ]
  },
  "refund_requested" => %{
    type: :noul,
    instructions: "Does the customer explicitly request a refund?"
  }
})
```

`choice.criteria` is an ordered list of `{label, description}` tuples.

```sh
mix run bench/laya_bench.exs
```

The English checkpoint downloads on first load.

The package does not select an Nx backend. Configure one in the host app, for
example `EMLX.Backend` on Apple Silicon or `EXLA.Backend` on CPU/NVIDIA.
