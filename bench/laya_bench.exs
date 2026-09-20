Mix.Task.run("app.start")

# This repository includes EMLX only for local Apple Silicon benchmarks. A
# consuming application should configure its own Nx backend.
Nx.default_backend({EMLX.Backend, device: :gpu})

{:ok, model} = Laya.load()

state = %{
  "from" => "user@acme.com",
  "subject" => "Duplicate charge on invoice #4411",
  "body" =>
    "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."
}

questions = %{
  "department" => %{
    type: :choice,
    instructions: "Which department should handle this request?",
    criteria: [
      {"billing", "invoices, payments, refunds"},
      {"technical", "bugs, outages, system errors"},
      {"sales", "pricing, new contracts"},
      {"other", "everything else"}
    ]
  },
  "urgency" => %{
    type: :score,
    instructions: "How urgent is this request?",
    criteria: ["not urgent", "soon", "critical deadline or blocking issue"]
  },
  "churn_risk" => %{
    type: :noul,
    instructions: "Does the user threaten to cancel or leave?"
  },
  "refund_requested" => %{
    type: :noul,
    instructions: "Does the user explicitly request a refund?"
  }
}

# Compile the encoder and the Nx head before timing. The benchmark measures
# warm inference only, not model download or compilation.
_ = Laya.predict(model, state, questions)

seconds =
  System.get_env("LAYA_BENCH_SECONDS", "10")
  |> Integer.parse()
  |> case do
    {seconds, ""} when seconds > 0 -> seconds
    _ -> raise "LAYA_BENCH_SECONDS must be a positive integer"
  end

Benchee.run(
  %{
    "4 questions / one batch" => fn -> Laya.predict(model, state, questions) end,
    "1 question / one batch" => fn ->
      Laya.predict(model, state, Map.take(questions, ["department"]))
    end
  },
  time: seconds,
  memory_time: 0,
  warmup: 3,
  parallel: 1
)
