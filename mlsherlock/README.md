```
mlsherlock/
├── cli.py                   # Entry point: mlsh train
├── engine/
│   ├── loop.py              # Agent loop: LLM calls, tool dispatch, history management
│   ├── providers.py         # Anthropic + OpenAI wrappers with normalised interface
│   ├── state.py             # Session state: paths, iteration count, stuck detection
│   └── system_prompt.py     # ML diagnostic protocol (heuristics + stopping criteria)
├── tools/
│   ├── registry.py          # Input schemas (Pydantic) + tool dispatcher
│   ├── read_data.py         # Profile CSV and inject df/target into sandbox
│   ├── run_python.py        # Execute Python in the persistent sandbox
│   ├── download_data.py     # Fetch dataset from URL, named shortcut, or Kaggle
│   ├── save_plot.py         # Save current matplotlib figure to output dir
│   ├── ask_user.py          # Pause for human input
│   └── finish.py            # Serialise model, write summary, end loop
├── execution/
│   ├── sandbox.py           # CodeExecutor: persistent exec namespace with timeout isolation
│   └── capture.py           # Context manager: captures stdout/stderr during exec
└── ui/
    └── cli_callbacks.py     # Rich terminal output (verbose + terse modes)
```
