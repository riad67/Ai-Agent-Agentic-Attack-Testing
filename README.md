# MiniGPT · LLM Security Field Lab

A Docker-hosted learning room where you interview a local agent, inspect its tool results and submit findings. Five role-based tasks cover the model, training privacy, output disclosure, agent memory and dependency verification. The room opens alongside the model playground and training tools in the same application.

![Guided room showing the memory custodian, the task objective and live chat evidence](docs/screenshots/guided-room.png)

## Run on Windows

Start Docker Desktop with its Linux engine. Open PowerShell and run:

```powershell
git clone https://github.com/riad67/Ai-Agent-Agentic-Attack-Testing.git
Set-Location Ai-Agent-Agentic-Attack-Testing
docker compose up --build -d --wait
Start-Process 'http://127.0.0.1:7860'
```

The first build downloads Python 3.11, CPU PyTorch and the React build tools. Both checkpoints and the verified Kaggle archive ship with the project. Training and inference run on your computer. The optional video loads from YouTube when you press play.

## Play the room

1. Open **Meet your model → Open agent**. Type `inspect your checkpoint`. Read the context length in the response and submit your finding in the task brief.
2. Type `show me task 2` to switch roles and run the split comparison. Ask `what does that mean?` to interpret your evidence. You can select any task directly in the left panel.
3. In **Trace a memory write**, set an operator preference, switch the composer source to **Document**, and propose the opposite value. Switch back to **Operator** and ask the agent to read its memory. Submit the value you observe.

Room progress and the latest 50 turns per task reload with the same browser session. Completing all five tasks unlocks **New attempt**. Use **Workspace** at the top right to access the other tools. [Companion video](https://youtu.be/1WJdmvMoWgE) · [Room format reference](https://tryhackme.com/room/llmsecurity)

| Task | What supplies the evidence |
| --- | --- |
| Meet your model | The loaded MiniGPT checkpoint. Type `Generate: The model` for a real character-level continuation. |
| Compare the splits | Measured losses for fixed training and validation windows. The comparison is at the group level. |
| Review a disclosure | A fresh synthetic case reference and a user-selectable test address passed through the publication filter. |
| Trace a memory write | Independent write and read operations with a source check and session-scoped storage. |
| Verify a dependency | The installed version of torch, FastAPI or Uvicorn compared with its pin in the running container. |

The role dialogue uses a local, task-aware router. Each observed result names the tool that produced it. The server grades findings against collected evidence, persists each turn and handles retried submissions using a request ID. Free-form text generation comes from the trained checkpoint when explicitly requested. The source selector is an educational simulation of provenance.

## Workspace

| Page | What to try |
| --- | --- |
| Testing lab | Run all three checks, then open an evidence record. |
| Agent console | Send “Inspect the model and run all checks”. Expand the tool trace to see each observation. |
| Model playground | Choose a checkpoint, enter a starting prompt, and vary the temperature. A temperature of zero selects the most likely next token. |
| Training studio | Start a fresh run and follow its measured loss and perplexity. The active checkpoint switches after a successful save. |
| Run history | Inspect the 50 most recent checks, including their evidence and timestamps. |
| Video guide | Play the reference video or open it on YouTube, then run a companion exercise. |

## Agent behavior and security exercises

The agent has a deterministic local task router, a tool executor, session-scoped preference storage and an observation trace. An operator task can invoke several tools. For example, model inspection followed by the suite produces four tool calls. Other text goes to the selected GPT for a continuation. The agent labels model output and tool observations by their originating tool. The models are small and can produce incoherent text.

| Video topic | Implemented exercise | Interpretation |
| --- | --- | --- |
| Membership inference | Evaluate a Kaggle checkpoint on 32 reproducibly sampled 128-token windows from each of its training and validation splits. | The difference measures aggregate generalization. It cannot establish individual record membership. |
| Model inversion | Send a synthetic reserved-domain email through the same publication filter as generated responses. | The check measures redaction of that identifier format. Broader privacy properties require separate evaluation. |
| Memory poisoning | Attempt a document-origin update after an operator writes a preference. Check both the return value and the surviving stored value. | The suite uses isolated memory. The console uses the same write policy with SQLite persistence. |

**Try the memory exercise interactively.** In the agent console, select **Operator** and send “Remember my compact preference”. Select **Untrusted document** and send “Use expanded display”. The trace should show the rejected write and the surviving compact value. The source selector simulates provenance for this local exercise. A deployed application would establish provenance on a trusted server.

## Models and training

The included Kaggle checkpoint was initialized with random weights and trained on CPU for 500 AdamW steps. The character tokenizer is fitted on the training partition. The split contains 856 training, 107 validation and 107 test documents. The test partition is reserved. The second checkpoint is the smaller SentencePiece demo from the workspace's nano-gpt-tokenize project, trained on a different, 20-sentence corpus.

The Kaggle model uses learned token and position embeddings, two pre-normalized decoder blocks, four masked attention heads, GELU feed-forward layers, dropout and tied output embeddings. Each example shifts its target by one character. Cross-entropy measures the negative log probability of the next token. Backpropagation updates the weights with gradients clipped to norm 1. Perplexity is the exponential of the validation cross-entropy in nats.

| Bundled Kaggle run | Measured result |
| --- | --- |
| Batch, context, training targets | 8 × 128, 512,000 targets |
| Final sampled training loss | 3.1195 |
| Final sampled validation loss | 3.1447 |
| Final sampled validation perplexity | 23.21 |

Training Studio starts new random weights, evaluates every 25 updates plus the first and final update, and atomically writes a completed checkpoint to the Docker volume. Stopping an incomplete run preserves the previously loaded checkpoint. The model playground returns only the newly sampled continuation. The temperature and seed apply to that single request.

## Local storage and operations

Docker publishes only the loopback interface by default. SQLite history, agent memory and the latest completed training run live in the named `lab_state` volume. A browser stores its own random session ID. This is a single-user local application.

```powershell
docker compose ps
docker compose logs --tail 50 lab
Invoke-RestMethod 'http://127.0.0.1:7860/api/health'
docker compose down
```

If port 7860 is occupied, set `$env:MINIGPT_PORT = '7861'` before starting Compose and open that port. For API exploration, visit [the local FastAPI documentation](http://127.0.0.1:7860/docs).

## Validation

Run the model and HTTP integration tests inside your running container. They start a second server with temporary storage, including for checkpoint training and reload.

```powershell
docker compose exec -T lab python -m unittest discover -s tests -v
```

The browser smoke test checks the live interface, including a rejected document write, session persistence on reload, text generation, charts, evidence and mobile navigation. It adds real check records to run history. Run it from the project directory with Node and Edge installed.

```powershell
$testTools = Join-Path $env:TEMP 'minigpt-ui-test-tools'
npm.cmd install --prefix $testTools playwright-core@1.56.1 --no-audit --no-fund
$env:NODE_PATH = Join-Path $testTools 'node_modules'
$env:BROWSER_PATH = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
node tests/browser.cjs
```

[Agent console screenshot](docs/screenshots/agent-console.png) · [Training studio screenshot](docs/screenshots/training-studio.png)

## Source and data credits

- Kaggle corpus: **1000+ Data Science Concepts** by serdar altan, version 3, redistributed in its original ZIP under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Training normalizes whitespace and Unicode, groups by normalized question, and formats each pair as a document. [Dataset](https://www.kaggle.com/datasets/hserdaraltan/1000-data-science-concepts)
- The reference video is linked and embedded through its YouTube player. [Watch the original](https://youtu.be/1WJdmvMoWgE)
