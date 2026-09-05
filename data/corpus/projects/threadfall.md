---
doc_type: project
project_id: threadfall
name: "Threadfall: The Shattered Pact"
tagline: "Every choice pulls a thread."
status: flagship
repo: https://github.com/DEMONKINGKAI/Threadfall
demo: null
domain: [interactive fiction, causal reasoning, LLM systems]
stack: [Python 3.11+, FastAPI, pgmpy, React 18, Tailwind, Vite, Cytoscape.js, HuggingFace InferenceClient, Qwen2.5-7B-Instruct, ChromaDB, Qdrant, sentence-transformers, Docker, nginx]
version: 1.1.0
---

# Threadfall — The Shattered Pact

## One-line summary
A solo narrative RPG where story outcomes are decided by a deterministic causal engine (a probabilistic DAG built with pgmpy), and the LLM is restricted to narrating what the engine has already determined.

## The problem it solves
LLMs lose track of world state in long-running interactive systems: they contradict themselves, drift, and hallucinate continuity. Most "AI games" hand both decision-making and narration to the model, so the world state is unreliable. Threadfall inverts this: **the causal engine is the author; the LLM is the voice.**

## Core design principle
The engine decides, the LLM narrates. The narrator receives a packet of pre-determined facts (outcome, probability, causal consequences) and writes prose around them. It cannot alter a result, invent a consequence, or contradict the causal graph. This guarantees outcomes are reproducible, interpretable, and auditable rather than hallucinated.

## How it works — the /action pipeline (9 steps per player input)
1. **Classify player text** into one of the canonical action types (combat_outcome, npc_interaction, resource_use, espionage_action, political_action; later additions: flee, rest, search, delve_action) using a verb-first weighted keyword classifier. No API call, no latency. Verb-first because the primary verb signals intent more reliably than noun context: "Ask the soldiers about the ambush" is npc_interaction, not combat.
2. **Map action type to an ability stat** (D&D 5e style: combat → Strength/Constitution, npc_interaction → Charisma/Wisdom, resource_use → Intelligence, espionage → Dexterity/Intelligence, political → Charisma/Intelligence). Modifier = floor((score − 10) / 2).
3. **Compute success probability** with a sigmoid (k=8) over the normalised modifier, then compress: p_success = p^1.5, p_failure = (1−p)^1.5, p_partial = the remainder. Even a STR 20 fighter can fail; no outcome is certain.
4. **Sample the outcome** (success / partial / failure) with a seeded RNG — sessions are reproducible.
5. **DAG.do()** — graph surgery: cut incoming edges to the intervened node, force it to the sampled outcome, propagate downstream in topological order. Propagation uses a weighted sigmoid (k=6) mapping "causal pressure" from parent states to the child's target state.
6. **Bayesian Network belief update** — a discrete BN (pgmpy) runs alongside the structural DAG. CPTs are auto-generated from DAG edge weights via Dirichlet-smoothed softmax (α=0.1). After each do(), Variable Elimination recomputes marginals over all non-intervened nodes (optimised to a single `query(joint=False)` call in v1.1.0).
7. **Advance act** if a milestone completes.
8. **LLM narrates** from pre-determined facts only.
9. **Check campaign over** — all milestones resolved?

## Pearl's causal hierarchy in the engine
- Association: P(Y|X) — "what is faction loyalty right now?"
- Intervention: P(Y|do(X)) — "if I attack, what changes?"
- Counterfactual: "would the pact have held if I had spoken first?"

## The campaign graph
"The Shattered Pact": 20 nodes, 26 edges, 5 acts. Node types: action nodes (player decisions), state nodes (player_health, enemy_defeated, npc_trust, town_reputation, gold_remaining, item_inventory, faction_loyalty, secret_knowledge, conflict_status, pact_integrity), milestone nodes (act1→act4, sequential gates), and a final_outcome node (story_failure / story_neutral / story_victory).

**Act gating — three layers of pacing:** score threshold (push_score ≥ 0.65 for incomplete, ≥ 0.88 for complete), one-step enforcement (at most one state advance per action), and sequential lock (actN blocked until act(N−1) is at least partial). final_outcome cannot resolve until all four act milestones are past incomplete.

Two further campaigns were added later: **The Stolen Crown** (3-act heist/intrigue set in Velmoor; adds espionage/political nodes) and **The Ashen Vault** (dungeon delve with delve_action, dungeon_depth, torch_remaining, trap_triggered, loot_found, exit_reached). A campaign selector on the character screen picks between them; DAG metadata and act titles are read from each campaign's JSON.

## LLM narrator
- Model: Qwen/Qwen2.5-7B-Instruct via HuggingFace InferenceClient (featherless-ai route); fallback chain Qwen2.5-3B → Llama-3.2-3B → Phi-3.5-mini; 30s timeout.
- Temperature 0.85, max 600 tokens.
- Structured output enforced by system prompt and regex-parsed: TITLE (4–7 word chapter title) / PARA1 (3–4 sentences of action and sensory result) / PARA2 (exactly 3 sentences with one unresolved causal hint).
- Streaming via SSE (`POST /stream_action`) so the outcome badge appears immediately while tokens arrive.
- Local tone-matched fallback prose if no HF token is set.
- A character randomizer endpoint uses the same LLM to generate a gothic D&D character in fixed key-value format, regex-parsed, stats clamped to [8, 18], with a curated local fallback.

## RAG inside Threadfall
Every resolved action is embedded (all-MiniLM-L6-v2, 384-dim) and stored in a vector store (ChromaDB default, Qdrant optional, behind a `VectorStore` ABC with an `init_store()` factory driven by env vars). At narration time the current player action is embedded and the 3 most similar past entries are retrieved and injected into the prompt as "Remembered echoes" (act, outcome, 180-char snippet) — long-range thematic memory without bloating context with the full history. The vector store also powers session persistence, "Load Chronicle", session replay (`GET /session/{id}/replay`), and transparent session restore after server restart (past interventions are replayed through `dag.do()` so the graph is in the correct post-history state).

## Architecture
```
backend/
  main.py                    FastAPI — all endpoints
  causal_engine/dag.py       CausalDAG, do(), topological propagation, cycle detection (Kahn)
  causal_engine/campaigns/   long.json (Shattered Pact), short.json (Stolen Crown), Ashen Vault
  pgm_engine/world_state.py  Bayesian Network, CPT generation, belief update
  pgm_engine/character.py    CharacterSheet, stat → probability mapping
  llm/narrator.py            HF InferenceClient, structured prose, streaming, RAG context
  llm/classifier.py          verb-first keyword classifier
  llm/randomizer.py          character generation
  storage/vector_store.py    VectorStore ABC, ChromaStore, QdrantStore
  storage/session_store.py   save/load/replay/purge/retrieve_rag_context
  models/schemas.py          Pydantic schemas
frontend/src/
  App.jsx, GameView.jsx (three-panel: Graph | Narrative | Stats), NarrativeFeed.jsx,
  StatsPanel.jsx, CausalGraph.jsx (Cytoscape), ReplayView.jsx, LoadScreen.jsx, Modal.jsx
```

## API
POST /new_game · POST /action · POST /stream_action (SSE) · GET /session/{id} · GET /session/{id}/replay · POST /resume_session · DELETE /session/{id} · GET /randomize_character

## Engineering and reliability work (from the changelog, v0.4 → v1.1)
- 64 unit tests across dag / character / classifier (sigmoid math, do() propagation, milestone gating, campaign-over detection, the "social verb beats combat noun" invariant).
- Dockerised: two-stage backend image (gcc build stage → slim runtime, non-root user, HF cache volume), Vite→nginx frontend with SSE-safe proxying (`proxy_buffering off`), docker-compose with a chroma service and optional qdrant; `.env.example` documents all variables.
- Session bearer-token auth (`secrets.token_hex(32)`, `x-session-token` header) on mutating endpoints.
- Streaming disconnect safety: a preliminary entry is written to the vector DB before the SSE stream starts and upserted on completion, so a network drop never loses an action.
- Performance: BN inference collapsed from N queries to one Variable Elimination pass.
- Correctness: DAG cycle detection at load; classifier dead zones (flee/rest/search) fixed; resume replays interventions so world state is not lost.
- UX: mobile-responsive layout with drawer overlays, node tooltips on the causal graph, act-transition title cards, differentiated game-over screens (VICTORY / RUIN / THE BALANCE HOLDS), replay player with speed control and scrubber, localStorage narrative history.

## Key decisions and why
- **Deterministic engine + LLM voice** — makes every consequence auditable and reproducible; the LLM has no decision authority.
- **Keyword classifier instead of an LLM call for intent** — zero latency and fully inspectable; verb-first weighting encodes the linguistic insight that verbs carry intent.
- **Dual DAG + BN** — the DAG tracks determined state; the BN tracks probabilistic belief over nodes not yet intervened on, displayed live in the Causal Web panel.
- **Dual vector-store backends** — Chroma for zero-config local dev, Qdrant for a production-shaped deployment, behind one interface.
- **Compression exponent on outcome probabilities** — prevents high stats from guaranteeing success, keeping tension in the story.

## Skills demonstrated
Probabilistic graphical models (pgmpy, Variable Elimination, CPT construction), structural causal modelling and the do-operator, LLM application architecture with structured outputs and fallback chains, streaming APIs (SSE), RAG for long-range memory, vector databases, FastAPI, React/Tailwind/Cytoscape, Docker/nginx, test design, security basics for a public demo.

## References cited in the repo
Pearl & Mackenzie *The Book of Why*; Pearl *Causality* (2000); Pearl (1995) backdoor/frontdoor; D&D 5e Player's Handbook (modifier formula); pgmpy (Ankan & Panda 2015); Russell & Norvig AIMA ch. 13; Fillmore frame semantics.
