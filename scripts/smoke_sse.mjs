#!/usr/bin/env node
// Consume POST /api/ask exactly the way the Vercel widget does: fetch() + a
// manual SSE reader over the response body stream. No dependencies (Node >= 18).
//
//   node scripts/smoke_sse.mjs [BASE_URL] [question]
//
// Exits non-zero unless the stream reaches a `done` event with prose and a
// `meta` event naming the generator.

const BASE = (process.argv[2] || "http://localhost:8000").replace(/\/$/, "");
const QUESTION =
  process.argv[3] || "Which project best shows Kai's RAG skills?";

const res = await fetch(`${BASE}/api/ask`, {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ question: QUESTION, audience: "auto" }),
});

if (!res.ok || !res.body) {
  console.error(`HTTP ${res.status} — ${await res.text().catch(() => "")}`);
  process.exit(1);
}

const decoder = new TextDecoder();
let buf = "";
let prose = "";
let meta = null;
let error = null;
const seen = new Set();

const handle = (event, data) => {
  seen.add(event);
  if (event === "token") {
    try {
      const p = JSON.parse(data);
      if (p.replace) prose = p.text;
      else prose += p.text;
    } catch {
      /* ignore */
    }
  } else if (event === "meta") {
    try {
      meta = JSON.parse(data);
    } catch {
      /* ignore */
    }
  } else if (event === "error") {
    error = data;
  }
};

for await (const chunk of res.body) {
  buf += decoder.decode(chunk, { stream: true });
  const frames = buf.split("\n\n");
  buf = frames.pop() ?? "";
  for (const frame of frames) {
    let ev = "message";
    const dataLines = [];
    for (const line of frame.split("\n")) {
      if (line.startsWith("event:")) ev = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (dataLines.length) handle(ev, dataLines.join("\n"));
  }
}

console.log("events :", [...seen].join(", "));
console.log("prose  :", JSON.stringify(prose.slice(0, 120) + (prose.length > 120 ? "…" : "")));
console.log("meta   :", meta && `${meta.generator} / ${meta.audience_resolved} / ${meta.latency_ms}ms`);

if (error) {
  console.error("FAIL — SSE error event:", error);
  process.exit(1);
}
if (!seen.has("done")) {
  console.error("FAIL — stream never emitted 'done'");
  process.exit(1);
}
if (!prose && !(meta && meta.in_scope === false)) {
  console.error("FAIL — no prose and not an in-scope decline");
  process.exit(1);
}
if (!meta || !meta.generator) {
  console.error("FAIL — no meta.generator");
  process.exit(1);
}
console.log("\nok — SSE stream well-formed");
