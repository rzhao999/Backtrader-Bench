#!/usr/bin/env node
// Run MCQs through the Cursor SDK while failing immediately on any tool call.
//
// Cursor does not document a true SDK/API "disable all tools" switch. This
// runner uses the strictest documented local SDK setup: no ambient settings,
// no inline MCP/subagents, sandbox enabled, and stream inspection that cancels
// the run if any tool call appears.

import { Agent } from "@cursor/sdk";
import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const ANSWER_RE = /<<<\s*([A-D])\s*>>>/i;
const DEFAULT_MODEL = "composer-2";
const DEFAULT_INPUT = "backtrader_mcq_64_hard.jsonl";
const DEFAULT_OUTPUT_DIR = "cursor_sdk_notools_runs";
const DIFFICULTY_ORDER = new Map([
  ["easy", 0],
  ["medium", 1],
  ["hard", 2],
  ["unknown", 3],
]);

function usage() {
  console.error(`Usage:
  node run_cursor_sdk_mcq_notool.mjs [options]

Options:
  --input <path>       JSON or JSONL MCQ file (default: ${DEFAULT_INPUT})
  --output-dir <dir>   Output directory (default: ${DEFAULT_OUTPUT_DIR})
  --model <id>         Cursor model id (default: ${DEFAULT_MODEL})
  --runs <n>           Number of full question-set runs (default: 1)
  --timeout <seconds>  Per-question timeout (default: 120)

Requires:
  npm install @cursor/sdk
  export CURSOR_API_KEY="cursor_..."
`);
}

function parseArgs(argv) {
  const args = {
    input: DEFAULT_INPUT,
    outputDir: DEFAULT_OUTPUT_DIR,
    model: DEFAULT_MODEL,
    runs: 1,
    timeoutMs: 60_000,
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    const readValue = () => {
      if (i + 1 >= argv.length || argv[i + 1].startsWith("--")) {
        throw new Error(`${arg} requires a value`);
      }
      i += 1;
      return argv[i];
    };

    if (arg === "--help" || arg === "-h") {
      usage();
      process.exit(0);
    } else if (arg === "--input") {
      args.input = readValue();
    } else if (arg.startsWith("--input=")) {
      args.input = arg.slice("--input=".length);
    } else if (arg === "--output-dir") {
      args.outputDir = readValue();
    } else if (arg.startsWith("--output-dir=")) {
      args.outputDir = arg.slice("--output-dir=".length);
    } else if (arg === "--model") {
      args.model = readValue();
    } else if (arg.startsWith("--model=")) {
      args.model = arg.slice("--model=".length);
    } else if (arg === "--runs") {
      args.runs = Number(readValue());
    } else if (arg.startsWith("--runs=")) {
      args.runs = Number(arg.slice("--runs=".length));
    } else if (arg === "--timeout") {
      args.timeoutMs = Number(readValue()) * 1000;
    } else if (arg.startsWith("--timeout=")) {
      args.timeoutMs = Number(arg.slice("--timeout=".length)) * 1000;
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (!Number.isFinite(args.timeoutMs) || args.timeoutMs <= 0) {
    throw new Error("--timeout must be a positive number of seconds");
  }
  if (!Number.isInteger(args.runs) || args.runs <= 0) {
    throw new Error("--runs must be a positive integer");
  }
  return args;
}

function safePathSegment(value) {
  return String(value || "unknown")
    .trim()
    .replace(/[^A-Za-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "") || "unknown";
}

async function loadQuestions(inputPath) {
  const raw = await fs.readFile(inputPath, "utf8");
  const trimmed = raw.trim();
  if (!trimmed) return [];

  try {
    const parsed = JSON.parse(trimmed);
    if (Array.isArray(parsed)) return parsed.filter((item) => item && typeof item === "object");
    if (parsed && typeof parsed === "object") return [parsed];
  } catch {
    // Fall through to JSONL.
  }

  return trimmed
    .split(/\r?\n/)
    .filter((line) => line.trim())
    .map((line, index) => {
      try {
        return JSON.parse(line);
      } catch (error) {
        throw new Error(`Invalid JSONL on line ${index + 1}: ${error.message}`);
      }
    })
    .filter((item) => item && typeof item === "object");
}

function buildPrompt(question) {
  return `You are a no-tools multiple-choice answerer. This is a one-shot Cursor SDK agent run.

Do not use tools. Do not call web search, web fetch, Bash, Python, MCPs, file search, file read, file write, browser, workspace inspection, or any other external capability.

Use only the question text and your internal reasoning. Reply with exactly <<< X >>> where X is one of A, B, C, or D. No explanation, no code, no extra text.

${question}`;
}

function parseAnswer(text) {
  const match = ANSWER_RE.exec(text || "");
  return match ? match[1].toUpperCase() : null;
}

function normalizeDifficulty(questionObj) {
  const difficulty = questionObj.difficulty;
  return typeof difficulty === "string" && difficulty.trim() ? difficulty.trim() : "unknown";
}

function estimateTokens(text) {
  const value = String(text || "");
  if (!value) return 0;

  // Cursor SDK local runs currently do not expose exact token usage. This
  // approximation keeps reports numeric while marking the source as estimated.
  const chunks = value.match(/[A-Za-z0-9_]+|[^\sA-Za-z0-9_]/g) || [];
  return Math.max(1, Math.ceil(chunks.length * 1.25));
}

function estimatedUsage(prompt, response, model) {
  const inputTokens = estimateTokens(prompt);
  const outputTokens = estimateTokens(response);
  return {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: inputTokens + outputTokens,
    cache_read_tokens: null,
    cache_creation_tokens: null,
    model,
    source: "estimated",
  };
}

function pickNumber(obj, keys) {
  for (const key of keys) {
    const value = obj?.[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function normalizeUsage(payload, model) {
  if (!payload || typeof payload !== "object") return null;
  const usage = payload.usage && typeof payload.usage === "object" ? payload.usage : payload;
  const inputTokens = pickNumber(usage, [
    "input_tokens",
    "inputTokens",
    "prompt_tokens",
    "promptTokens",
    "promptTokenCount",
  ]);
  let outputTokens = pickNumber(usage, [
    "output_tokens",
    "outputTokens",
    "completion_tokens",
    "completionTokens",
    "candidatesTokenCount",
  ]);
  const totalTokens = pickNumber(usage, ["total_tokens", "totalTokens", "totalTokenCount"]);

  if (outputTokens === null && inputTokens !== null && totalTokens !== null) {
    outputTokens = totalTokens - inputTokens;
  }
  if (inputTokens === null && outputTokens === null && totalTokens === null) return null;

  return {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    total_tokens: totalTokens,
    cache_read_tokens: pickNumber(usage, [
      "cache_read_tokens",
      "cacheReadTokens",
      "cache_read_input_tokens",
      "cacheReadInputTokens",
      "cached_input_tokens",
      "cachedInputTokens",
    ]),
    cache_creation_tokens: pickNumber(usage, [
      "cache_creation_tokens",
      "cacheCreationTokens",
      "cache_write_input_tokens",
      "cacheWriteInputTokens",
    ]),
    model: typeof payload.model === "string" ? payload.model : model,
    source: "exact",
  };
}

function extractUsage(payload, model, seen = new WeakSet()) {
  if (!payload || typeof payload !== "object") return null;
  if (seen.has(payload)) return null;
  seen.add(payload);

  const direct = normalizeUsage(payload, model);
  if (direct) return direct;

  for (const value of Object.values(payload)) {
    if (value && typeof value === "object") {
      const nested = extractUsage(value, model, seen);
      if (nested) return nested;
    }
  }
  return null;
}

function aggregateAccuracyByDifficulty(records) {
  const grouped = new Map();
  for (const record of records) {
    if (!record.expected_answer) continue;
    const difficulty = record.difficulty || "unknown";
    if (!grouped.has(difficulty)) grouped.set(difficulty, []);
    grouped.get(difficulty).push(record);
  }

  const entries = [...grouped.entries()].sort(([a], [b]) => {
    const ao = DIFFICULTY_ORDER.get(a.toLowerCase()) ?? DIFFICULTY_ORDER.size;
    const bo = DIFFICULTY_ORDER.get(b.toLowerCase()) ?? DIFFICULTY_ORDER.size;
    return ao - bo || a.localeCompare(b);
  });

  return Object.fromEntries(
    entries.map(([difficulty, items]) => {
      const answered = items.filter((record) => record.parsed_answer !== null);
      const correct = items.filter((record) => record.is_correct);
      return [
        difficulty,
        {
          gradable: items.length,
          answered: answered.length,
          correct: correct.length,
          accuracy_pct: items.length ? Math.round((correct.length / items.length) * 10000) / 100 : 0,
          answered_pct: items.length ? Math.round((answered.length / items.length) * 10000) / 100 : 0,
        },
      ];
    }),
  );
}

function aggregateUsage(records) {
  const usages = records.map((record) => record.usage).filter((usage) => usage && typeof usage === "object");
  const numericUsages = usages.filter((usage) =>
    ["input_tokens", "output_tokens", "total_tokens", "cache_read_tokens", "cache_creation_tokens"].some(
      (key) => typeof usage[key] === "number",
    ),
  );
  if (numericUsages.length === 0) return null;

  const totals = {
    input_tokens: 0,
    output_tokens: 0,
    total_tokens: 0,
    cache_read_tokens: 0,
    cache_creation_tokens: 0,
  };
  const counts = Object.fromEntries(Object.keys(totals).map((key) => [key, 0]));
  const models = new Set();

  for (const usage of numericUsages) {
    if (typeof usage.model === "string") models.add(usage.model);
    for (const key of Object.keys(totals)) {
      const value = usage[key];
      if (typeof value === "number" && Number.isFinite(value)) {
        totals[key] += value;
        counts[key] += 1;
      }
    }
  }

  return {
    n_with_usage: numericUsages.length,
    models: [...models].sort(),
    totals,
    averages: Object.fromEntries(
      Object.keys(totals).map((key) => [`avg_${key}`, counts[key] ? Math.round((totals[key] / counts[key]) * 100) / 100 : null]),
    ),
  };
}

function eventLooksLikeToolUse(event) {
  if (!event || typeof event !== "object") return false;
  const type = String(event.type || event.kind || "");
  if (type.includes("tool")) return true;

  // Some SDK event shapes can nest tool-call details in payload-like fields.
  const serialized = JSON.stringify(event);
  return /"tool_call"|"toolCall"|"tool_use"|"toolUse"/.test(serialized);
}

function collectAssistantText(event) {
  if (event?.type !== "assistant") return "";
  let text = "";
  for (const block of event.message?.content || []) {
    if (block?.type === "text" && typeof block.text === "string") {
      text += block.text;
    }
  }
  return text;
}

function eventSummary(event) {
  const summary = {
    type: event?.type ?? null,
    kind: event?.kind ?? null,
    keys: event && typeof event === "object" ? Object.keys(event).sort() : [],
    looks_like_tool_use: eventLooksLikeToolUse(event),
  };
  if (event?.message?.content) {
    summary.message_content_types = event.message.content.map((block) => block?.type ?? null);
  }
  return summary;
}

async function waitWithTimeout(promise, timeoutMs, onTimeout) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(async () => {
          await onTimeout?.();
          reject(new Error(`Timed out after ${timeoutMs / 1000}s`));
        }, timeoutMs);
      }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

async function runOne({ agent, questionObj, index, total, timeoutMs, outputRoot }) {
  const question = String(questionObj.question || "").trim();
  const expected = questionObj.answer ? String(questionObj.answer).toUpperCase() : null;
  const difficulty = normalizeDifficulty(questionObj);
  const qDir = path.join(outputRoot, `q_${String(index).padStart(3, "0")}`);
  await fs.mkdir(qDir, { recursive: true });

  const prompt = buildPrompt(question);
  await fs.writeFile(path.join(qDir, "prompt.txt"), prompt + "\n", "utf8");

  const started = Date.now();
  let run;
  let stdout = "";
  let stderr = "";
  let status = "ok";
  let toolAttempted = false;
  let usage = null;
  const eventSummaries = [];
  const toolEvents = [];

  try {
    run = await agent.send(prompt, { mcpServers: {} });
    const stream = run.stream();

    await waitWithTimeout(
      (async () => {
        for await (const event of stream) {
          const summary = eventSummary(event);
          eventSummaries.push(summary);
          usage = extractUsage(event, agent.model?.id) || usage;
          if (eventLooksLikeToolUse(event)) {
            toolAttempted = true;
            toolEvents.push(summary);
            status = "tool_attempted";
            stderr += `Tool use attempted and blocked: ${JSON.stringify(event)}\n`;
            if (run.supports?.("cancel")) await run.cancel();
            break;
          }
          stdout += collectAssistantText(event);
        }
        const result = await run.wait();
        usage = extractUsage(result, result?.model?.id || agent.model?.id) || usage;
        if (result?.status && result.status !== "finished" && status === "ok") {
          status = result.status;
        }
      })(),
      timeoutMs,
      async () => {
        if (run?.supports?.("cancel")) await run.cancel();
      },
    );
  } catch (error) {
    if (!toolAttempted) {
      status = "error";
      stderr += `${error.name || "Error"}: ${error.message || String(error)}\n`;
    }
  }

  const parsed = parseAnswer(stdout);
  const elapsedSec = Math.round((Date.now() - started) / 100) / 10;
  const recordUsage = usage || estimatedUsage(prompt, stdout, agent.model?.id ?? null);
  const model = recordUsage.model ?? agent.model?.id ?? null;
  const record = {
    index,
    model,
    status,
    difficulty,
    elapsed_sec: elapsedSec,
    parsed_answer: parsed,
    expected_answer: expected,
    is_correct: expected ? parsed === expected : null,
    question_dir: qDir,
    usage: recordUsage,
    event_count: eventSummaries.length,
    event_types: [...new Set(eventSummaries.map((event) => event.type).filter(Boolean))],
    tool_event_count: toolEvents.length,
  };

  await fs.writeFile(path.join(qDir, "response.txt"), stdout, "utf8");
  await fs.writeFile(path.join(qDir, "stderr.txt"), stderr, "utf8");
  await fs.writeFile(
    path.join(qDir, "event_summaries.jsonl"),
    eventSummaries.map((event) => JSON.stringify(event)).join("\n") + "\n",
    "utf8",
  );
  await fs.writeFile(
    path.join(qDir, "tool_events.json"),
    JSON.stringify(toolEvents, null, 2) + "\n",
    "utf8",
  );
  console.log(
    `[CursorSDKNoTools] [${index}/${total}] model=${model} status=${status} difficulty=${difficulty} parsed=${parsed} expected=${expected} correct=${record.is_correct} tool_events=${toolEvents.length} tokens_in=${record.usage.input_tokens} tokens_out=${record.usage.output_tokens} token_source=${record.usage.source} elapsed=${elapsedSec}s`,
  );
  return record;
}

function buildReport({ args, inputPath, records }) {
  const gradable = records.filter((record) => record.expected_answer);
  const correct = gradable.filter((record) => record.is_correct);
  const toolAttempts = records.filter((record) => record.status === "tool_attempted");
  const usageTotals = aggregateUsage(records);
  const modelsUsed = [...new Set(records.map((record) => record.model).filter(Boolean))].sort();
  const report = {
    agent: "CursorSDKNoTools",
    model: args.model,
    models_used: modelsUsed,
    input_file: inputPath,
    total_questions: records.length,
    gradable: gradable.length,
    correct: correct.length,
    accuracy_pct: gradable.length ? Math.round((correct.length / gradable.length) * 10000) / 100 : 0,
    accuracy_by_difficulty: aggregateAccuracyByDifficulty(records),
    tool_attempts: toolAttempts.length,
    per_question: records,
  };
  if (usageTotals) report.usage_totals = usageTotals;
  return report;
}

function summarizeReports({ args, inputPath, batchRoot, reports }) {
  const accuracyValues = reports.map((report) => report.accuracy_pct);
  const correctValues = reports.map((report) => report.correct);
  const toolAttemptValues = reports.map((report) => report.tool_attempts);
  const avg = (values) => values.length ? Math.round((values.reduce((sum, value) => sum + value, 0) / values.length) * 100) / 100 : 0;
  const usageReports = reports.map((report) => report.usage_totals).filter(Boolean);
  const difficulties = [...new Set(reports.flatMap((report) => Object.keys(report.accuracy_by_difficulty || {})))].sort(
    (a, b) => {
      const ao = DIFFICULTY_ORDER.get(a.toLowerCase()) ?? DIFFICULTY_ORDER.size;
      const bo = DIFFICULTY_ORDER.get(b.toLowerCase()) ?? DIFFICULTY_ORDER.size;
      return ao - bo || a.localeCompare(b);
    },
  );
  const accuracyByDifficulty = Object.fromEntries(
    difficulties.map((difficulty) => {
      const stats = reports
        .map((report) => report.accuracy_by_difficulty?.[difficulty])
        .filter((item) => item && typeof item === "object");
      return [
        difficulty,
        {
          runs_with_level: stats.length,
          gradable_per_run: stats[0]?.gradable ?? 0,
          avg_answered: avg(stats.map((item) => item.answered)),
          avg_correct: avg(stats.map((item) => item.correct)),
          avg_accuracy_pct: avg(stats.map((item) => item.accuracy_pct)),
          avg_answered_pct: avg(stats.map((item) => item.answered_pct)),
        },
      ];
    }),
  );

  const summary = {
    agent: "CursorSDKNoTools",
    model: args.model,
    input_file: inputPath,
    output_root: batchRoot,
    runs_requested: args.runs,
    runs_completed: reports.length,
    total_questions_per_run: reports[0]?.total_questions ?? 0,
    gradable_per_run: reports[0]?.gradable ?? 0,
    averages: {
      accuracy_pct: avg(accuracyValues),
      correct: avg(correctValues),
      tool_attempts: avg(toolAttemptValues),
      accuracy_by_difficulty: accuracyByDifficulty,
    },
    runs: reports.map((report, index) => ({
      run_index: index + 1,
      output_dir: report.output_dir,
      correct: report.correct,
      gradable: report.gradable,
      accuracy_pct: report.accuracy_pct,
      accuracy_by_difficulty: report.accuracy_by_difficulty,
      tool_attempts: report.tool_attempts,
    })),
  };

  if (usageReports.length > 0) {
    summary.averages.tokens = {
      input_tokens: avg(usageReports.map((usage) => usage.totals.input_tokens)),
      output_tokens: avg(usageReports.map((usage) => usage.totals.output_tokens)),
      total_tokens: avg(usageReports.map((usage) => usage.totals.total_tokens)),
    };
  }

  return summary;
}

async function runQuestionSet({ args, inputPath, questions, outputRoot, runIndex }) {
  await fs.mkdir(outputRoot, { recursive: true });
  console.log(`[CursorSDKNoTools] Run ${runIndex}/${args.runs} output: ${outputRoot}`);

  const agent = await Agent.create({
    apiKey: process.env.CURSOR_API_KEY,
    model: { id: args.model },
    local: {
      cwd: process.cwd(),
      settingSources: [],
      sandboxOptions: { enabled: true },
    },
    mcpServers: {},
    agents: {},
  });

  const records = [];
  try {
    for (let i = 0; i < questions.length; i += 1) {
      records.push(
        await runOne({
          agent,
          questionObj: questions[i],
          index: i + 1,
          total: questions.length,
          timeoutMs: args.timeoutMs,
          outputRoot,
        }),
      );
      await fs.appendFile(
        path.join(outputRoot, "summary.jsonl"),
        JSON.stringify(records.at(-1)) + "\n",
        "utf8",
      );
    }
  } finally {
    await agent[Symbol.asyncDispose]?.();
  }

  const report = buildReport({ args, inputPath, records });
  report.run_index = runIndex;
  report.output_dir = outputRoot;
  await fs.writeFile(path.join(outputRoot, "report.json"), JSON.stringify(report, null, 2) + "\n", "utf8");

  for (const [difficulty, stats] of Object.entries(report.accuracy_by_difficulty)) {
    console.log(
      `[CursorSDKNoTools] Accuracy (${difficulty}): ${stats.correct}/${stats.gradable} (${stats.accuracy_pct.toFixed(2)}%). Answered: ${stats.answered}/${stats.gradable} (${stats.answered_pct.toFixed(2)}%).`,
    );
  }
  console.log(
    `[CursorSDKNoTools] Model: ${args.model}. Models used: ${report.models_used.length ? report.models_used.join(", ") : "unknown"}. Accuracy: ${report.correct}/${report.gradable} (${report.accuracy_pct.toFixed(2)}%). Tool attempts: ${report.tool_attempts}.`,
  );
  if (report.usage_totals) {
    console.log(
      `[CursorSDKNoTools] Tokens: in=${report.usage_totals.totals.input_tokens} out=${report.usage_totals.totals.output_tokens} total=${report.usage_totals.totals.total_tokens}.`,
    );
  }
  return report;
}

async function main() {
  const args = parseArgs(process.argv);
  if (!process.env.CURSOR_API_KEY) {
    throw new Error("Missing CURSOR_API_KEY");
  }

  const inputPath = path.resolve(args.input);
  const questions = await loadQuestions(inputPath);
  if (questions.length === 0) throw new Error(`No questions found in ${inputPath}`);

  const timestamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\..*/, "").replace("T", "_");
  const batchId = `${timestamp}_${safePathSegment(args.model)}`;
  const batchRoot = path.resolve(args.outputDir, batchId);

  console.log(`[CursorSDKNoTools] Input: ${inputPath}`);
  console.log(`[CursorSDKNoTools] Questions: ${questions.length}`);
  console.log(`[CursorSDKNoTools] Output: ${batchRoot}`);
  console.log(`[CursorSDKNoTools] Model: ${args.model}`);
  console.log(`[CursorSDKNoTools] Runs: ${args.runs}`);

  const reports = [];
  for (let runIndex = 1; runIndex <= args.runs; runIndex += 1) {
    const outputRoot = args.runs === 1 ? batchRoot : path.join(batchRoot, `run_${String(runIndex).padStart(3, "0")}`);
    reports.push(await runQuestionSet({ args, inputPath, questions, outputRoot, runIndex }));
  }

  const summary = summarizeReports({ args, inputPath, batchRoot, reports });
  await fs.writeFile(path.join(batchRoot, "summary.json"), JSON.stringify(summary, null, 2) + "\n", "utf8");
  console.log(
    `[CursorSDKNoTools] Average over ${summary.runs_completed} run(s): accuracy=${summary.averages.accuracy_pct.toFixed(2)}%, correct=${summary.averages.correct}/${summary.gradable_per_run}, tool_attempts=${summary.averages.tool_attempts}.`,
  );
  if (reports.some((report) => report.tool_attempts > 0)) process.exitCode = 1;
}

main().catch((error) => {
  console.error(`${error.name || "Error"}: ${error.message || String(error)}`);
  process.exitCode = 1;
});
