// Wire translation for the six client-by-provider paths, plus the
// normalized provider answer and the client-side rendering.
// Same-wire pairs relay the request verbatim (only the model name and,
// for a streamed OpenAI call, stream_options change). Cross-wire pairs
// parse the provider answer into text, usage counts and a stop reason,
// then render it again in the client's wire. Every request field is
// carried across untouched: system prompt, user text including images
// byte-identical, tool definitions in order, stop sequences, max_tokens
// and JSON mode.
import type { ProviderDir } from "./providers.ts";

export interface Norm {
  text: string;
  input: number;
  output: number;
  stop: string;
}
export interface ProviderCall {
  url: string;
  headers: Record<string, string>;
  body: Record<string, any>;
}

const CREATED = 1767225600;

function isRec(v: any): v is Record<string, any> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function clone(v: any): any {
  return JSON.parse(JSON.stringify(v));
}

// --- client request field extraction -------------------------------------

function openaiSystemText(body: any): string {
  const parts: string[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    if (!isRec(m) || m.role !== "system") continue;
    const t = openaiContentText(m.content);
    if (t !== "") parts.push(t);
  }
  return parts.join("\n");
}

function openaiContentText(content: any): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const b of content) {
    if (isRec(b) && b.type === "text" && typeof b.text === "string") parts.push(b.text);
  }
  return parts.join("\n");
}

function anthropicBlocksText(content: any): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const b of content) {
    if (isRec(b) && b.type === "text" && typeof b.text === "string") parts.push(b.text);
  }
  return parts.join("\n");
}

function openaiMaxTokens(body: any): number | null {
  if (typeof body.max_tokens === "number") return body.max_tokens;
  if (typeof body.max_completion_tokens === "number") return body.max_completion_tokens;
  return null;
}

function openaiStops(body: any): string[] {
  if (typeof body.stop === "string") return [body.stop];
  if (Array.isArray(body.stop)) return body.stop.filter((s: any) => typeof s === "string");
  return [];
}

function dataUrlParts(url: string): { mime: string; data: string } | null {
  const m = /^data:([^;]+);base64,(.*)$/s.exec(url ?? "");
  if (!m) return null;
  return { mime: m[1], data: m[2] };
}

// OpenAI content blocks -> Anthropic content blocks (image bytes identical).
function openaiBlocksToAnthropic(content: any): any[] {
  if (typeof content === "string") return [{ type: "text", text: content }];
  if (!Array.isArray(content)) return [];
  const out: any[] = [];
  for (const b of content) {
    if (!isRec(b)) continue;
    if (b.type === "text" && typeof b.text === "string") {
      out.push({ type: "text", text: b.text });
    } else if (b.type === "image_url" && isRec(b.image_url) && typeof b.image_url.url === "string") {
      const parts = dataUrlParts(b.image_url.url);
      if (parts) {
        out.push({ type: "image", source: { type: "base64", media_type: parts.mime, data: parts.data } });
      } else {
        out.push({ type: "image", source: { type: "url", url: b.image_url.url } });
      }
    }
  }
  return out;
}

// Anthropic content blocks -> OpenAI content blocks (image bytes identical).
function anthropicBlocksToOpenai(content: any): any[] {
  if (typeof content === "string") return [{ type: "text", text: content }];
  if (!Array.isArray(content)) return [];
  const out: any[] = [];
  for (const b of content) {
    if (!isRec(b)) continue;
    if (b.type === "text" && typeof b.text === "string") {
      out.push({ type: "text", text: b.text });
    } else if (b.type === "image" && isRec(b.source)) {
      if (b.source.type === "base64" && typeof b.source.data === "string") {
        const mime = typeof b.source.media_type === "string" ? b.source.media_type : "image/png";
        out.push({ type: "image_url", image_url: { url: `data:${mime};base64,${b.source.data}` } });
      } else if (b.source.type === "url" && typeof b.source.url === "string") {
        out.push({ type: "image_url", image_url: { url: b.source.url } });
      }
    }
  }
  return out;
}

function openaiMessagesToAnthropic(body: any): any[] {
  const out: any[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    if (!isRec(m) || m.role === "system") continue;
    out.push({ role: m.role === "assistant" ? "assistant" : "user", content: openaiBlocksToAnthropic(m.content) });
  }
  return out;
}

function anthropicMessagesToOpenai(body: any): any[] {
  const out: any[] = [];
  const system = anthropicBlocksText(body.system);
  if (system !== "") out.push({ role: "system", content: system });
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    if (!isRec(m)) continue;
    out.push({ role: m.role === "assistant" ? "assistant" : "user", content: anthropicBlocksToOpenai(m.content) });
  }
  return out;
}

function openaiToolsToAnthropic(body: any): any[] | undefined {
  if (!Array.isArray(body.tools)) return undefined;
  const out: any[] = [];
  for (const t of body.tools) {
    if (!isRec(t) || !isRec(t.function) || typeof t.function.name !== "string") continue;
    const tool: any = { name: t.function.name, input_schema: t.function.parameters ?? { type: "object" } };
    if (typeof t.function.description === "string") tool.description = t.function.description;
    out.push(tool);
  }
  return out;
}

function anthropicToolsToOpenai(body: any): any[] | undefined {
  if (!Array.isArray(body.tools)) return undefined;
  const out: any[] = [];
  for (const t of body.tools) {
    if (!isRec(t) || typeof t.name !== "string") continue;
    const fn: any = { name: t.name, parameters: t.input_schema ?? { type: "object" } };
    if (typeof t.description === "string") fn.description = t.description;
    out.push({ type: "function", function: fn });
  }
  return out;
}

function openaiToolChoiceToAnthropic(body: any): any {
  const tc = body.tool_choice;
  if (tc === undefined) return undefined;
  if (tc === "auto") return { type: "auto" };
  if (tc === "none") return { type: "none" };
  if (tc === "required") return { type: "any" };
  if (isRec(tc) && tc.type === "function" && isRec(tc.function) && typeof tc.function.name === "string") {
    return { type: "tool", name: tc.function.name };
  }
  return { type: "auto" };
}

function anthropicToolChoiceToOpenai(body: any): any {
  const tc = body.tool_choice;
  if (tc === undefined) return undefined;
  if (!isRec(tc)) return undefined;
  if (tc.type === "auto") return "auto";
  if (tc.type === "any") return "required";
  if (tc.type === "none") return "none";
  if (tc.type === "tool" && typeof tc.name === "string") return { type: "function", function: { name: tc.name } };
  return undefined;
}

// --- request builders: client body -> provider wire ----------------------

function toOpenai(clientApi: string, body: any, model: string, stream: boolean): Record<string, any> {
  let out: any;
  if (clientApi === "openai") {
    out = clone(body);
  } else {
    out = { model, messages: anthropicMessagesToOpenai(body) };
    const tools = anthropicToolsToOpenai(body);
    if (tools !== undefined) out.tools = tools;
    const tc = anthropicToolChoiceToOpenai(body);
    if (tc !== undefined) out.tool_choice = tc;
    if (typeof body.max_tokens === "number") out.max_tokens = body.max_tokens;
    if (Array.isArray(body.stop_sequences)) out.stop = body.stop_sequences;
    if (typeof body.temperature === "number") out.temperature = body.temperature;
    if (typeof body.top_p === "number") out.top_p = body.top_p;
  }
  out.model = model;
  out.stream = stream;
  if (stream) out.stream_options = { include_usage: true };
  return out;
}

function toAnthropic(clientApi: string, body: any, model: string, stream: boolean): Record<string, any> {
  let out: any;
  if (clientApi === "anthropic") {
    out = clone(body);
  } else {
    out = { model, messages: openaiMessagesToAnthropic(body) };
    const system = openaiSystemText(body);
    if (system !== "") out.system = system;
    const tools = openaiToolsToAnthropic(body);
    if (tools !== undefined) out.tools = tools;
    const tc = openaiToolChoiceToAnthropic(body);
    if (tc !== undefined) out.tool_choice = tc;
    const max = openaiMaxTokens(body);
    if (max !== null) out.max_tokens = max;
    const stops = openaiStops(body);
    if (stops.length > 0) out.stop_sequences = stops;
    if (typeof body.temperature === "number") out.temperature = body.temperature;
    if (typeof body.top_p === "number") out.top_p = body.top_p;
  }
  out.model = model;
  out.stream = stream;
  return out;
}

function toGemini(clientApi: string, body: any, model: string): Record<string, any> {
  const contents: any[] = [];
  let systemText = "";
  let maxTokens: number | null = null;
  let stops: string[] = [];
  let jsonMode = false;
  let temperature: number | undefined;
  let topP: number | undefined;
  let tools: any[] | undefined;
  if (clientApi === "openai") {
    systemText = openaiSystemText(body);
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const m of messages) {
      if (!isRec(m) || m.role === "system") continue;
      contents.push({ role: m.role === "assistant" ? "model" : "user", parts: openaiBlocksToGemini(m.content) });
    }
    maxTokens = openaiMaxTokens(body);
    stops = openaiStops(body);
    jsonMode = isRec(body.response_format) && (body.response_format.type === "json_object" || body.response_format.type === "json_schema");
    if (typeof body.temperature === "number") temperature = body.temperature;
    if (typeof body.top_p === "number") topP = body.top_p;
    const at = openaiToolsToAnthropic(body);
    if (at !== undefined) {
      tools = [{ functionDeclarations: at.map((t: any) => ({ name: t.name, description: t.description, parameters: t.input_schema })) }];
    }
  } else {
    systemText = anthropicBlocksText(body.system);
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const m of messages) {
      if (!isRec(m)) continue;
      contents.push({ role: m.role === "assistant" ? "model" : "user", parts: anthropicBlocksToGemini(m.content) });
    }
    if (typeof body.max_tokens === "number") maxTokens = body.max_tokens;
    if (Array.isArray(body.stop_sequences)) stops = body.stop_sequences.filter((s: any) => typeof s === "string");
    if (typeof body.temperature === "number") temperature = body.temperature;
    if (typeof body.top_p === "number") topP = body.top_p;
    if (Array.isArray(body.tools)) {
      tools = [{
        functionDeclarations: body.tools.filter((t: any) => isRec(t) && typeof t.name === "string").map((t: any) => ({
          name: t.name,
          description: t.description,
          parameters: t.input_schema ?? { type: "object" },
        })),
      }];
    }
  }
  const out: any = { contents };
  if (systemText !== "") out.systemInstruction = { parts: [{ text: systemText }] };
  const config: any = {};
  if (maxTokens !== null) config.maxOutputTokens = maxTokens;
  if (stops.length > 0) config.stopSequences = stops;
  if (temperature !== undefined) config.temperature = temperature;
  if (topP !== undefined) config.topP = topP;
  if (jsonMode) config.responseMimeType = "application/json";
  if (Object.keys(config).length > 0) out.generationConfig = config;
  if (tools !== undefined) out.tools = tools;
  return out;
}

function openaiBlocksToGemini(content: any): any[] {
  if (typeof content === "string") return [{ text: content }];
  if (!Array.isArray(content)) return [];
  const out: any[] = [];
  for (const b of content) {
    if (!isRec(b)) continue;
    if (b.type === "text" && typeof b.text === "string") out.push({ text: b.text });
    else if (b.type === "image_url" && isRec(b.image_url) && typeof b.image_url.url === "string") {
      const parts = dataUrlParts(b.image_url.url);
      if (parts) out.push({ inlineData: { mimeType: parts.mime, data: parts.data } });
      else out.push({ fileData: { fileUri: b.image_url.url } });
    }
  }
  return out;
}

function anthropicBlocksToGemini(content: any): any[] {
  if (typeof content === "string") return [{ text: content }];
  if (!Array.isArray(content)) return [];
  const out: any[] = [];
  for (const b of content) {
    if (!isRec(b)) continue;
    if (b.type === "text" && typeof b.text === "string") out.push({ text: b.text });
    else if (b.type === "image" && isRec(b.source)) {
      if (b.source.type === "base64" && typeof b.source.data === "string") {
        out.push({ inlineData: { mimeType: typeof b.source.media_type === "string" ? b.source.media_type : "image/png", data: b.source.data } });
      } else if (b.source.type === "url" && typeof b.source.url === "string") {
        out.push({ fileData: { fileUri: b.source.url } });
      }
    }
  }
  return out;
}

function providerAuthHeaders(dir: ProviderDir): Record<string, string> {
  if (dir.auth === "bearer") return { authorization: `Bearer ${dir.key}` };
  if (dir.auth === "x-api-key") return { "x-api-key": dir.key };
  if (dir.auth === "x-goog-api-key") return { "x-goog-api-key": dir.key };
  return {};
}

export function buildProviderRequest(
  envUrl: string,
  clientApi: string,
  body: any,
  dir: ProviderDir,
  model: string,
  stream: boolean,
): ProviderCall {
  const headers: Record<string, string> = { "content-type": "application/json", ...providerAuthHeaders(dir) };
  if (dir.wire === "openai") {
    return { url: `${envUrl}${dir.path}/v1/chat/completions`, headers, body: toOpenai(clientApi, body, model, stream) };
  }
  if (dir.wire === "anthropic") {
    return { url: `${envUrl}${dir.path}/v1/messages`, headers, body: toAnthropic(clientApi, body, model, stream) };
  }
  const verb = stream ? "streamGenerateContent?alt=sse" : "generateContent";
  return {
    url: `${envUrl}${dir.path}/v1beta/models/${encodeURIComponent(model)}:${verb}`,
    headers,
    body: toGemini(clientApi, body, model),
  };
}

// --- provider answer parsing ---------------------------------------------

export interface SseFrame {
  event: string | null;
  data: string;
}

export function parseSse(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, "\n");
  for (const chunk of normalized.split("\n\n")) {
    if (chunk.trim() === "") continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of chunk.split("\n")) {
      if (line.startsWith("event:")) event = line.slice("event:".length).trim();
      else if (line.startsWith("data:")) data.push(line.slice("data:".length).replace(/^ /, ""));
    }
    if (data.length > 0) frames.push({ event, data: data.join("\n") });
  }
  return frames;
}

function geminiPartsText(parts: any): string {
  if (!Array.isArray(parts)) return "";
  const out: string[] = [];
  for (const p of parts) {
    if (isRec(p) && typeof p.text === "string") out.push(p.text);
  }
  return out.join("");
}

function parseOpenaiAnswer(stream: boolean, text: string): Norm | null {
  if (!stream) {
    let value: any;
    try {
      value = JSON.parse(text);
    } catch {
      return null;
    }
    const choice = Array.isArray(value?.choices) ? value.choices[0] : null;
    if (!isRec(choice) || !isRec(choice.message)) return null;
    const content = typeof choice.message.content === "string" ? choice.message.content : anthropicBlocksText(choice.message.content);
    const usage = isRec(value.usage) ? value.usage : {};
    return {
      text: content,
      input: typeof usage.prompt_tokens === "number" ? usage.prompt_tokens : 0,
      output: typeof usage.completion_tokens === "number" ? usage.completion_tokens : 0,
      stop: typeof choice.finish_reason === "string" ? choice.finish_reason : "stop",
    };
  }
  let out = "";
  let input = 0;
  let output = 0;
  let stop = "stop";
  for (const frame of parseSse(text)) {
    if (frame.data.trim() === "[DONE]") continue;
    let chunk: any;
    try {
      chunk = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRec(chunk)) continue;
    if (isRec(chunk.usage)) {
      if (typeof chunk.usage.prompt_tokens === "number") input = chunk.usage.prompt_tokens;
      if (typeof chunk.usage.completion_tokens === "number") output = chunk.usage.completion_tokens;
    }
    const choice = Array.isArray(chunk.choices) ? chunk.choices[0] : null;
    if (!isRec(choice)) continue;
    if (isRec(choice.delta) && typeof choice.delta.content === "string") out += choice.delta.content;
    if (typeof choice.finish_reason === "string") stop = choice.finish_reason;
  }
  return { text: out, input, output, stop };
}

function parseAnthropicAnswer(stream: boolean, text: string): Norm | null {
  if (!stream) {
    let value: any;
    try {
      value = JSON.parse(text);
    } catch {
      return null;
    }
    if (!isRec(value) || !Array.isArray(value.content)) return null;
    const usage = isRec(value.usage) ? value.usage : {};
    return {
      text: anthropicBlocksText(value.content),
      input: typeof usage.input_tokens === "number" ? usage.input_tokens : 0,
      output: typeof usage.output_tokens === "number" ? usage.output_tokens : 0,
      stop: typeof value.stop_reason === "string" ? value.stop_reason : "end_turn",
    };
  }
  let out = "";
  let input = 0;
  let output = 0;
  let stop = "end_turn";
  for (const frame of parseSse(text)) {
    let data: any;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRec(data)) continue;
    if (frame.event === "message_start" && isRec(data.message) && isRec(data.message.usage)) {
      if (typeof data.message.usage.input_tokens === "number") input = data.message.usage.input_tokens;
    } else if (frame.event === "content_block_delta" && isRec(data.delta) && data.delta.type === "text_delta" && typeof data.delta.text === "string") {
      out += data.delta.text;
    } else if (frame.event === "message_delta") {
      if (isRec(data.delta) && typeof data.delta.stop_reason === "string") stop = data.delta.stop_reason;
      if (isRec(data.usage) && typeof data.usage.output_tokens === "number") output = data.usage.output_tokens;
    }
  }
  return { text: out, input, output, stop };
}

function parseGeminiAnswer(stream: boolean, text: string): Norm | null {
  if (!stream) {
    let value: any;
    try {
      value = JSON.parse(text);
    } catch {
      return null;
    }
    const cand = Array.isArray(value?.candidates) ? value.candidates[0] : null;
    if (!isRec(cand)) return null;
    const meta = isRec(value.usageMetadata) ? value.usageMetadata : {};
    return {
      text: geminiPartsText(cand.content?.parts),
      input: typeof meta.promptTokenCount === "number" ? meta.promptTokenCount : 0,
      output: typeof meta.candidatesTokenCount === "number" ? meta.candidatesTokenCount : 0,
      stop: typeof cand.finishReason === "string" ? cand.finishReason : "STOP",
    };
  }
  let out = "";
  let input = 0;
  let output = 0;
  let stop = "STOP";
  for (const frame of parseSse(text)) {
    let data: any;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRec(data)) continue;
    const cand = Array.isArray(data.candidates) ? data.candidates[0] : null;
    if (isRec(cand)) {
      out += geminiPartsText(cand.content?.parts);
      if (typeof cand.finishReason === "string") stop = cand.finishReason;
    }
    if (isRec(data.usageMetadata)) {
      if (typeof data.usageMetadata.promptTokenCount === "number") input = data.usageMetadata.promptTokenCount;
      if (typeof data.usageMetadata.candidatesTokenCount === "number") output = data.usageMetadata.candidatesTokenCount;
    }
  }
  return { text: out, input, output, stop };
}

export function parseProviderAnswer(wire: string, stream: boolean, text: string): Norm | null {
  if (wire === "openai") return parseOpenaiAnswer(stream, text);
  if (wire === "anthropic") return parseAnthropicAnswer(stream, text);
  return parseGeminiAnswer(stream, text);
}

// --- client-side rendering -----------------------------------------------

const STOP_FAMILY = new Map<string, number>([
  ["stop", 0],
  ["end_turn", 0],
  ["STOP", 0],
  ["length", 1],
  ["max_tokens", 1],
  ["MAX_TOKENS", 1],
  ["tool_calls", 2],
  ["function_call", 2],
  ["tool_use", 2],
  ["content_filter", 3],
  ["refusal", 3],
  ["SAFETY", 3],
]);

function stopFamily(reason: string): number {
  return STOP_FAMILY.get(reason) ?? 0;
}

function split13(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [""] : pieces;
}

function sseFrame(event: string | null, data: any): string {
  return `${event === null ? "" : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function renderOpenai(stream: boolean, norm: Norm, model: string): { headers: Record<string, string>; body: string } {
  const usage = { prompt_tokens: norm.input, completion_tokens: norm.output, total_tokens: norm.input + norm.output };
  const finish = ["stop", "length", "tool_calls", "content_filter"][stopFamily(norm.stop)];
  if (!stream) {
    return {
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        id: "chatcmpl-sim",
        object: "chat.completion",
        created: CREATED,
        model,
        choices: [{ index: 0, message: { role: "assistant", content: norm.text }, finish_reason: finish, logprobs: null }],
        usage,
      }),
    };
  }
  const chunk = (delta: any, fin: string | null) => ({
    id: "chatcmpl-sim",
    object: "chat.completion.chunk",
    created: CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: fin }],
  });
  const frames: string[] = [sseFrame(null, chunk({ role: "assistant", content: "" }, null))];
  for (const piece of split13(norm.text)) frames.push(sseFrame(null, chunk({ content: piece }, null)));
  frames.push(sseFrame(null, chunk({}, finish)));
  frames.push(sseFrame(null, { id: "chatcmpl-sim", object: "chat.completion.chunk", created: CREATED, model, choices: [], usage }));
  frames.push("data: [DONE]\n\n");
  return { headers: { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache", connection: "keep-alive" }, body: frames.join("") };
}

function renderAnthropic(stream: boolean, norm: Norm, model: string): { headers: Record<string, string>; body: string } {
  const stop = ["end_turn", "max_tokens", "tool_use", "refusal"][stopFamily(norm.stop)];
  if (!stream) {
    return {
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        id: "msg_sim",
        type: "message",
        role: "assistant",
        model,
        content: [{ type: "text", text: norm.text }],
        stop_reason: stop,
        stop_sequence: null,
        usage: { input_tokens: norm.input, output_tokens: norm.output },
      }),
    };
  }
  const frames: string[] = [
    sseFrame("message_start", {
      type: "message_start",
      message: { id: "msg_sim", type: "message", role: "assistant", model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: norm.input, output_tokens: 0 } },
    }),
    sseFrame("content_block_start", { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
  ];
  for (const piece of split13(norm.text)) {
    frames.push(sseFrame("content_block_delta", { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } }));
  }
  frames.push(
    sseFrame("content_block_stop", { type: "content_block_stop", index: 0 }),
    sseFrame("message_delta", { type: "message_delta", delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: norm.output } }),
    sseFrame("message_stop", { type: "message_stop" }),
  );
  return { headers: { "content-type": "text/event-stream; charset=utf-8", "cache-control": "no-cache", connection: "keep-alive" }, body: frames.join("") };
}

export function renderClient(clientApi: string, stream: boolean, norm: Norm, model: string): { headers: Record<string, string>; body: string } {
  if (clientApi === "openai") return renderOpenai(stream, norm, model);
  return renderAnthropic(stream, norm, model);
}
