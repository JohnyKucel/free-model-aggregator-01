// Free-model aggregator: one server for the OpenAI Chat Completions API
// and the Anthropic Messages API over ten simulated free-tier providers.
// Startup reads the brief, probes each provider's catalog route in
// order, builds the routing plan from the workload manifest, registers
// port plus catalog with the env, serves the client's requests one at a
// time with sequential failover, and exits 0 when the env says the
// workload is done. No wall clock or unseeded randomness reaches any
// request: the only timestamp in any answer is the fixed creation time.
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import {
  buildPlan,
  freeModelIds,
  parseBrief,
  probeCatalog,
  type Brief,
  type Catalog,
  type PlanEntry,
  type ProviderDir,
} from "./lib/providers.ts";
import { buildProviderRequest, parseProviderAnswer, renderClient } from "./lib/wire.ts";

const CREATED = 1767225600;
const EMPTY_CATALOG: Catalog = { providers: [] };

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function loadBrief(): Brief | null {
  const file = process.env.MAKEFUN_PROMPT_FILE;
  if (!file) return null;
  try {
    return parseBrief(readFileSync(file, "utf8"));
  } catch {
    return null;
  }
}

async function loadCatalog(brief: Brief | null): Promise<Catalog> {
  const envUrl = process.env.MAKEFUN_ENV_URL;
  if (!brief || !envUrl) return EMPTY_CATALOG;
  try {
    return await probeCatalog(envUrl, brief);
  } catch {
    return EMPTY_CATALOG;
  }
}

function jsonBody(res: any, status: number, value: any): void {
  const text = JSON.stringify(value);
  res.writeHead(status, { "content-type": "application/json", "content-length": Buffer.byteLength(text) });
  res.end(text);
}

function modelsAnswer(catalog: Catalog, req: any): any {
  const ids = freeModelIds(catalog);
  const auth = typeof req.headers.authorization === "string" ? req.headers.authorization : "";
  if (typeof req.headers["x-api-key"] === "string" && !auth.startsWith("Bearer ")) {
    return {
      data: ids.map((id) => ({ type: "model", id, display_name: id, created_at: "2026-01-01T00:00:00Z" })),
      has_more: false,
    };
  }
  return {
    object: "list",
    data: ids.map((id) => ({ id, object: "model", created: CREATED, owned_by: "free-model-aggregator" })),
  };
}

async function callProvider(
  envUrl: string,
  clientApi: string,
  body: any,
  dir: ProviderDir,
  model: string,
  stream: boolean,
): Promise<{ text: string; input: number; output: number; stop: string } | null> {
  const call = buildProviderRequest(envUrl, clientApi, body, dir, model, stream);
  let res: Response;
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 15000);
    try {
      res = await fetch(call.url, {
        method: "POST",
        headers: call.headers,
        body: JSON.stringify(call.body),
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timer);
    }
  } catch {
    return null;
  }
  if (res.status < 200 || res.status >= 300) {
    try {
      await res.text();
    } catch {
      // The body is drained only so the connection can be reused.
    }
    return null;
  }
  const text = await res.text();
  return parseProviderAnswer(dir.wire, stream, text);
}

async function main(): Promise<void> {
  const brief = loadBrief();
  const envUrl = process.env.MAKEFUN_ENV_URL ?? "";
  const catalog = await loadCatalog(brief);
  if (process.argv.includes("--list")) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    return;
  }
  const dirById = new Map<string, ProviderDir>();
  if (brief) for (const p of brief.providers) dirById.set(p.id, p);
  const plan: Map<string, PlanEntry> = brief ? buildPlan(brief) : new Map();
  const manifest = brief ? brief.manifest : [];
  let served = 0;
  // Provider calls actually made so far, successes and refusals alike.
  // Serve time re-ranks each request's candidates by this live count, so
  // a provider that just refused (or is nearly spent) is tried last and
  // a later request still finds a provider with budget left.
  const used = new Map<string, number>();

  const server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => {
      void (async () => {
        try {
          const url = new URL(req.url ?? "/", "http://entry");
          if (req.method === "GET" && url.pathname === "/v1/models") {
            jsonBody(res, 200, modelsAnswer(catalog, req));
            return;
          }
          const isChat = req.method === "POST" && url.pathname === "/v1/chat/completions";
          const isMessages = req.method === "POST" && url.pathname === "/v1/messages";
          if (!isChat && !isMessages) {
            jsonBody(res, 404, { error: { message: `no route ${req.method} ${url.pathname}`, type: "invalid_request_error", code: "not_found" } });
            return;
          }
          const clientApi = isChat ? "openai" : "anthropic";
          let body: any = null;
          try {
            body = JSON.parse(Buffer.concat(chunks).toString("utf8"));
          } catch {
            jsonBody(res, 400, { error: { message: "the request body is not one JSON object", type: "invalid_request_error", code: "invalid_request" } });
            return;
          }
          const stream = body.stream === true;
          const entry = manifest[served] ?? null;
          served += 1;
          let candidates = entry ? plan.get(entry.id)?.ordered ?? [] : [];
          if (candidates.length === 0) {
            // No plan entry (more requests than the manifest holds): fall
            // back to every free model, best rating first.
            const rows = (brief?.ratings ?? []).filter((r) => r.free);
            rows.sort((a, b) => b.rating_milli - a.rating_milli);
            candidates = rows.map((r) => ({ provider: r.provider, model: r.model, rating: r.rating_milli }));
          }
          let answered: { text: string; input: number; output: number; stop: string } | null = null;
          let answeredModel = "";
          // Try the candidates least-used-first, highest rating first on a
          // tie, re-ranking after every refusal: the provider a request
          // starts on is the one with the most budget left, not the one
          // the static plan named before any failure was known.
          const pool = candidates.map((c, i) => ({ ...c, order: i }));
          const ratingOrder = [...pool].sort((a, b) => b.rating - a.rating || a.order - b.order);
          const remaining = [...ratingOrder];
          while (remaining.length > 0) {
            remaining.sort(
              (a, b) =>
                (used.get(a.provider) ?? 0) - (used.get(b.provider) ?? 0) ||
                b.rating - a.rating ||
                a.order - b.order,
            );
            const cand = remaining.shift()!;
            const dir = dirById.get(cand.provider);
            if (!dir) continue;
            used.set(cand.provider, (used.get(cand.provider) ?? 0) + 1);
            const norm = await callProvider(envUrl, clientApi, body, dir, cand.model, stream);
            if (norm !== null) {
              answered = norm;
              answeredModel = cand.model;
              break;
            }
          }
          if (answered === null) {
            if (clientApi === "openai") {
              jsonBody(res, 502, { error: { message: "every provider refused the request", type: "api_error", code: "upstream_error" } });
            } else {
              jsonBody(res, 502, { type: "error", error: { type: "api_error", message: "every provider refused the request" } });
            }
            return;
          }
          const rendered = renderClient(clientApi, stream, answered, answeredModel);
          res.writeHead(200, { ...rendered.headers, "content-length": Buffer.byteLength(rendered.body) });
          res.end(rendered.body);
        } catch (err) {
          try {
            jsonBody(res, 500, { error: { message: String(err), type: "api_error", code: "internal" } });
          } catch {
            // The response is already gone; nothing left to do.
          }
        }
      })();
    });
  });

  const port = await new Promise<number>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => {
      const addr = server.address();
      if (addr && typeof addr === "object") resolve(addr.port);
      else reject(new Error("the server has no TCP address"));
    });
  });

  // Register, retrying briefly while the env comes up, then poll /work
  // every 100 ms and exit 0 as soon as the env says the workload is done.
  let registered = false;
  for (let attempt = 0; attempt < 50 && !registered; attempt++) {
    try {
      const res = await fetch(`${envUrl}/register`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
      });
      if (res.ok) registered = true;
      else await res.text();
    } catch {
      // The env is not reachable yet; try again below.
    }
    if (!registered) await sleep(200);
  }
  for (;;) {
    try {
      const res = await fetch(`${envUrl}/work`);
      if (res.ok) {
        const work: any = await res.json();
        if (work.done === true) {
          server.close();
          process.exit(0);
        }
      }
    } catch {
      // A missed poll is retried on the next tick.
    }
    await sleep(100);
  }
}

main().catch((err) => {
  process.stderr.write(`${String(err?.stack ?? err)}\n`);
  process.exit(1);
});
