// Brief parsing, catalog probing and routing plan for the aggregator.
// The brief file holds three JSON documents, found by scanning for their
// schema markers instead of assuming any order: the provider directory,
// the sealed quality table and the workload manifest. The catalog itself
// is probed live from each provider, one provider at a time.
export interface ProviderDir {
  id: string;
  name: string;
  wire: string;
  auth: string;
  key_env: string;
  key: string;
  path: string;
}
export interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}
export interface WorkReq {
  id: string;
  api: string;
  stream: boolean;
  weight: number;
  requires: { tools?: boolean; vision?: boolean; json?: boolean; min_context?: number };
}
export interface Brief {
  providers: ProviderDir[];
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
  manifest: WorkReq[];
}
export interface CatModel {
  id: string;
  free: boolean;
  context_length: number | null;
}
export interface CatProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: string;
  models: CatModel[];
}
export interface Catalog {
  providers: CatProvider[];
}
export interface Candidate {
  provider: string;
  model: string;
  rating: number;
}
export interface PlanEntry {
  primary: Candidate;
  ordered: Candidate[];
}

function extractDocs(text: string): any[] {
  const docs: any[] = [];
  let i = 0;
  while (true) {
    const start = text.indexOf('{"schema"', i);
    if (start < 0) break;
    let depth = 0;
    let inStr = false;
    let esc = false;
    let j = start;
    for (; j < text.length; j++) {
      const ch = text[j];
      if (inStr) {
        if (esc) esc = false;
        else if (ch === "\\") esc = true;
        else if (ch === '"') inStr = false;
        continue;
      }
      if (ch === '"') inStr = true;
      else if (ch === "{") depth++;
      else if (ch === "}") {
        depth--;
        if (depth === 0) break;
      }
    }
    try {
      docs.push(JSON.parse(text.slice(start, j + 1)));
    } catch {
      // A malformed span is skipped; the other documents still count.
    }
    i = j + 1;
  }
  return docs;
}

export function parseBrief(text: string): Brief {
  const docs = extractDocs(text);
  let providers: ProviderDir[] = [];
  let ratings: RatingRow[] = [];
  let best: { id: string; rating_milli: number }[] = [];
  let manifest: WorkReq[] = [];
  for (const doc of docs) {
    if (doc.schema === "free-model-aggregator-01/providers/v1") {
      providers = (doc.providers ?? []).map((p: any) => ({
        id: p.id,
        name: p.name,
        wire: p.wire,
        auth: p.auth,
        key_env: p.key_env,
        key: p.key,
        path: p.path,
      }));
    } else if (doc.schema === "free-model-aggregator-01/ratings/v1") {
      ratings = (doc.ratings ?? []).map((r: any) => ({
        model: r.model,
        provider: r.provider,
        rating_milli: r.rating_milli,
        context_length: r.context_length ?? null,
        tools: r.tools === true,
        vision: r.vision === true,
        json: r.json === true,
        free: r.free === true,
      }));
      best = doc.best ?? [];
    } else if (doc.schema === "free-model-aggregator-01/workload/v1") {
      manifest = (doc.manifest ?? []).map((m: any) => ({
        id: m.id,
        api: m.api,
        stream: m.stream === true,
        weight: m.weight ?? 1,
        requires: m.requires ?? {},
      }));
    }
  }
  return { providers, ratings, best, manifest };
}

function authHeaders(dir: ProviderDir): Record<string, string> {
  if (dir.auth === "bearer") return { authorization: `Bearer ${dir.key}` };
  if (dir.auth === "x-api-key") return { "x-api-key": dir.key };
  if (dir.auth === "x-goog-api-key") return { "x-goog-api-key": dir.key };
  return {};
}

function listingModels(dir: ProviderDir, body: any): { id: string; free: boolean; context_length: number | null }[] {
  const out: { id: string; free: boolean; context_length: number | null }[] = [];
  if (dir.wire === "gemini") {
    const arr = Array.isArray(body?.models) ? body.models : [];
    for (const m of arr) {
      if (!m || typeof m !== "object") continue;
      const name = typeof m.name === "string" ? m.name : "";
      const id = name.startsWith("models/") ? name.slice("models/".length) : name;
      if (!id) continue;
      out.push({
        id,
        free: m.free === true,
        context_length: typeof m.context_length === "number" ? m.context_length : typeof m.inputTokenLimit === "number" ? m.inputTokenLimit : null,
      });
    }
    return out;
  }
  const arr = Array.isArray(body?.data) ? body.data : [];
  for (const m of arr) {
    if (!m || typeof m !== "object" || typeof m.id !== "string") continue;
    out.push({
      id: m.id,
      free: m.free === true,
      context_length: typeof m.context_length === "number" ? m.context_length : null,
    });
  }
  return out;
}

// Probe every provider's catalog route strictly one after another, then
// assemble the catalog document: providers in brief order, models sorted
// by id, the free flag taken from the sealed rating row when one exists
// and from the live listing otherwise. A provider that cannot be reached
// falls back to the rating rows so --list still works outside a trial.
export async function probeCatalog(envUrl: string, brief: Brief): Promise<Catalog> {
  const ratingByKey = new Map<string, RatingRow>();
  for (const row of brief.ratings) ratingByKey.set(`${row.provider}/${row.model}`, row);
  const catalog: Catalog = { providers: [] };
  for (const dir of brief.providers) {
    let listed: { id: string; free: boolean; context_length: number | null }[] = [];
    const route = dir.wire === "gemini" ? "/v1beta/models" : "/v1/models";
    try {
      const res = await fetch(`${envUrl}${dir.path}${route}`, { headers: authHeaders(dir) });
      if (res.ok) listed = listingModels(dir, await res.json());
    } catch {
      listed = [];
    }
    if (listed.length === 0) {
      for (const row of brief.ratings) {
        if (row.provider === dir.id) {
          listed.push({ id: row.model, free: row.free, context_length: row.context_length });
        }
      }
    }
    const models: CatModel[] = listed
      .map((m) => {
        const row = ratingByKey.get(`${dir.id}/${m.id}`);
        return {
          id: m.id,
          free: row ? row.free : m.free,
          context_length: m.context_length ?? (row ? row.context_length : null),
        };
      })
      .sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    catalog.providers.push({ id: dir.id, base_url: dir.path, key_env: dir.key_env, wire: dir.wire, models });
  }
  return catalog;
}

export function meets(row: RatingRow, req: WorkReq): boolean {
  if (req.requires.tools === true && !row.tools) return false;
  if (req.requires.vision === true && !row.vision) return false;
  if (req.requires.json === true && !row.json) return false;
  if (req.requires.min_context !== undefined) {
    if (row.context_length === null || row.context_length < req.requires.min_context) return false;
  }
  return true;
}

// Build the routing plan before serving anything. Per manifest request,
// in manifest order: candidates are the free models that meet the
// request and rate within 20 milli-points of the best legal rating for
// it. A JSON-mode request from an OpenAI client never goes to an
// Anthropic-wire provider, whose wire cannot carry response_format. The
// primary is the candidate whose provider has taken the fewest plan
// assignments so far, which spreads load across the providers' budgets;
// ties keep the higher rating first. Failover order is rating order.
// Serve time re-ranks these candidates again by the provider calls
// actually made so far (see main.ts): the plan names the candidate set,
// the live counts pick the order, so refusals move a provider down.
export function buildPlan(brief: Brief): Map<string, PlanEntry> {
  const wireById = new Map<string, string>();
  for (const p of brief.providers) wireById.set(p.id, p.wire);
  const plan = new Map<string, PlanEntry>();
  const assigned = new Map<string, number>();
  for (const req of brief.manifest) {
    const jsonFromOpenAI = req.api === "openai" && req.requires.json === true;
    const eligible = brief.ratings.filter((row) => {
      if (!row.free) return false;
      if (jsonFromOpenAI && wireById.get(row.provider) === "anthropic") return false;
      return meets(row, req);
    });
    if (eligible.length === 0) continue;
    const best = Math.max(...eligible.map((r) => r.rating_milli));
    const candidates: Candidate[] = eligible
      .filter((r) => r.rating_milli >= best - 20)
      .sort((a, b) => b.rating_milli - a.rating_milli)
      .map((r) => ({ provider: r.provider, model: r.model, rating: r.rating_milli }));
    let primary = candidates[0];
    for (const cand of candidates) {
      const candLoad = assigned.get(cand.provider) ?? 0;
      const bestLoad = assigned.get(primary.provider) ?? 0;
      if (candLoad < bestLoad) primary = cand;
    }
    assigned.set(primary.provider, (assigned.get(primary.provider) ?? 0) + 1);
    const ordered = [primary, ...candidates.filter((c) => c !== primary)];
    plan.set(req.id, { primary, ordered });
  }
  return plan;
}

export function freeModelIds(catalog: Catalog): string[] {
  const seen = new Set<string>();
  const ids: string[] = [];
  for (const p of catalog.providers) {
    for (const m of p.models) {
      if (m.free && !seen.has(m.id)) {
        seen.add(m.id);
        ids.push(m.id);
      }
    }
  }
  return ids;
}
