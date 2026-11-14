# free-model-aggregator-01

The best free AI model aggregator: one server that serves the OpenAI
Chat Completions API and the Anthropic Messages API, in JSON and in
server-sent events, over ten free-tier model providers.

It reads the brief, probes each provider's catalog, and routes every
request to the strongest free model that meets the request, failing over
to the next best free model when a provider refuses. `node main.ts --list`
prints the catalog it registered, byte for byte.

## Run

```sh
node main.ts
```

No dependencies beyond the Node standard library.
