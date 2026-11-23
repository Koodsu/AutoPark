// Generate TypeScript types from the backend's exported JSON Schema.
// Usage: pnpm gen:types  (run `parkpilot export-schema` first; `make schema` does both)
import { readFile, writeFile } from "node:fs/promises";
import { compile } from "json-schema-to-typescript";

const schemaPath = new URL("../schema/protocol.schema.json", import.meta.url);
const outPath = new URL("../src/generated/protocol.ts", import.meta.url);

const schema = JSON.parse(await readFile(schemaPath, "utf8"));
const ts = await compile(schema, "ParkPilotProtocol", {
  bannerComment:
    "/* eslint-disable */\n/**\n * GENERATED from backend pydantic models by scripts/gen-types.mjs. Do not edit.\n * Regenerate with `make schema`.\n */",
  additionalProperties: false,
  unreachableDefinitions: true,
  style: { printWidth: 100, semi: true, singleQuote: false, trailingComma: "all" },
});
await writeFile(outPath, ts);
console.log(`wrote ${outPath.pathname}`);
