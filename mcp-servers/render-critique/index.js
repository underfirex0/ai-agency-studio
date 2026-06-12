import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { chromium } from "playwright";
import { GoogleAuth } from "google-auth-library";
import { Storage } from "@google-cloud/storage";
import { randomUUID } from "crypto";
import { readFileSync, existsSync } from "fs";
import { resolve, extname } from "path";
import { pathToFileURL } from "url";

// ── Config ────────────────────────────────────────────────────────────────────

const PROJECT      = "my-system-488711";
const BUCKET       = "my-system-488711-agency-assets";
const GEMINI_MODEL = "gemini-2.5-flash";
const GEMINI_REGION = "us-central1";

// ── GCP clients ───────────────────────────────────────────────────────────────

const auth    = new GoogleAuth({ scopes: ["https://www.googleapis.com/auth/cloud-platform"] });
const storage = new Storage();

async function authHeader() {
  const client = await auth.getClient();
  const token  = await client.getAccessToken();
  return `Bearer ${token.token}`;
}

async function uploadBuffer(buffer, destination, contentType) {
  const file = storage.bucket(BUCKET).file(destination);
  await file.save(buffer, { contentType, resumable: false });
  return `gs://${BUCKET}/${destination}`;
}

// Download a GCS object as a Buffer.
async function downloadFromGcs(gcsUri) {
  const withoutScheme = gcsUri.slice("gs://".length);
  const slashIdx      = withoutScheme.indexOf("/");
  const bucketName    = withoutScheme.slice(0, slashIdx);
  const objectName    = withoutScheme.slice(slashIdx + 1);
  const [buf] = await storage.bucket(bucketName).file(objectName).download();
  return buf;
}

// Fetch an HTTPS URL as a Buffer.
async function fetchBuffer(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${url}`);
  return { buffer: Buffer.from(await res.arrayBuffer()),
           mimeType: res.headers.get("content-type")?.split(";")[0] ?? "image/jpeg" };
}

// ── Vertex AI helper ──────────────────────────────────────────────────────────

async function vertexPost(region, path, body) {
  const url = `https://${region}-aiplatform.googleapis.com/v1${path}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { Authorization: await authHeader(), "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const json = await res.json();
  if (!res.ok) throw new Error(`Vertex AI ${res.status}: ${json?.error?.message ?? JSON.stringify(json)}`);
  return json;
}

// ── Tool: render_and_screenshot ───────────────────────────────────────────────

async function renderAndScreenshot({
  path_or_url,
  viewport_width  = 1440,
  viewport_height = 900,
  full_page       = true,
  wait_for        = "networkidle",
  delay_ms        = 0,
}) {
  // Resolve the target URL.
  let targetUrl;
  if (/^https?:\/\//.test(path_or_url)) {
    targetUrl = path_or_url;
  } else {
    const absPath = resolve(path_or_url);
    if (!existsSync(absPath)) throw new Error(`File not found: ${absPath}`);
    targetUrl = pathToFileURL(absPath).href;
  }

  const browser = await chromium.launch({
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--disable-dev-shm-usage",
      "--disable-gpu",
    ],
  });

  let screenshotBuf;
  try {
    const page = await browser.newPage();
    await page.setViewportSize({ width: viewport_width, height: viewport_height });

    // file:// pages fire "load"; http pages use waitUntil.
    const waitUntil = targetUrl.startsWith("file://") ? "load" : wait_for;
    await page.goto(targetUrl, { waitUntil, timeout: 30_000 });

    if (delay_ms > 0) await page.waitForTimeout(delay_ms);

    screenshotBuf = await page.screenshot({ fullPage: full_page, type: "png" });
  } finally {
    await browser.close();
  }

  const filename = `screenshots/${randomUUID()}.png`;
  const gcsUri   = await uploadBuffer(screenshotBuf, filename, "image/png");

  return {
    gcs_uri: gcsUri,
    viewport: { width: viewport_width, height: viewport_height },
    full_page,
    source: targetUrl,
  };
}

// ── Tool: compare_to_references ───────────────────────────────────────────────

async function compareToReferences({
  screenshot_uri,
  brief_excerpt,
  design_system_excerpt,
  motion_spec_excerpt,
  focus_areas,
}) {
  // Load the screenshot into base64.
  let imageBase64, imageMimeType;
  if (screenshot_uri.startsWith("gs://")) {
    const buf    = await downloadFromGcs(screenshot_uri);
    imageBase64  = buf.toString("base64");
    imageMimeType = "image/png";
  } else {
    const { buffer, mimeType } = await fetchBuffer(screenshot_uri);
    imageBase64   = buffer.toString("base64");
    imageMimeType = mimeType;
  }

  // Build the critique prompt from whichever brief sections were supplied.
  const briefSections = [];
  if (brief_excerpt)        briefSections.push(`CREATIVE DIRECTION / BRIEF:\n${brief_excerpt}`);
  if (design_system_excerpt) briefSections.push(`DESIGN SYSTEM TOKENS:\n${design_system_excerpt}`);
  if (motion_spec_excerpt)  briefSections.push(`MOTION SPEC:\n${motion_spec_excerpt}`);

  const focusClause = focus_areas?.length
    ? `\nPay particular attention to: ${focus_areas.join(", ")}.`
    : "";

  const systemContext = briefSections.length
    ? `\n\nBRIEF REFERENCE MATERIAL (use this as the ground truth for evaluation):\n\n${briefSections.join("\n\n")}`
    : "";

  const prompt = `You are a senior digital creative director doing a build-review pass.
You are looking at a rendered screenshot of a web page.
Your job is to identify every gap between what was built and what the brief specifies.
Be concrete — name the exact element, the exact token or spec it violates, and the exact fix.${systemContext}${focusClause}

Score 0–100 where:
  90–100 = implementation matches brief intent exactly
  75–89  = minor deviations, one or two clear fixes
  60–74  = meaningful gaps that dilute the brief's intent
  40–59  = significant departures, rework needed
  0–39   = does not reflect the brief

Return a JSON object matching the schema.`;

  const responseSchema = {
    type: "OBJECT",
    properties: {
      score: {
        type: "INTEGER",
        description: "0–100 implementation fidelity to brief.",
      },
      headline: {
        type: "STRING",
        description: "One sentence naming the most critical gap or the strongest success.",
      },
      implemented_correctly: {
        type: "ARRAY",
        items: { type: "STRING" },
        description: "Brief elements correctly present in the render.",
      },
      gaps: {
        type: "ARRAY",
        description: "Brief requirements missing or violated — ordered by impact.",
        items: {
          type: "OBJECT",
          properties: {
            element:    { type: "STRING", description: "Which component, token, or spec." },
            brief_says: { type: "STRING", description: "What the brief specifies." },
            render_has: { type: "STRING", description: "What was actually rendered." },
            fix:        { type: "STRING", description: "Exact change needed to close the gap." },
          },
          required: ["element", "brief_says", "render_has", "fix"],
        },
      },
      verdict: {
        type: "STRING",
        description: "2–3 sentences on overall implementation quality and priority next step.",
      },
    },
    required: ["score", "headline", "implemented_correctly", "gaps", "verdict"],
  };

  const result = await vertexPost(
    GEMINI_REGION,
    `/projects/${PROJECT}/locations/${GEMINI_REGION}/publishers/google/models/${GEMINI_MODEL}:generateContent`,
    {
      contents: [{
        role: "user",
        parts: [
          { inlineData: { mimeType: imageMimeType, data: imageBase64 } },
          { text: prompt },
        ],
      }],
      generationConfig: {
        responseMimeType: "application/json",
        responseSchema,
        temperature: 0.2,
      },
    }
  );

  const raw = result.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!raw) throw new Error("Gemini returned no content");

  return JSON.parse(raw);
}

// ── MCP tool definitions ──────────────────────────────────────────────────────

const TOOLS = [
  {
    name: "render_and_screenshot",
    description:
      "Render a local HTML/CSS/JS file or a running dev server URL headlessly via Playwright Chromium, " +
      "take a full-page screenshot, and save it to gs://my-system-488711-agency-assets/screenshots/. " +
      "Returns the GCS URI of the PNG.",
    inputSchema: {
      type: "object",
      required: ["path_or_url"],
      properties: {
        path_or_url: {
          type: "string",
          description: "Absolute or relative path to a local HTML file, or an http(s):// URL of a running dev server.",
        },
        viewport_width: {
          type: "number",
          default: 1440,
          description: "Viewport width in pixels (default 1440).",
        },
        viewport_height: {
          type: "number",
          default: 900,
          description: "Viewport height in pixels (default 900).",
        },
        full_page: {
          type: "boolean",
          default: true,
          description: "Capture the full scrollable page, not just the visible viewport.",
        },
        wait_for: {
          type: "string",
          enum: ["load", "domcontentloaded", "networkidle"],
          default: "networkidle",
          description: "Playwright waitUntil condition before screenshotting.",
        },
        delay_ms: {
          type: "number",
          default: 0,
          description: "Additional wait in milliseconds after the page is ready (useful for CSS animations to settle).",
        },
      },
    },
  },
  {
    name: "compare_to_references",
    description:
      "Load a screenshot (GCS URI or HTTPS URL) and compare it against the project brief using " +
      "Gemini 2.5 Flash vision. Returns a structured score and specific gaps versus the brief's intent.",
    inputSchema: {
      type: "object",
      required: ["screenshot_uri"],
      properties: {
        screenshot_uri: {
          type: "string",
          description: "GCS URI (gs://…) or HTTPS URL of the screenshot to evaluate.",
        },
        brief_excerpt: {
          type: "string",
          description: "Creative direction and concept text from the project brief.",
        },
        design_system_excerpt: {
          type: "string",
          description: "Relevant design system tokens (colours, typography, spacing) from the brief.",
        },
        motion_spec_excerpt: {
          type: "string",
          description: "Relevant motion/animation spec from the brief.",
        },
        focus_areas: {
          type: "array",
          items: { type: "string" },
          description: "Specific aspects to foreground, e.g. ['colour tokens', 'type scale', 'spacing', 'nav pattern'].",
        },
      },
    },
  },
];

// ── MCP server ────────────────────────────────────────────────────────────────

const server = new Server(
  { name: "render-critique", version: "0.1.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  try {
    let result;
    switch (name) {
      case "render_and_screenshot": result = await renderAndScreenshot(args); break;
      case "compare_to_references": result = await compareToReferences(args); break;
      default:
        return { content: [{ type: "text", text: `Unknown tool: ${name}` }], isError: true };
    }
    return { content: [{ type: "text", text: JSON.stringify(result, null, 2) }] };
  } catch (err) {
    return { content: [{ type: "text", text: `Error: ${err.message}` }], isError: true };
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
