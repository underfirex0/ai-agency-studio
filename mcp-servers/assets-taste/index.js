import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { GoogleAuth } from "google-auth-library";
import { Storage } from "@google-cloud/storage";
import textToSpeech from "@google-cloud/text-to-speech";
import { randomUUID } from "crypto";

// ── Config ────────────────────────────────────────────────────────────────────

const PROJECT   = "my-system-488711";
const BUCKET    = "my-system-488711-agency-assets";
// Imagen 3 is available in europe-west1; Gemini 2.5 Flash and Veo 2 only in us-central1.
const IMAGEN_REGION = "europe-west1";
const GEMINI_REGION = "us-central1";
const VEO_REGION    = "us-central1";

const IMAGEN_MODEL = "imagen-3.0-generate-001";
const GEMINI_MODEL = "gemini-2.5-flash";
const VEO_MODEL    = "veo-2.0-generate-001";

// ── GCP clients ───────────────────────────────────────────────────────────────

const auth    = new GoogleAuth({ scopes: ["https://www.googleapis.com/auth/cloud-platform"] });
const storage = new Storage();
const ttsClient = new textToSpeech.TextToSpeechClient();

async function authHeader() {
  const client = await auth.getClient();
  const token  = await client.getAccessToken();
  return `Bearer ${token.token}`;
}

// ── GCS helpers ───────────────────────────────────────────────────────────────

async function uploadBuffer(buffer, destination, contentType) {
  const file = storage.bucket(BUCKET).file(destination);
  await file.save(buffer, { contentType, resumable: false });
  return `gs://${BUCKET}/${destination}`;
}

// Fetch an HTTPS image URL and return { base64, mimeType }.
async function fetchImageAsBase64(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch image ${url}: ${res.status}`);
  const buf      = Buffer.from(await res.arrayBuffer());
  const mimeType = res.headers.get("content-type")?.split(";")[0] ?? "image/jpeg";
  return { base64: buf.toString("base64"), mimeType };
}

// ── Vertex AI REST helper ─────────────────────────────────────────────────────

async function vertexPost(region, path, body) {
  const url = `https://${region}-aiplatform.googleapis.com/v1${path}`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      Authorization: await authHeader(),
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });
  const json = await res.json();
  if (!res.ok) {
    const msg = json?.error?.message ?? JSON.stringify(json);
    throw new Error(`Vertex AI ${res.status}: ${msg}`);
  }
  return json;
}

async function vertexGet(region, path) {
  const url = `https://${region}-aiplatform.googleapis.com/v1${path}`;
  const res = await fetch(url, {
    headers: { Authorization: await authHeader() },
  });
  const json = await res.json();
  if (!res.ok) {
    const msg = json?.error?.message ?? JSON.stringify(json);
    throw new Error(`Vertex AI GET ${res.status}: ${msg}`);
  }
  return json;
}

// ── Tool implementations ──────────────────────────────────────────────────────

async function generateImage({ prompt, style_reference, aspect_ratio = "1:1" }) {
  // Prepend style reference to prompt if provided, the same way the brief uses named photographers.
  const fullPrompt = style_reference
    ? `${prompt} — style reference: ${style_reference}`
    : prompt;

  const storagePrefix = `gs://${BUCKET}/images/`;
  const result = await vertexPost(
    IMAGEN_REGION,
    `/projects/${PROJECT}/locations/${IMAGEN_REGION}/publishers/google/models/${IMAGEN_MODEL}:predict`,
    {
      instances: [{ prompt: fullPrompt }],
      parameters: {
        sampleCount: 1,
        aspectRatio: aspect_ratio,
        storageUri: storagePrefix,
        addWatermark: false,
        safetySetting: "block_some",
        personGeneration: "allow_adult",
      },
    }
  );

  const prediction = result.predictions?.[0];
  if (!prediction) throw new Error("Imagen returned no predictions");

  // Prefer the GCS URI written directly by Imagen; fall back to uploading the base64 bytes.
  let gcsUri = prediction.gcsUri;
  if (!gcsUri && prediction.bytesBase64Encoded) {
    const ext       = prediction.mimeType === "image/jpeg" ? "jpg" : "png";
    const filename  = `images/${randomUUID()}.${ext}`;
    const buffer    = Buffer.from(prediction.bytesBase64Encoded, "base64");
    gcsUri = await uploadBuffer(buffer, filename, prediction.mimeType ?? "image/png");
  }
  if (!gcsUri) throw new Error("Could not obtain GCS URI from Imagen response");

  return { gcs_uri: gcsUri, aspect_ratio, prompt: fullPrompt };
}


async function critiqueDesign({ images, brief_excerpt, focus_areas }) {
  // Build the multimodal parts list — mix of image data + text.
  const parts = [];

  for (const imageSource of images) {
    if (imageSource.startsWith("gs://")) {
      parts.push({
        fileData: {
          mimeType: guessMimeFromPath(imageSource),
          fileUri: imageSource,
        },
      });
    } else {
      // HTTPS URL — fetch and inline as base64 (Vertex AI Gemini doesn't accept arbitrary HTTP URLs).
      const { base64, mimeType } = await fetchImageAsBase64(imageSource);
      parts.push({ inlineData: { mimeType, data: base64 } });
    }
  }

  const focusClause = focus_areas?.length
    ? `Pay particular attention to: ${focus_areas.join(", ")}.`
    : "";

  const briefClause = brief_excerpt
    ? `\n\nPROJECT BRIEF EXCERPT (evaluate the design against this):\n${brief_excerpt}`
    : "";

  parts.push({
    text: `You are a senior digital creative director reviewing a design asset.
Evaluate the image(s) above with precision and candour — no diplomatic hedging.${briefClause}

${focusClause}

Score 0–100 where:
  90–100 = ships immediately, award-worthy
  75–89  = strong, one meaningful revision needed
  60–74  = competent foundation, two or three specific issues
  40–59  = significant rework needed
  0–39   = fundamental problems, reconsider direction

Return a JSON object matching the schema provided.`,
  });

  const responseSchema = {
    type: "OBJECT",
    properties: {
      score: { type: "INTEGER", description: "0–100 design quality score." },
      headline: { type: "STRING", description: "One sentence naming the single strongest quality or most critical weakness." },
      strengths: { type: "ARRAY", items: { type: "STRING" }, description: "2–4 specific things working well." },
      improvements: {
        type: "ARRAY",
        description: "Ordered by impact — most important first.",
        items: {
          type: "OBJECT",
          properties: {
            area:  { type: "STRING", description: "Design area (e.g. 'typography', 'colour contrast', 'hierarchy')." },
            issue: { type: "STRING", description: "Specific problem observed." },
            fix:   { type: "STRING", description: "Concrete actionable fix." },
          },
          required: ["area", "issue", "fix"],
        },
      },
      brief_alignment: { type: "STRING", description: "How well the design serves the stated brief. Omit if no brief was provided." },
      verdict: { type: "STRING", description: "2–3 sentence directional recommendation." },
    },
    required: ["score", "headline", "strengths", "improvements", "verdict"],
  };

  const result = await vertexPost(
    GEMINI_REGION,
    `/projects/${PROJECT}/locations/${GEMINI_REGION}/publishers/google/models/${GEMINI_MODEL}:generateContent`,
    {
      contents: [{ role: "user", parts }],
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


async function generateVideo({ prompt, duration_seconds = 8, aspect_ratio = "16:9" }) {
  // Veo 2 is a long-running operation. Kick it off, poll until done.
  const outputPrefix = `gs://${BUCKET}/videos/`;

  const op = await vertexPost(
    VEO_REGION,
    `/projects/${PROJECT}/locations/${VEO_REGION}/publishers/google/models/${VEO_MODEL}:predictLongRunning`,
    {
      instances: [{ prompt }],
      parameters: {
        storageUri: outputPrefix,
        durationSeconds: Math.min(Math.max(duration_seconds, 5), 8), // Veo 2: 5–8s
        aspectRatio: aspect_ratio,
        sampleCount: 1,
      },
    }
  );

  const operationName = op.name;
  if (!operationName) throw new Error("Veo did not return an operation name");

  // Poll — Veo video generation typically takes 2–4 minutes.
  const pollPath = `/${operationName}`;
  const deadline = Date.now() + 8 * 60 * 1000; // 8 min timeout

  while (Date.now() < deadline) {
    await sleep(15_000);
    const status = await vertexGet(VEO_REGION, pollPath);

    if (status.done) {
      if (status.error) {
        throw new Error(`Veo operation failed: ${status.error.message}`);
      }
      // Response shape: response.videos[0].gcsUri  OR  generatedSamples[0].video.uri
      const video =
        status.response?.videos?.[0] ??
        status.response?.generatedSamples?.[0]?.video;
      const gcsUri = video?.gcsUri ?? video?.uri;
      if (!gcsUri) {
        throw new Error(`Veo completed but no video URI found: ${JSON.stringify(status.response)}`);
      }
      return { gcs_uri: gcsUri, operation_name: operationName, duration_seconds, aspect_ratio };
    }
  }

  throw new Error(`Veo operation timed out after 8 minutes: ${operationName}`);
}


async function generateSpeech({ text, voice_description = "", speaking_rate = 1.0 }) {
  const voice   = selectVoice(voice_description);
  const [resp]  = await ttsClient.synthesizeSpeech({
    input:       { text },
    voice:       { languageCode: voice.languageCode, name: voice.name },
    audioConfig: { audioEncoding: "MP3", speakingRate: speaking_rate },
  });

  const filename = `audio/${randomUUID()}.mp3`;
  const gcsUri   = await uploadBuffer(resp.audioContent, filename, "audio/mpeg");

  return { gcs_uri: gcsUri, voice: voice.name, language_code: voice.languageCode };
}


// ── Voice selection ───────────────────────────────────────────────────────────
// Cloud TTS Studio voices (high quality, lifelike). Keyword-matched from voice_description.

const VOICES = [
  { name: "en-US-Studio-O", languageCode: "en-US", tags: ["female", "american", "us", "warm", "studio"] },
  { name: "en-US-Studio-Q", languageCode: "en-US", tags: ["male", "american", "us", "studio"] },
  { name: "en-US-Neural2-C", languageCode: "en-US", tags: ["female", "american", "neural"] },
  { name: "en-US-Neural2-D", languageCode: "en-US", tags: ["male", "american", "neural", "deep"] },
  { name: "en-GB-Studio-B", languageCode: "en-GB", tags: ["male", "british", "uk", "studio"] },
  { name: "en-GB-Studio-C", languageCode: "en-GB", tags: ["female", "british", "uk", "studio"] },
  { name: "en-AU-Neural2-A", languageCode: "en-AU", tags: ["female", "australian"] },
  { name: "en-AU-Neural2-B", languageCode: "en-AU", tags: ["male", "australian"] },
];

function selectVoice(description) {
  const lower = description.toLowerCase();
  let best = VOICES[0]; // default: en-US-Studio-O
  let bestScore = -1;
  for (const voice of VOICES) {
    const score = voice.tags.filter(t => lower.includes(t)).length;
    if (score > bestScore) { bestScore = score; best = voice; }
  }
  return best;
}


// ── Misc helpers ──────────────────────────────────────────────────────────────

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function guessMimeFromPath(path) {
  if (path.endsWith(".png"))  return "image/png";
  if (path.endsWith(".webp")) return "image/webp";
  if (path.endsWith(".gif"))  return "image/gif";
  return "image/jpeg";
}


// ── MCP tool definitions ──────────────────────────────────────────────────────

const TOOLS = [
  {
    name: "generate_image",
    description:
      "Generate an image using Vertex AI Imagen 3 (europe-west1). " +
      "Returns the GCS URI of the saved image.",
    inputSchema: {
      type: "object",
      required: ["prompt"],
      properties: {
        prompt: {
          type: "string",
          description: "Full text-to-image prompt. Include camera, lighting, and subject details.",
        },
        style_reference: {
          type: "string",
          description: "Named photographer or visual style to append to the prompt (e.g. 'Charlie Schuck — deep shadow, single key light').",
        },
        aspect_ratio: {
          type: "string",
          enum: ["1:1", "16:9", "9:16", "4:3", "3:4"],
          default: "1:1",
          description: "Output aspect ratio.",
        },
      },
    },
  },
  {
    name: "generate_video",
    description:
      "Generate a short video using Vertex AI Veo 2 (us-central1). " +
      "Submits a long-running operation and polls until complete (~2–4 min). " +
      "Returns the GCS URI of the saved MP4.",
    inputSchema: {
      type: "object",
      required: ["prompt"],
      properties: {
        prompt: {
          type: "string",
          description: "Video generation prompt describing the scene, movement, and atmosphere.",
        },
        duration_seconds: {
          type: "number",
          minimum: 5,
          maximum: 8,
          default: 8,
          description: "Video duration in seconds (Veo 2 supports 5–8 seconds).",
        },
        aspect_ratio: {
          type: "string",
          enum: ["16:9", "9:16", "1:1"],
          default: "16:9",
        },
      },
    },
  },
  {
    name: "generate_speech",
    description:
      "Synthesise speech from text using Google Cloud Text-to-Speech Studio voices. " +
      "Returns the GCS URI of the saved MP3.",
    inputSchema: {
      type: "object",
      required: ["text"],
      properties: {
        text: {
          type: "string",
          description: "Text to synthesise. Plain text or SSML.",
        },
        voice_description: {
          type: "string",
          description: "Natural-language voice description: gender, accent, and tone keywords (e.g. 'warm female British', 'deep male American').",
        },
        speaking_rate: {
          type: "number",
          minimum: 0.25,
          maximum: 4.0,
          default: 1.0,
          description: "Speaking rate multiplier (1.0 = normal speed).",
        },
      },
    },
  },
  {
    name: "critique_design",
    description:
      "Critique one or more design images using Gemini 2.5 Flash vision (us-central1). " +
      "Accepts GCS URIs (gs://) or HTTPS URLs. Returns a structured score and improvement notes " +
      "calibrated against the project brief.",
    inputSchema: {
      type: "object",
      required: ["images"],
      properties: {
        images: {
          type: "array",
          items: { type: "string" },
          minItems: 1,
          description: "List of image sources: GCS URIs (gs://…) or HTTPS URLs.",
        },
        brief_excerpt: {
          type: "string",
          description: "Relevant text from the project brief to evaluate the design against.",
        },
        focus_areas: {
          type: "array",
          items: { type: "string" },
          description: "Design dimensions to foreground, e.g. ['typography', 'colour contrast', 'hierarchy', 'brand_alignment'].",
        },
      },
    },
  },
];


// ── MCP server ────────────────────────────────────────────────────────────────

const server = new Server(
  { name: "assets-taste", version: "0.1.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    let result;
    switch (name) {
      case "generate_image":   result = await generateImage(args);   break;
      case "generate_video":   result = await generateVideo(args);   break;
      case "generate_speech":  result = await generateSpeech(args);  break;
      case "critique_design":  result = await critiqueDesign(args);  break;
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
