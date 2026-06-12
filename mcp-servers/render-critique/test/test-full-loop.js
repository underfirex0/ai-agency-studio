// Full loop test: render tectonic-hero.html → screenshot to GCS → compare_to_references → score.
// Run: node test/test-full-loop.js
import { chromium } from "playwright";
import { GoogleAuth } from "google-auth-library";
import { Storage } from "@google-cloud/storage";
import { randomUUID } from "crypto";
import { resolve } from "path";
import { pathToFileURL, fileURLToPath } from "url";

const __dirname  = fileURLToPath(new URL(".", import.meta.url));
const PROJECT    = "my-system-488711";
const BUCKET     = "my-system-488711-agency-assets";
const GEMINI_MODEL  = "gemini-2.5-flash";
const GEMINI_REGION = "us-central1";

const auth    = new GoogleAuth({ scopes: ["https://www.googleapis.com/auth/cloud-platform"] });
const storage = new Storage();

async function authHeader() {
  const client = await auth.getClient();
  const token  = await client.getAccessToken();
  return `Bearer ${token.token}`;
}

// ── Step 1: render_and_screenshot ─────────────────────────────────────────────

async function renderAndScreenshot(htmlPath) {
  console.log("\n▶  Step 1 — render_and_screenshot");
  const targetUrl = pathToFileURL(resolve(__dirname, htmlPath)).href;
  console.log(`   file   : ${targetUrl.slice(0, 80)}`);

  const t0      = Date.now();
  const browser = await chromium.launch({
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
  });

  let screenshotBuf;
  try {
    const page = await browser.newPage();
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto(targetUrl, { waitUntil: "load", timeout: 30_000 });
    // Extra wait so CSS animations that are triggered on load have time to render
    await page.waitForTimeout(3000);
    screenshotBuf = await page.screenshot({ fullPage: true, type: "png" });
  } finally {
    await browser.close();
  }

  const filename = `screenshots/${randomUUID()}.png`;
  const file     = storage.bucket(BUCKET).file(filename);
  await file.save(screenshotBuf, { contentType: "image/png", resumable: false });
  const gcsUri   = `gs://${BUCKET}/${filename}`;

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`   ✓  Screenshot captured in ${elapsed}s`);
  console.log(`   size   : ${(screenshotBuf.length / 1024).toFixed(0)} KB`);
  console.log(`   gcs_uri: ${gcsUri}`);
  return gcsUri;
}

// ── Step 2: compare_to_references ─────────────────────────────────────────────

async function compareToReferences(screenshotUri) {
  console.log("\n▶  Step 2 — compare_to_references");
  console.log(`   source : ${screenshotUri}`);

  // Download screenshot from GCS for inline embedding.
  const withoutScheme = screenshotUri.slice("gs://".length);
  const slashIdx      = withoutScheme.indexOf("/");
  const [bkt, obj]    = [withoutScheme.slice(0, slashIdx), withoutScheme.slice(slashIdx + 1)];
  const [buf] = await storage.bucket(bkt).file(obj).download();

  const briefExcerpt = `
Tectonic Studio — Creative Direction
Concept: The site is proof-of-process — every word, pause, and material detail demonstrates
that they think before they draw, so clients feel understood before they're asked to trust.

Narrative: The visitor arrives into near-silence: a single building photographed in flat
morning light, no headlines, no credentials — just the thing itself, asking to be looked
at slowly. As they move deeper, they encounter not a portfolio but a sequence of decisions.

Signature moment: On any project page, when the visitor dwells on a material photograph for
more than four seconds without scrolling, the image slowly desaturates while a verbatim
sentence from the client brief fades in at the bottom edge, 13px regular weight.

Brand personality: Precise in language, comfortable with long silences. Would never use an
exclamation mark, never describe themselves as 'collaborative' or 'design-led'. Would sooner
lose a commission than agree to a brief they believe is misconceived.
  `.trim();

  const designSystemExcerpt = `
Typography:
  heading: Cormorant Garamond 300/400/500
  body: Suisse Int'l 300/400/500 (Inter substituted in test)
  scale: display=5.5rem, h1=3.5rem, h2=2.25rem, h3=1.5rem, label=0.75rem, caption=0.6875rem

Colors:
  #1A1916  Formed Ash        → primary-text
  #F5F3EF  Drawn Light       → background
  #D6D2CB  Site Concrete     → surface
  #8C8880  Uncut Limestone   → secondary-text
  #2E3A35  Cold Seam         → navigation-background
  #5C7A6E  Oxidised Copper   → accent (dot nav active state)
  #A0522D  Fired Terracotta  → highlight (max 3 uses per page, feels like a stamp of refusal)

Key components:
  SilentNavBar: fixed 24px-height top bar, Cold Seam background, wordmark in 11px uppercase only
  ThresholdFrame: full-viewport entry, single architectural image, no copy
  DecisionTimeline: numbered sequence of material decisions with date, title, body, Resolved/Declined tag
  BriefFragmentQuote: Cold Seam 1px vertical rule, Cormorant Garamond italic quote, source caption
  ConversationOpener: open question in Cormorant italic h2, CTA underlined label text only
  SectionDotNav: 5px dots right edge, active filled Oxidised Copper
  `.trim();

  const focusAreas = [
    "colour token accuracy",
    "typography scale and weight",
    "SilentNavBar height and style",
    "ThresholdFrame composition",
    "DecisionTimeline structure",
    "Fired Terracotta usage restraint",
    "BriefFragmentQuote Cold Seam rule",
    "spacing scale adherence",
  ];

  const prompt = `You are a senior digital creative director doing a build-review pass.
You are looking at a rendered screenshot of a web page.
Your job is to identify every gap between what was built and what the brief specifies.
Be concrete — name the exact element, the exact token or spec it violates, and the exact fix.

BRIEF REFERENCE MATERIAL (use this as the ground truth for evaluation):

CREATIVE DIRECTION / BRIEF:
${briefExcerpt}

DESIGN SYSTEM TOKENS:
${designSystemExcerpt}

Pay particular attention to: ${focusAreas.join(", ")}.

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
      score: { type: "INTEGER" },
      headline: { type: "STRING" },
      implemented_correctly: { type: "ARRAY", items: { type: "STRING" } },
      gaps: {
        type: "ARRAY",
        items: {
          type: "OBJECT",
          properties: {
            element:    { type: "STRING" },
            brief_says: { type: "STRING" },
            render_has: { type: "STRING" },
            fix:        { type: "STRING" },
          },
          required: ["element", "brief_says", "render_has", "fix"],
        },
      },
      verdict: { type: "STRING" },
    },
    required: ["score", "headline", "implemented_correctly", "gaps", "verdict"],
  };

  const t0  = Date.now();
  const url = `https://${GEMINI_REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT}/locations/${GEMINI_REGION}/publishers/google/models/${GEMINI_MODEL}:generateContent`;

  const res = await fetch(url, {
    method: "POST",
    headers: { Authorization: await authHeader(), "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{
        role: "user",
        parts: [
          { inlineData: { mimeType: "image/png", data: buf.toString("base64") } },
          { text: prompt },
        ],
      }],
      generationConfig: { responseMimeType: "application/json", responseSchema, temperature: 0.2 },
    }),
  });

  const json = await res.json();
  if (!res.ok) {
    console.error("✗  Gemini error:", json?.error?.message ?? JSON.stringify(json));
    process.exit(1);
  }

  const raw     = json.candidates?.[0]?.content?.parts?.[0]?.text;
  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);

  if (!raw) {
    console.error("✗  No content:", JSON.stringify(json, null, 2));
    process.exit(1);
  }

  const critique = JSON.parse(raw);

  console.log(`   ✓  Gemini responded in ${elapsed}s\n`);
  console.log(`   Score    : ${critique.score}/100`);
  console.log(`   Headline : ${critique.headline}\n`);

  if (critique.implemented_correctly?.length) {
    console.log("   Implemented correctly:");
    critique.implemented_correctly.forEach(item => console.log(`     ✓ ${item}`));
  }

  if (critique.gaps?.length) {
    console.log("\n   Gaps vs brief:");
    critique.gaps.forEach(g => {
      console.log(`\n     [${g.element}]`);
      console.log(`       Brief says : ${g.brief_says}`);
      console.log(`       Render has : ${g.render_has}`);
      console.log(`       Fix        : ${g.fix}`);
    });
  }

  console.log(`\n   Verdict  : ${critique.verdict}`);
  return critique;
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  console.log("═══════════════════════════════════════════════════");
  console.log("  render-critique — full loop test");
  console.log("  Tectonic Studio hero page → screenshot → critique");
  console.log("═══════════════════════════════════════════════════");

  const t0 = Date.now();

  const gcsUri  = await renderAndScreenshot("tectonic-hero.html");
  const critique = await compareToReferences(gcsUri);

  const total = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`\n${"─".repeat(51)}`);
  console.log(`  Total time : ${total}s`);
  console.log(`  Screenshot : ${gcsUri}`);
  console.log(`  Score      : ${critique.score}/100`);
  console.log("─".repeat(51));
  console.log("\n✓  Full loop test passed");
}

main().catch(err => { console.error("\n✗  Unhandled error:", err.message); process.exit(1); });
