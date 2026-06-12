// End-to-end test for generate_image.
// Run: node test/test-imagen.js
import { GoogleAuth } from "google-auth-library";
import { Storage } from "@google-cloud/storage";
import { randomUUID } from "crypto";

const PROJECT      = "my-system-488711";
const BUCKET       = "my-system-488711-agency-assets";
const IMAGEN_MODEL = "imagen-3.0-generate-001";
const REGION       = "europe-west1";

const auth    = new GoogleAuth({ scopes: ["https://www.googleapis.com/auth/cloud-platform"] });
const storage = new Storage();

async function authHeader() {
  const client = await auth.getClient();
  const token  = await client.getAccessToken();
  return `Bearer ${token.token}`;
}

async function run() {
  console.log("▶  generate_image — end-to-end test");
  console.log(`   model  : ${IMAGEN_MODEL}`);
  console.log(`   region : ${REGION}`);

  const prompt = [
    "Wide establishing shot of a raw concrete staircase in a Nordic building,",
    "raking light from a narrow skylight above, deep shadow in the lower treads,",
    "shot on Hasselblad medium format, slight grain, desaturated grey tones,",
    "architectural photography style"
  ].join(" ");

  const styleRef = "Hélène Binet architectural photography — shadow-heavy, single natural light source";

  console.log(`\n   prompt : ${prompt.slice(0, 80)}…`);
  console.log(`   style  : ${styleRef}`);

  const t0     = Date.now();
  const reqBody = {
    instances: [{ prompt: `${prompt} — style reference: ${styleRef}` }],
    parameters: {
      sampleCount: 1,
      aspectRatio: "16:9",
      storageUri: `gs://${BUCKET}/images/`,
      addWatermark: false,
      safetySetting: "block_some",
      personGeneration: "allow_adult",
    },
  };

  const url = `https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT}/locations/${REGION}/publishers/google/models/${IMAGEN_MODEL}:predict`;
  const res = await fetch(url, {
    method: "POST",
    headers: { Authorization: await authHeader(), "Content-Type": "application/json" },
    body: JSON.stringify(reqBody),
  });

  const json = await res.json();
  if (!res.ok) {
    console.error("✗  API error:", json?.error?.message ?? JSON.stringify(json));
    process.exit(1);
  }

  const prediction = json.predictions?.[0];
  const elapsed    = ((Date.now() - t0) / 1000).toFixed(1);

  let gcsUri = prediction?.gcsUri;

  // If Imagen wrote the storageUri directly the gcsUri will be set; otherwise upload bytes.
  if (!gcsUri && prediction?.bytesBase64Encoded) {
    const ext      = prediction.mimeType === "image/jpeg" ? "jpg" : "png";
    const filename = `images/${randomUUID()}.${ext}`;
    const buf      = Buffer.from(prediction.bytesBase64Encoded, "base64");
    const file     = storage.bucket(BUCKET).file(filename);
    await file.save(buf, { contentType: prediction.mimeType ?? "image/png", resumable: false });
    gcsUri = `gs://${BUCKET}/${filename}`;
    console.log(`\n   ✓  Uploaded ${(buf.length / 1024).toFixed(0)} KB to GCS in ${elapsed}s`);
  } else {
    console.log(`\n   ✓  Imagen wrote directly to GCS in ${elapsed}s`);
  }

  if (!gcsUri) {
    console.error("✗  No image URI in response:", JSON.stringify(json, null, 2));
    process.exit(1);
  }

  console.log(`   gcs_uri: ${gcsUri}`);
  console.log("\n✓  generate_image test passed");
}

run().catch(err => { console.error("✗  Unhandled error:", err.message); process.exit(1); });
