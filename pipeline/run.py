#!/usr/bin/env python3
"""
Brief generation pipeline.

Usage:
    python pipeline/run.py brief/inputs/test-project.json
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import jsonschema

REPO_ROOT = Path(__file__).parent.parent
SCHEMA_PATH = REPO_ROOT / "brief" / "schema.json"
OUTPUT_DIR = REPO_ROOT / "brief" / "output"

MODEL = "claude-sonnet-4-6"

STAGES = [
    "creative_direction",
    "ux_structure",
    "design_system",
    "motion_spec",
    "spatial_3d_spec",
    "asset_manifest",
]

# ── Per-stage prompt instructions ─────────────────────────────────────────────
# Each instruction block tells Claude what quality bar to hit for that section.
# References the coffee-roastery sample to anchor specificity expectations.

STAGE_INSTRUCTIONS = {

    "creative_direction": """\
You are a Creative Director writing the creative_direction section of a web project brief.

QUALITY BAR — match this level of specificity:
  concept:          "Every bag of coffee is a compressed journey from a specific hillside to your cup —
                     the site makes that altitude, terroir, and human chain viscerally felt."
  signature_moment: "On the origin section, hovering a farm pin on the map triggers a 3-second ambient
                     field recording from that specific farm (birds, wind, distant machinery) while the
                     elevation number counts up in large type — making the geography audible."
  brand_personality: Named a specific archetype with vocabulary they USE and words they would NEVER say.
                     ("Would never say 'artisanal' or 'passionate'. Comfortable with silence.")

RULES:
- concept must be one sentence that works as an elevator pitch AND a design filter.
- narrative must walk from landing → middle → close, naming the emotional beats, not the sections.
- signature_moment must be ONE interaction that is concrete enough to spec in a ticket:
  what triggers it, what exactly happens visually/sonically, how long it lasts.
- brand_personality must name what the brand would never say or do — negative constraints are as
  important as positive ones.
- Do NOT use the words: artisanal, passionate, journey, curated, innovative, premium, luxury,
  cutting-edge, bespoke, or world-class.
- Draw directly on the project's meta.target_audience and meta.tone_keywords.
""",

    "ux_structure": """\
You are a UX Strategist writing the ux_structure section of a web project brief.

QUALITY BAR — match this level of specificity:
  sitemap:          6–8 sections with slug IDs that will be referenced in motion_spec later.
                    Each purpose is a single job ("Build credibility through farm-level provenance").
                    Include children where a section has a slide-in panel or sub-state.
  conversion_flow:  Traces the path step by step: "Hero creates atmosphere → X earns trust →
                     Y adds accountability → Z lets them choose → CTA captures recurring revenue."
  navigation_pattern: Names the exact behaviour at the breakpoint ("≤768px a 24px hamburger opens
                       a full-screen overlay nav").

RULES:
- Section IDs must be lowercase kebab-case slugs with no spaces.
- Every section needs a purpose that is ONE job, not a description of its contents.
- The conversion_flow must mention the number of scroll steps to the primary CTA.
- navigation_pattern must specify desktop AND mobile behaviour with a pixel breakpoint.
- Design the IA to serve the creative_direction concept — if the concept is about revealing
  something hidden, the IA should have moments of reveal, not just a list of features.
- For industries where trust is the primary barrier, front-load credibility sections before
  the commercial sections.
""",

    "design_system": """\
You are a Visual Designer writing the design_system section of a web project brief.

QUALITY BAR — match this level of specificity:
  typography:    Named real fonts with correct weights. Pair a distinctive display/heading face
                 with a workhorse body face. Weights like "300,400" or "400,500,700" — not just "regular".
  color_palette: 5–7 colours. Every colour has a NAMED swatch (evocative, industry-specific names,
                 not "Dark Blue"), a precise hex, and a CSS-role. Anchored to the brand world:
                 coffee uses "Wet Earth #1C1512", "Blonde Roast #C8A96E", "Dried Cherry #8B2E2E".
  components:    8–12 PascalCase component names that are specific to THIS project, not generic.
                 "FarmPin", "ElevationCounter", "AmbientAudioPlayer" — not "Button", "Card", "Modal".
                 Each description specifies states, variants, or visual treatments.

RULES:
- Fonts: use real names from Google Fonts, Adobe Fonts, or well-known custom type foundries.
  Source must be "google", "adobe", "custom", or "system".
- Color names must be evocative of the brand world, not generic ("Polished Concrete" not "Medium Grey").
- Every color_palette entry must have a distinct role — no two entries share the same role.
- spacing_scale must cover xs through at least 3xl, all values in rem.
- Component names must map directly to the sitemap sections — a component per meaningful UI state.
- Describe component states: hover, active, disabled, or responsive variants where relevant.
- Highlight one colour that is UNDERUSED as a design challenge to solve (like "Dried Cherry" in coffee).
""",

    "motion_spec": """\
You are a Motion Designer writing the motion_spec section of a web project brief.

QUALITY BAR — match this level of specificity:
  global_principles: Discipline constraints ("Nothing moves unless scroll or pointer caused it"),
                     easing philosophy ("always decelerating"), property constraints ("opacity and
                     transform only — no layout-triggering animations"), accessibility rule
                     ("reduced-motion query disables all transitions, substituting instant cuts").
  per_section:       One entry per sitemap ID (including children). Each entry specifies:
                     - scroll_behavior: what scroll does (parallax rate, pin duration in vh, threshold)
                     - transitions: enter/exit with exact CSS properties (translateY, clip-path, opacity)
                       and direction
                     - timing: exact duration in ms and cubic-bezier values (not named easings)

RULES:
- per_section must cover EVERY section_id from the ux_structure.sitemap (including children).
- Global principles must be at least 4 items — include at least one about easing, one about
  reduced-motion accessibility, one about what properties are allowed.
- Scroll-pinned sections must specify pin duration in viewport-height units (e.g. "pins for 150vh").
- Spring/overshoot effects need a cubic-bezier with a value > 1 in the third parameter
  (e.g. "cubic-bezier(0.34, 1.56, 0.64, 1)").
- Stagger animations must give the per-item delay in ms.
- Timing strings must be "Xms cubic-bezier(a, b, c, d)" — no named easings like "ease-in-out".
- The signature_moment from creative_direction must have its own motion entry or be embedded
  in the relevant section's transitions.
""",

    "spatial_3d_spec": """\
You are a 3D/WebGL specialist writing the spatial_3d_spec section of a web project brief.

DECISION RULE — use THREE.JS (used: true) only if at least ONE of these is true:
  1. The creative_direction signature_moment requires real-time rendering (particles, geometry, shaders).
  2. The brand world is inherently spatial (architecture, materials, physical products with geometry).
  3. The ux_structure has a hero or feature section that would be hollow without a 3D element.
  Otherwise set used: false and stop — do not invent unnecessary 3D scenes.

IF used: true, quality bar:
  scenes: Each scene has a concrete description of what geometry/particles/shaders are used,
          the narrative purpose ("creates atmosphere of X"), pointer/scroll interaction behaviour,
          and a realistic performance budget (max_triangles, target_fps on mid-range mobile).
  renderer: "three.js" is the default unless there is a specific reason for babylon.js, spline, or model-viewer.

RULES:
- If used: false, output ONLY {"used": false} — no renderer, no scenes array.
- Performance budgets must be realistic: particle systems = 0 triangles, hero mesh = under 50k triangles.
- target_fps must be 60 for desktop-only experiences, 30 for mobile-first or battery-sensitive contexts.
- camera_movement must describe the rig type (fixed, scroll-driven, pointer-parallax, orbit) concretely.
- Do not create 3D scenes just to be impressive — each scene must serve a specific narrative purpose
  named in the description.
""",

    "asset_manifest": """\
You are a Creative Producer writing the asset_manifest section of a web project brief.

QUALITY BAR — match this level of specificity:
  images:  Each prompt is a full text-to-image generation prompt: lens/focal-length, lighting
           direction, colour treatment, subject pose or framing, background. Not a description
           but an instruction. E.g.: "Extreme close-up of a slow-motion pour of black coffee
           into a white ceramic cup, steam rising, dark background, shot on 85mm f/1.4,
           shallow depth of field, slightly underexposed, editorial photography style"
  style_reference: A named photographer, film stock, director, or visual movement.
                   Not "clean and modern" — "Charlie Schuck food photography — deep shadow,
                   single key light from above-left".
  audio:   If the signature_moment or any component involves audio, specify the exact ambient
           recording with environmental details (birds, machinery, water, human voices at low volume).

RULES:
- Every image prompt must include: subject, setting, camera/lens spec, lighting, colour treatment.
- style_reference must name a real photographer, director, or visual artist — not a genre.
- Every component named in design_system.components that requires a visual asset must have one here.
- If the hero section is atmospheric, include a video loop asset as a WebGL fallback.
- Audio assets are required if the signature_moment involves sound OR if any component is an audio player.
- images[].dimensions must match the usage (hero = 1920×1080, portrait panel = 800×1000, product = 600×600).
- Identify any GAP — a section that relies on typography alone with no visual asset — and note it
  in the usage field of an adjacent asset.
""",
}


# ── JSON extraction ────────────────────────────────────────────────────────────

def extract_json(text: str) -> dict:
    """Extract the first JSON object from a string (strips markdown fences)."""
    # Strip ```json ... ``` fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))
    # Bare JSON — find the outermost { }
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")
    depth, end = 0, -1
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        raise ValueError("Unmatched braces in response")
    return json.loads(text[start : end + 1])


# ── Schema validation ─────────────────────────────────────────────────────────

def validate_section(data: dict, stage: str, full_schema: dict) -> None:
    section_schema = full_schema["properties"][stage]
    # Use the full schema as resolver root so self-referential $ref (e.g. sitemap children) resolves.
    resolver = jsonschema.RefResolver.from_schema(full_schema)
    jsonschema.validate(instance=data, schema=section_schema, resolver=resolver)


# ── Prompt construction ───────────────────────────────────────────────────────

def build_prompt(stage: str, brief_so_far: dict, full_schema: dict) -> str:
    section_schema = full_schema["properties"][stage]
    return f"""\
{STAGE_INSTRUCTIONS[stage]}

PROJECT CONTEXT (brief so far):
{json.dumps(brief_so_far, indent=2)}

OUTPUT SCHEMA for the `{stage}` section:
{json.dumps(section_schema, indent=2)}

Respond with ONLY a valid JSON object matching the schema above — no prose, no markdown fences,
no commentary. The object must be the `{stage}` value (not wrapped in a parent object).
Every string field must be substantive; no placeholder text.
"""


# ── API call ──────────────────────────────────────────────────────────────────

def call_claude(client: anthropic.Anthropic, stage: str, prompt: str) -> tuple[dict, dict]:
    """Call Claude and return (parsed_section_dict, usage_dict). Retries once on JSON error."""
    system = (
        "You are a senior digital agency specialist generating one section of a project brief. "
        "You output ONLY valid JSON — no prose, no markdown, no explanation. "
        "Your output is consumed directly by a JSON parser."
    )
    for attempt in range(2):
        response = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        try:
            return extract_json(raw), usage
        except (json.JSONDecodeError, ValueError) as exc:
            if attempt == 0:
                print(f"  [warn] JSON parse failed ({exc}), retrying…", flush=True)
                continue
            raise RuntimeError(f"Could not parse JSON from Claude response: {exc}\n\nRaw:\n{raw}")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(input_path: Path) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("Error: ANTHROPIC_API_KEY environment variable is not set.")

    with open(input_path) as f:
        project_input = json.load(f)

    required_meta = {"project_name", "industry", "target_audience", "tone_keywords"}
    missing = required_meta - project_input.keys()
    if missing:
        sys.exit(f"Error: input file is missing fields: {', '.join(sorted(missing))}")

    with open(SCHEMA_PATH) as f:
        full_schema = json.load(f)

    client = anthropic.Anthropic(api_key=api_key)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    project_name = project_input["project_name"]
    slug = re.sub(r"[^a-z0-9]+", "-", project_name.lower()).strip("-")
    output_path = OUTPUT_DIR / f"{slug}.json"

    # Resume from existing partial output if present, otherwise start fresh.
    if output_path.exists():
        with open(output_path) as f:
            brief = json.load(f)
        completed = [s for s in STAGES if s in brief]
        if completed:
            print(f"  Resuming — already completed: {', '.join(completed)}\n")
    else:
        brief = {
            "meta": {
                "project_name": project_input["project_name"],
                "industry": project_input["industry"],
                "target_audience": project_input["target_audience"],
                "tone_keywords": project_input["tone_keywords"],
            },
            "taste_scores": {
                "creative_direction": {"score": 0, "notes": "Not yet scored."},
                "visual_design":      {"score": 0, "notes": "Not yet scored."},
                "motion":             {"score": 0, "notes": "Not yet scored."},
                "assets":             {"score": 0, "notes": "Not yet scored."},
                "final":              {"score": 0, "notes": "Not yet scored."},
            },
            "version_history": [
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "stage": "intake",
                    "summary": f"Brief initialised from {input_path.name} for {project_name}.",
                }
            ],
        }

    total_input_tokens = 0
    total_output_tokens = 0

    print(f"\n{'─'*60}")
    print(f"  Project : {project_name}")
    print(f"  Industry: {project_input['industry']}")
    print(f"  Output  : {output_path.relative_to(REPO_ROOT)}")
    print(f"{'─'*60}\n")

    for stage in STAGES:
        if stage in brief:
            print(f"   ↩  {stage} (skipped — already in brief)", flush=True)
            continue
        print(f"▶  {stage}", flush=True)
        t0 = time.perf_counter()

        prompt = build_prompt(stage, brief, full_schema)

        section_data, usage = call_claude(client, stage, prompt)

        # Validate against schema
        try:
            validate_section(section_data, stage, full_schema)
        except jsonschema.ValidationError as exc:
            # Surface the error clearly but continue — partial output is useful
            print(f"  [warn] Schema validation failed: {exc.message}", flush=True)

        brief[stage] = section_data
        brief["version_history"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "summary": _stage_summary(stage, section_data),
        })

        # Write after every stage so a crash mid-run leaves a useful partial brief
        with open(output_path, "w") as f:
            json.dump(brief, f, indent=2, ensure_ascii=False)

        elapsed = time.perf_counter() - t0
        total_input_tokens  += usage["input_tokens"]
        total_output_tokens += usage["output_tokens"]

        print(
            f"   ✓  {elapsed:.1f}s  |  "
            f"in={usage['input_tokens']:,}  out={usage['output_tokens']:,}  "
            f"tokens",
            flush=True,
        )

    print(f"\n{'─'*60}")
    print(f"  Done. {len(STAGES)} stages completed.")
    print(f"  Total tokens — input: {total_input_tokens:,}  output: {total_output_tokens:,}")
    print(f"  Brief saved to: {output_path}")
    print(f"{'─'*60}\n")


def _stage_summary(stage: str, data: dict) -> str:
    summaries = {
        "creative_direction": lambda d: (
            f"Creative direction set: concept anchored on "
            f"'{d.get('concept','')[:80]}…'"
        ),
        "ux_structure": lambda d: (
            f"IA defined: {len(d.get('sitemap', []))} top-level sections; "
            f"navigation pattern locked."
        ),
        "design_system": lambda d: (
            f"Design system: {d.get('typography',{}).get('heading_font',{}).get('name','?')} / "
            f"{d.get('typography',{}).get('body_font',{}).get('name','?')} typefaces; "
            f"{len(d.get('color_palette', []))} palette entries; "
            f"{len(d.get('components', []))} components."
        ),
        "motion_spec": lambda d: (
            f"Motion spec: {len(d.get('global_principles', []))} global principles; "
            f"{len(d.get('per_section', []))} per-section entries."
        ),
        "spatial_3d_spec": lambda d: (
            f"3D spec: used={d.get('used')}; "
            + (f"{len(d.get('scenes', []))} scene(s)." if d.get("used") else "no scenes required.")
        ),
        "asset_manifest": lambda d: (
            f"Asset manifest: {len(d.get('images', []))} images, "
            f"{len(d.get('videos', []))} videos, "
            f"{len(d.get('audio', []))} audio assets."
        ),
    }
    try:
        return summaries[stage](data)
    except Exception:
        return f"{stage} stage completed."


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the staged brief-generation pipeline.")
    parser.add_argument("input", type=Path, help="Path to the project input JSON file.")
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"Error: input file not found: {args.input}")

    run_pipeline(args.input)
