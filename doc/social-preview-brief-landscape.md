# Social Media Preview Illustration Brief: Probability Landscape

## Project Context

**Repository:** GFlowNet for Therapeutic Peptide Generation
**Purpose:** GitHub social media preview image
**Target Audience:** Biotech/pharma scientists, computational biologists, ML researchers in drug discovery

### What is GFlowNet?

GFlowNet (Generative Flow Network) is a machine learning method that generates molecular sequences by learning to navigate a "reward landscape." It explores the vast space of possible peptide sequences and learns to sample diverse, high-quality sequences proportionally to their predicted therapeutic value—finding multiple peaks, not just the single highest one.

---

## Technical Specifications

| Attribute | Requirement |
|-----------|-------------|
| **Dimensions** | 1280 × 640 pixels (2:1 aspect ratio) |
| **Minimum size** | 640 × 320 pixels |
| **File format** | PNG (preferred) or JPG |
| **Color mode** | RGB |
| **Safe zone** | Leave 40pt border around all edges (critical elements may crop outside this zone) |

---

## Creative Direction

### Concept: The Probability Landscape

A 3D topographical surface representing the "reward landscape" that GFlowNet learns to navigate. The terrain has peaks (high-reward peptide sequences) and valleys (low-reward sequences). Flowing particles or paths show how the algorithm explores this landscape and converges toward multiple high-reward regions simultaneously.

### Core Metaphor

Imagine a mountainous terrain viewed from an elevated angle. Instead of water flowing downhill, probability mass flows uphill toward the peaks. Multiple streams of particles climb toward different summits, representing GFlowNet's ability to discover diverse high-quality solutions rather than collapsing to a single answer.

### Tone

- Serious and academic
- Mathematically evocative
- Clean and precise
- Conveys optimization and exploration

### Color Palette

**Primary:** Monochrome (black, white, grays)
**Accent:** Single accent color for emphasis

| Role | Color Suggestion | Usage |
|------|------------------|-------|
| Background | Off-white (#F8F8F8) or subtle warm gray (#F5F3F0) | Sky/canvas behind terrain |
| Terrain surface | Gradient from dark gray (#2A2A2A) to light gray (#D0D0D0) | 3D surface with shading |
| Contour lines | Medium gray (#777777) with varying opacity | Topographical detail |
| Grid lines (optional) | Light gray (#CCCCCC), subtle | Mathematical foundation |
| Accent | Deep teal (#0D7377), scientific blue (#2E5EAA), or ember orange (#D35400) | Peaks, particle paths, key highlights |

*Note: Choose ONE accent color. Teal/blue feels analytical; orange suggests energy/activity.*

---

## Composition Details

### Overall Layout

```
[ELEVATION VIEW - Looking at terrain from ~30-45° above horizon]

         Peak A                    Peak B
           /\                        /\
          /  \    valley    ridge   /  \      Peak C
         /    \__________/\_______ /    \       /\
        /                          \     \     /  \
    ___/                            \     \___/    \___
   /                                 \                 \
  /                                   \                 \
 ──────────────────────────────────────────────────────────
                    [BASE PLANE]

 ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ →
                   Sequence Space
```

- **Viewpoint:** Isometric or slight perspective, elevated ~30-45° above horizontal
- **Horizon:** Upper third of frame, giving prominence to the terrain
- **Depth:** Terrain should have clear foreground, midground, and background

### Structural Elements

#### 1. The Terrain Surface

**Shape and topology:**
- Continuous 3D surface representing the reward function over sequence space
- **Multiple peaks** (critical): 3-5 distinct peaks of varying heights
  - One dominant peak (highest reward region)
  - 2-3 secondary peaks (local optima, still valuable)
  - Peaks should not be uniform—vary in height, width, and steepness
- **Valleys and saddles:** Low regions between peaks
- **Ridges:** Connecting pathways between some peaks

**Surface rendering:**
- Smooth, continuous surface (not jagged or noisy)
- Clear elevation shading: darker at base, lighter at peaks (or vice versa)
- Subtle grid lines or contour lines to emphasize the mathematical nature
- Surface should feel solid and tangible, not ethereal

**Contour lines (optional but recommended):**
- Topographical contour lines at regular elevation intervals
- Tighter spacing near peaks (steeper gradients)
- Wider spacing in valleys and plateaus
- Rendered in subtle gray, not competing with main elements

#### 2. The Particle Flows

**Purpose:** Show probability mass flowing toward high-reward regions

**Visual representation options:**

*Option A: Particle streams*
- Small dots/particles moving uphill toward peaks
- Streams originate from various points across the landscape
- Particles converge toward peaks, with density increasing near summits
- Motion blur or trailing effect to suggest movement

*Option B: Flow lines/paths*
- Curved lines tracing paths from base to peaks
- Lines thicken or intensify as they approach peaks
- Multiple lines converging on each peak
- Similar to wind flow maps or magnetic field lines

*Option C: Gradient arrows (most technical)*
- Small arrows scattered across surface pointing "uphill"
- Arrow size/opacity proportional to gradient steepness
- Arrows converge at peaks, sparse in flat regions

**Flow behavior:**
- Flows should reach ALL significant peaks, not just the highest
- This demonstrates GFlowNet's diversity property
- More flow toward higher peaks (proportional sampling)
- Some paths should branch—showing exploration

**Accent color application:**
- The particle flows/paths rendered in accent color
- Alternatively: peaks themselves glow with accent color
- Or both: subtle accent on paths, stronger accent on peak summits

#### 3. The Base Plane

**Purpose:** Ground the visualization, suggest the underlying sequence space

**Rendering:**
- Flat plane beneath the terrain surface
- Subtle grid pattern suggesting coordinates/dimensionality
- Fades toward edges (vignette effect)
- Should not compete with terrain for attention

#### 4. Peak Highlights

**The summits should draw the eye:**
- Accent color glow or highlight at peak tops
- Could be rendered as:
  - Glowing points/orbs at summits
  - Concentration of particle density
  - Subtle radial gradient emanating from peaks
- Highest peak should be most prominent, but others clearly visible

---

## Visual Hierarchy

1. **Primary focus:** The multiple peaks with accent highlights
2. **Secondary focus:** The flowing particles/paths climbing toward peaks
3. **Supporting context:** The terrain surface showing topology
4. **Background:** Base plane and subtle mathematical grid

---

## Depth and Perspective

### Recommended Viewpoint

- **Angle:** 30-45° elevation above horizontal plane
- **Rotation:** Slight rotation so terrain extends diagonally (more dynamic than straight-on)
- **Focal point:** Composition centered on the main peak cluster
- **Depth of field:** Optional subtle blur on far background terrain

### Atmospheric Perspective (Optional)

- Distant terrain slightly faded/lighter
- Creates depth without complexity
- Foreground elements crisp and detailed

---

## Style Reference

### What to Emulate

- Topographical maps and terrain visualizations
- 3D surface plots from scientific papers (MATLAB, Python matplotlib style but polished)
- Elevation models used in geology/geography
- Optimization landscape visualizations from ML papers
- Flow field visualizations (ocean currents, wind patterns)

### Specific References

- Loss landscape visualizations (e.g., "Visualizing the Loss Landscape of Neural Nets" paper)
- Potential energy surfaces from computational chemistry
- Fitness landscapes from evolutionary biology
- Contour plots with 3D perspective from Nature/Science papers

### What to Avoid

- Photorealistic mountain renders (too literal)
- Fantasy/game terrain styling
- Excessive texture or noise
- Harsh shadows or dramatic lighting
- Busy or chaotic particle systems
- Rainbow color gradients
- Wireframe-only rendering (too sparse)

---

## Scientific Accuracy Notes

For the illustrator's understanding:

1. **Reward landscape:** In optimization, we imagine all possible solutions arranged in a space, with "height" representing quality. GFlowNet learns to navigate this landscape.

2. **Multiple peaks matter:** Traditional optimization finds ONE best solution. GFlowNet finds MANY good solutions (multiple peaks). This is its key advantage—the illustration should emphasize multiplicity.

3. **Proportional sampling:** GFlowNet samples peaks proportionally to their height. Higher peaks get more samples, but lower peaks aren't ignored. The flow visualization should reflect this—more particles toward higher peaks, but still some toward secondary peaks.

4. **High dimensionality:** The real landscape is extremely high-dimensional (impossible to visualize). This 3D terrain is a conceptual simplification—a metaphor, not literal representation.

5. **Exploration vs. exploitation:** The flows show both exploring the landscape (spreading out) and exploiting good regions (converging on peaks).

---

## Refinement Options

### Option A: Minimal

- Single clean terrain surface
- 2-3 peaks only
- Simple accent dots at peaks
- No particle flows, just static terrain
- Maximum whitespace

### Option B: Moderate (Recommended)

- Detailed terrain with 3-5 peaks
- Contour lines for mathematical feel
- Clear particle flows in accent color
- Balanced complexity and clarity
- Readable at small sizes

### Option C: Detailed

- Rich terrain with subtle texture
- Dense contour lines
- Animated-feeling particle streams (motion blur)
- Subtle grid on base plane
- Atmospheric depth effects

---

## Composition Sketch

```
                                    ·  ·
                                   · ** ·         (Peak B - accent glow)
                                  ·  /\  ·
          · · ·                    ╱    ╲
         · *** ·    ___           ╱      ╲    · ·
        ·  /\   ·  /   \    _____╱        ╲  · * ·   (Peak C)
         ╱    ╲   /     \  /              ╲  ╱\
        ╱      ╲_/       \/                ╲/  \
       ╱                                       ╲
    __╱    ← ← ←    ↖ ↖ ↖    ↑ ↑ ↑   ↗ ↗        ╲__
   /        (particle flows moving toward peaks)    \
  /___________________________________________________\
  |  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  · |
  |  ·  ·  ·  ·  ·  [base plane grid]  ·  ·  ·  ·  · |

  (Peak A - highest,
   strongest accent)

  Legend:
  *** = Accent color glow at peaks
  ← ↖ ↑ = Particle flow directions (rendered as streams or arrows)
  /\ ╱╲ = Terrain surface
  · · · = Subtle grid on base plane
```

---

## Color Application Summary

| Element | Monochrome Treatment | Accent Application |
|---------|---------------------|-------------------|
| Terrain surface | Gray gradient (dark valleys, light ridges) | None |
| Contour lines | Medium gray, varying opacity | None |
| Base plane | Light gray with subtle grid | None |
| Particle flows | — | Full accent color |
| Peak summits | — | Accent glow/highlight |
| Background | Off-white to white gradient | None |

---

## Deliverables Requested

1. **Final illustration:** 1280 × 640 px, PNG format
2. **Vector source file:** AI, SVG, or PDF (for future modifications)
3. **Variations (optional):**
   - Dark mode version (dark background, light terrain)
   - Square crop version (640 × 640) for other platforms

---

## Comparison with Alternative Concept

This "Probability Landscape" concept differs from the "Generative Cascade" (tree/graph) concept:

| Aspect | Probability Landscape | Generative Cascade |
|--------|----------------------|-------------------|
| Metaphor | Mountain terrain | Decision tree |
| Emphasis | Reward function, optimization | Sequential generation |
| Feeling | Exploration, discovery | Construction, assembly |
| Visual style | Organic, topographical | Structured, diagrammatic |
| Scientific angle | Optimization theory | Probabilistic modeling |

Both are valid representations—this concept emphasizes the exploration/optimization aspect of GFlowNet.

---

## Contact for Questions

If any aspect of this brief is unclear, please reach out before proceeding. Key clarifications likely needed:

- Exact accent color preference
- Level of detail (Option A/B/C)
- Particle flow style preference (streams vs. lines vs. arrows)
- Whether dark mode version is needed

---

## Approval Process

1. Initial sketch/wireframe for composition approval
2. Grayscale terrain render for form approval
3. Add particle flows and accent color
4. Final refinements and delivery

---

*Brief prepared for GFlowNet Peptide Generation repository*
*Alternative concept: Probability Landscape (vs. Generative Cascade)*
