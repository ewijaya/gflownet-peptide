# Social Media Preview Illustration Brief

## Project Context

**Repository:** GFlowNet for Therapeutic Peptide Generation
**Purpose:** GitHub social media preview image
**Target Audience:** Biotech/pharma scientists, computational biologists, ML researchers in drug discovery

### What is GFlowNet?

GFlowNet (Generative Flow Network) is a machine learning method that generates molecular sequences by making sequential decisions. It builds peptides one amino acid at a time, learning to sample diverse, high-quality sequences proportionally to their predicted therapeutic value.

---

## Technical Specifications

| Attribute | Requirement |
|-----------|-------------|
| **Dimensions** | 1280 × 640 pixels (2:1 aspect ratio) |
| **Minimum size** | 640 × 320 pixels |
| **File format** | PNG (preferred) or JPG |
| **Color mode** | RGB |
| **Safe zone** | Keep critical elements within center 80% (edges may crop on some platforms) |

---

## Creative Direction

### Concept: The Generative Cascade

A cross-sectional view of a branching decision tree that visualizes how GFlowNet builds peptide sequences through sequential amino acid choices. The illustration should feel like a scientific figure from a high-impact journal—rigorous, precise, yet visually compelling.

### Tone

- Serious and academic
- Clean and precise
- Sophisticated, not flashy
- Conveys scientific rigor

### Color Palette

**Primary:** Monochrome (black, white, grays)
**Accent:** Single accent color for emphasis

| Role | Color Suggestion | Usage |
|------|------------------|-------|
| Background | Off-white or very light gray (#F5F5F5 to #FAFAFA) | Canvas |
| Primary lines/nodes | Dark gray to black (#1A1A1A to #333333) | Tree structure |
| Secondary lines | Medium gray (#888888 to #AAAAAA) | Lower-probability paths |
| Accent | Deep teal (#0D7377), scientific blue (#2E5EAA), or muted gold (#B8860B) | Terminal high-reward peptides, key highlights |

*Note: Choose ONE accent color. Teal or blue feels more biotech; gold feels more prestigious/academic.*

---

## Composition Details

### Overall Layout

```
[LEFT]                    [CENTER]                    [RIGHT]

Single root node    →    Branching structure    →    Multiple terminal nodes
(Generation start)       (Decision cascade)          (Complete peptides)
```

- **Orientation:** Left-to-right flow (representing sequential generation)
- **Shape:** Tree/directed acyclic graph expanding from left to right
- **Balance:** Denser branching in center, clean endpoints on right

### Structural Elements

#### 1. Root Node (Left Side)
- Single circular node representing the START token
- Position: ~10-15% from left edge, vertically centered
- Size: Slightly larger than internal nodes
- Style: Solid fill, dark gray or black

#### 2. Internal Nodes (Center)
- Represent amino acid choices at each position
- Arranged in vertical "layers" moving left to right
- Each layer = one position in the peptide sequence
- **Suggested layers:** 5-7 visible layers (not too cluttered)
- **Nodes per layer:** Varies (shows branching/pruning)
  - Layer 1: 3-4 nodes
  - Layer 2: 5-7 nodes
  - Layer 3: 8-10 nodes (maximum spread)
  - Layer 4: 6-8 nodes
  - Layer 5: 4-6 nodes
  - Layer 6-7: 3-5 nodes (convergence toward high-reward)
- Node style: Small circles, uniform size, medium gray fill

#### 3. Terminal Nodes (Right Side)
- Represent completed peptide sequences
- Position: ~85-90% from left edge
- **Key detail:** 2-3 terminal nodes highlighted with accent color (these are the high-reward peptides)
- Remaining terminal nodes in standard gray
- Accent nodes slightly larger to draw attention

#### 4. Edges (Connections)
- Directed edges connecting nodes from left to right
- **Line weight varies by probability:**
  - Thick lines (2-3px): High-probability paths
  - Medium lines (1-1.5px): Moderate-probability paths
  - Thin lines (0.5-1px): Low-probability paths
- **Line style:** Straight or very subtle curves (not chaotic)
- **Flow indication:** Subtle arrowheads on edges OR tapered lines (thick→thin) to show direction
- Edges leading to accent terminal nodes should be more prominent

#### 5. Probability Flow Visualization
- The "cascade" effect: visual sense that probability mass flows and concentrates
- Achieve through:
  - Line weight variation (described above)
  - Subtle opacity gradient (paths to low-reward terminals more transparent)
  - Convergence pattern (many paths funnel toward fewer high-reward outputs)

---

## Visual Hierarchy

1. **Primary focus:** The 2-3 accent-colored terminal nodes (high-reward peptides)
2. **Secondary focus:** The thick probability paths leading to them
3. **Context:** The broader tree structure showing the decision space
4. **Background:** The pruned/low-probability paths (subtle, not distracting)

---

## Style Reference

### What to Emulate
- Scientific journal figures (Nature, Science, Cell style)
- Network/graph visualizations from computational biology papers
- Clean infographics from biotech companies (e.g., Recursion, Generate Biomedicines)
- Sankey diagrams (for flow visualization inspiration)

### What to Avoid
- Overly artistic or abstract interpretations
- Neon colors or gradients
- 3D effects or heavy shadows
- Decorative elements without meaning
- Busy or cluttered compositions
- Cartoon-like or playful styling

---

## Scientific Accuracy Notes

For the illustrator's understanding (not necessarily visible in final art):

1. **Amino acids:** There are 20 standard amino acids. Each node in the tree represents choosing one amino acid to add to the growing sequence.

2. **Sequential generation:** Peptides are built left-to-right, one amino acid at a time. This is why the tree flows in that direction.

3. **Probability proportional to reward:** GFlowNet samples sequences with probability proportional to their "reward" (predicted therapeutic value). This is why some paths are thicker—they're more likely to be taken.

4. **Diversity:** Unlike other methods that collapse to a single "best" answer, GFlowNet maintains diverse outputs. The multiple terminal nodes represent this diversity.

5. **The STOP decision:** Generation ends when the model chooses to stop. Terminal nodes represent this stopping point.

---

## Refinement Options

### Option A: Minimal
- Pure geometric: circles and lines only
- Maximum whitespace
- Very few nodes (simplified tree)
- Single accent highlight

### Option B: Moderate (Recommended)
- As described above
- Balanced detail and clarity
- Clear visual hierarchy
- Readable at small sizes

### Option C: Detailed
- More nodes and layers
- Subtle background grid or axis lines
- Small amino acid letter labels inside select nodes
- More nuanced line weight variation

---

## Deliverables Requested

1. **Final illustration:** 1280 × 640 px, PNG format
2. **Vector source file:** AI, SVG, or PDF (for future modifications)
3. **Variations (optional):**
   - Dark mode version (dark background, light elements)
   - Square crop version (640 × 640) for other platforms

---

## Example Sketch

```
                                    ○
                               /
                          ○───○     ○ ← (gray, low reward)
                     /   /     \   /
                ○───○───○       ○─○
               /     \   \     /   \
    [START]  ●        ○───○───○     ◉ ← (ACCENT: high reward)
               \     /   /     \   /
                ○───○───○       ○─○
                     \   \     /   \
                          ○───○     ◉ ← (ACCENT: high reward)
                               \
                                    ○

    ←─────────────────────────────────→
    Generation proceeds left to right

    ● = Start node (black)
    ○ = Internal/terminal nodes (gray)
    ◉ = High-reward terminal nodes (accent color)

    Line thickness indicates probability
```

---

## Contact for Questions

If any aspect of this brief is unclear, please reach out before proceeding. Key clarifications likely needed:

- Exact accent color preference
- Level of detail (Option A/B/C)
- Whether dark mode version is needed
- Any additional scientific context

---

## Approval Process

1. Initial sketch/wireframe for concept approval
2. Refined grayscale version for structure approval
3. Final colored version with accent
4. Delivery of all file formats

---

*Brief prepared for GFlowNet Peptide Generation repository*
*GitHub: github.com/[your-username]/gflownet-peptide*
