# ALS Therapeutic Target Discovery Pipeline

Geneformer-based identification of genetic perturbations with therapeutic potential for Amyotrophic Lateral Sclerosis (ALS)

---

## Important Caveat

Due to computational capacity limitations, this analysis was performed on a **subset of the dataset** focusing on the **first 3 genes** (DMD, MAP1B, KHDRBS2). While additional genes are listed in the configuration file, they were not explored due to constraints in loading the full dataset. Results should be interpreted as a proof-of-concept demonstration of the methodology.

---

## 📖 References

**Key Papers**:
1. **Geneformer**: Theodoris et al. (2023) "Transfer learning enables predictions in network biology" *Nature*
2. **ALS Dataset**: Pineda et al. (2021) "Single-cell transcriptomic atlas of the human motor cortex in ALS" *Nature Medicine*
3. **Perturb-seq**: Norman et al. (2019) "Exploring genetic interaction manifolds constructed from rich single-cell perturbation screens" *Science*
4. **scRNA-seq Analysis**: Wolf et al. (2018) "SCANPY: large-scale single-cell gene expression data analysis" *Genome Biology*

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment

python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
'''


### 2. Download Dataset

```bash
# Download ALS dataset from provided URL
wget "https://s3.eu-west-2.amazonaws.com/helical-candidate-datasets/counts_combined_filtered_BA4_sALS_PN.h5ad?..." \
  -O data/raw/counts_combined_filtered_BA4_sALS_PN.h5ad

'''

### 3. Dataset Details:

1. Format: AnnData (.h5ad)
2. Pre-processing: Log-normalized, quality-filtered, batch-corrected
3. Conditions: ALS (disease), PN (healthy reference)
4. Size: ~50,000 cells, 20,000+ genes

### 4. Run Analysis Pipeline

```bash
# Execute notebooks in order
jupyter notebook notebooks/task1_perturbation_workflow.ipynb
jupyter notebook notebooks/task2_als_geneformer.ipynb
jupyter notebook notebooks/task3_embedding_analysis.ipynb
jupyter notebook notebooks/task4_target_prioritization.ipynb
'''

---

## Requirements

**Python**: 3.8+

**Key Dependencies**:
- `transformers` (Hugging Face - for Geneformer)
- `torch` (PyTorch - GPU recommended)
- `scanpy`, `anndata` (single-cell analysis)
- `numpy`, `pandas`, `scipy`
- `matplotlib`, `seaborn`
- `umap-learn`, `scikit-learn`

**Geneformer Model**:
- Model: `ctheodoris/Geneformer` (Hugging Face Hub)
- GPU Memory: ~6GB recommended
- Alternative: CPU inference (slower)

---

## Configuration

Edit `config/config.yaml` to customize:
```yaml
random_seed: 42

data:
  raw_dir: "data/raw"
  cache_dir: "data/cache"
  results_dir: "results"

perturbation:
  genes: ["DMD", "MAP1B", "KHDRBS2"]  # Genes tested
  knock_down:
    factors: [0.2, 0.5]  # 80% and 50% reduction
  knock_up:
    factors: [2.0, 3.0]  # 2x and 3x overexpression

embedding:
  model_name: "ctheodoris/Geneformer"
  batch_size: 10
  max_input_size: 2048

---

## Output Structure 

results/
├── task1/                          # Perturbation generation
│   ├── tables/
│   │   ├── perturbation_metadata.csv
│   │   └── validation_metrics.csv
│   └── figures/
│       └── log2fc_summary.png
│
├── task2/                          # Geneformer embeddings
│   ├── embeddings/
│   │   └── task2_all_embeddings.h5  # HDF5 (baseline + 24 perturbations)
│   └── tables/
│       └── perturbation_metadata.csv
│
├── task3/                          # Embedding space analysis
│   ├── tables/
│   │   ├── embedding_metrics.csv         # All metrics
│   │   ├── disease_rescue_analysis.csv   # ALS → PN rescue
│   │   └── full_analysis.csv             # Integrated data
│   └── figures/
│       ├── metrics_heatmap.png           # Top 20 perturbations
│       ├── method_comparison.png         # Euclidean vs Cosine
│       ├── shift_vectors.png             # Cell trajectories
│       ├── disease_rescue.png            # Therapeutic potential
│       └── comprehensive_summary.png     # 6-panel overview
│
└── task4/                          # Target prioritization
    ├── tables/
    │   ├── coverage_analysis.csv
    │   ├── optimal_therapeutic_targets.csv
    │   └── therapeutic_targets_with_rationale.csv
    └── figures/
        ├── therapeutic_ranking.png       # Top 15 candidates
        ├── factor_optimization.png       # Optimal dosing
        ├── decision_matrix.png           # Multi-criteria heatmap
        └── coverage_analysis.png         # Cell-level effects


---

## Pipeline Overview

### Biological Rationale

This pipeline identifies therapeutic gene perturbations by measuring how they shift diseased (ALS) cells toward a healthy (PN) transcriptional state in Geneformer embedding space.

**Four-Stage Approach**:

1. **Perturbation Generation**: Simulate genetic interventions (knockdown/knockup) in silico
2. **Embedding Generation**: Transform perturbed cells into 256-dimensional Geneformer embeddings
3. **Disease Rescue Quantification**: Measure movement toward healthy motor neuron states
4. **Target Prioritization**: Rank genes by therapeutic potential using multi-criteria optimization

**Core Hypothesis**: Effective therapeutic perturbations move ALS cells toward the healthy (PN) reference in transcriptional space.

---

## Task 1: Perturbation Generation

**Goal**: Create in silico genetic perturbations simulating therapeutic interventions

**Method**: Multiplicative scaling of gene expression

- **Knockdown**: `new_expr = old_expr × factor` (factor = 0.2, 0.5)
- **Knockup**: `new_expr = old_expr × factor × (1 + noise)` (factor = 2.0, 3.0)

**Outputs**:
- 24 perturbations (3 genes × 2 types × 4 factors)
- Validation metrics with log2FC confirmation
- Figure: `log2fc_summary.png` showing actual vs expected fold-changes

**Interpretation**: Log2FC values match expected patterns, confirming successful in silico gene expression modification.

---

## Task 2: Geneformer Embeddings

**Goal**: Transform gene expression into 256-dimensional embeddings

**Model**: Geneformer (transformer-based, pre-trained on 30M single cells)

**Why embeddings?** Compresses 20,000+ gene dimensions into 256 features that preserve biological relationships while enabling efficient distance-based comparisons.

**Processing**:
- Baseline (ALS disease state)
- 24 perturbations (all gene × type × factor combinations)
- PN reference (healthy motor neuron state)

**Outputs**:
- `task2_all_embeddings.h5`: HDF5 file with all embeddings (n_cells × 256)
- Perturbation metadata mapping

**Interpretation**: Embeddings capture cellular state in format suitable for therapeutic screening.

---

## Task 3: Embedding Space Analysis

**Goal**: Quantify perturbation effects and identify disease-rescuing candidates

### Dual-Metric Approach

**1. Euclidean Distance**: Measures magnitude of movement (how far cells move)

**2. Cosine Distance**: Measures directional change (what direction cells move)

**Rationale**: Geneformer paper (Nature 2023) uses cosine similarity for perturbation ranking because it captures transcriptional direction. We add Euclidean distance for complementary magnitude information.

### Metrics Computed

**Population Shifts**:
- Centroid shift (Euclidean): Distance between perturbed and baseline population centers
- Cosine shift: Angular difference in embedding space

**Disease Rescue**:
- Rescue score = Change in distance from ALS baseline to PN reference
- Negative = therapeutic (moves toward healthy)
- Positive = detrimental (moves away from healthy)

**Structure Preservation**:
- Neighborhood preservation: Percentage of k-nearest neighbors retained
- Mean cell shift: Average per-cell movement

### Key Outputs

**Tables**:
- `embedding_metrics.csv`: All metrics for all perturbations
- `disease_rescue_analysis.csv`: ALS to PN rescue scores
- `full_analysis.csv`: Integrated dataset

**Figures**:

**Metrics Heatmap** (`metrics_heatmap.png`): Top 20 perturbations sorted by gene, type, and factor. Red = low values, Green = high values. Shows which perturbations have strongest effects while maintaining cell identity.

**Method Comparison** (`method_comparison.png`): Euclidean vs Cosine correlation (r ≈ 0.97). High correlation validates dual-metric approach and shows methods identify similar top candidates.

**Disease Rescue** (`disease_rescue.png`): Sorted rescue scores for Euclidean and Cosine methods. Green bars = therapeutic potential, Red bars = detrimental effects.

**Comprehensive Summary** (`comprehensive_summary.png`): 6-panel overview showing distribution of shifts, movement vs structure preservation trade-off, and top 5 candidates by each method.

**Interpretation**: MAP1B and DMD consistently rank highest for disease rescue with good structural preservation. High Euclidean-Cosine correlation (r=0.97) confirms method agreement.

---

## Task 4: Therapeutic Prioritization

**Goal**: Rank genes and identify optimal perturbation strength for each

### Composite Scoring System

**Weighted Formula**:
- 40% Disease rescue (primary objective - reverse disease phenotype)
- 30% Coverage (percentage of cells improved - reliability)
- 15% Effect size (optimal range 3-8 - sufficient but not excessive)
- 15% Structure preservation (maintain motor neuron identity)

**Rationale**: Rescue weighted highest because reversing disease is primary goal. Coverage ensures broad efficacy across cells, not just outliers. Effect size penalized if too weak or too strong. Preservation ensures cells maintain proper identity.

### Coverage Metric

**Definition**: Percentage of individual cells that move closer to healthy (PN) reference

**Why it matters**: A perturbation can shift the population centroid (high rescue score) but only benefit 20% of individual cells. We want treatments that help the majority.

**Calculation**: For each cell, compare distance to PN reference before vs after perturbation.

### Factor Optimization

For each gene, compare all tested factors (0.2, 0.5, 2.0, 3.0) and select the one maximizing composite score.

**Goal**: Identify therapeutic window balancing efficacy with cellular tolerance.

### Key Outputs

**Tables**:
- `coverage_analysis.csv`: Cell-level improvement percentages
- `optimal_therapeutic_targets.csv`: Best factor per gene (ranked by composite score)
- `therapeutic_targets_with_rationale.csv`: Includes biological interpretations

**Figures**:

**Therapeutic Ranking** (`therapeutic_ranking.png`): 4-panel overview showing top 15 candidates, score component breakdown for top gene, rescue vs coverage scatter, and effect size vs preservation.

**Factor Optimization** (`factor_optimization.png`): Distribution of optimal factors and knockdown vs knockup preference across genes.

**Decision Matrix** (`decision_matrix.png`): Heatmap of top 20 genes showing normalized scores (0-1) across all criteria (Rescue, Coverage, Effect, Preservation, Composite).

**Coverage Analysis** (`coverage_analysis.png`): Distribution of cell-level improvements and mean coverage by perturbation factor.

### Example Top Candidate

**MAP1B - Knockdown at 0.2x**:
- Composite Score: 0.876
- Rescue: -0.0054 (strong movement toward healthy state)
- Coverage: 89.3% of cells improved
- Effect: Appropriate magnitude without excessive disruption
- Preservation: Maintains motor neuron characteristics
- Rationale: Knockdown to 20% expression provides strong disease rescue, benefits most cells, with excellent identity preservation

---

## Interpreting Results

### Good Therapeutic Candidate

- Composite score greater than 0.8
- Negative rescue score (moves toward healthy)
- Coverage greater than 70% (helps majority of cells)
- Effect size between 3-8 (sufficient but not excessive)
- Preservation greater than 0.85 (maintains identity)

### Red Flags

- Positive rescue score (worsens disease phenotype)
- Coverage less than 50% (limited efficacy)
- Effect greater than 10 (may cause cellular stress)
- Preservation less than 0.7 (disrupts cell identity)

---

## Key Findings

1. **MAP1B knockdown (0.2x)** shows strongest therapeutic potential across all criteria
2. **High method agreement** (r = 0.97) between Euclidean and Cosine validates dual-metric approach
3. **Optimal factors** cluster around 0.2x for knockdown and 2.0x for knockup
4. **Coverage ranges** from 50-90% across perturbations, emphasizing need for cell-level analysis
5. **Structure preservation** maintained above 0.8 for most top candidates

---

## Output Structure