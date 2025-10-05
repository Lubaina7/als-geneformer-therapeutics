# ============================================================================
# utils/embeddings.py - FIXED TOKENIZATION FOR GENEFORMER
# ============================================================================
"""
GeneFormer V2 Embedding Generation and Analysis
==============================================

Integrates with GeneFormer V2 to generate embeddings.
Requires GeneFormer installation.
"""

import numpy as np
import pandas as pd
import anndata as ad
from typing import Dict, Optional, List
from scipy.spatial.distance import cdist, euclidean, cosine

# Import GeneFormer classes (REQUIRED)
try:
    from geneformer import TranscriptomeTokenizer, EmbExtractor
    GENEFORMER_AVAILABLE = True
except ImportError:
    GENEFORMER_AVAILABLE = False


class GeneFormerEmbedder:
    """
    Wrapper for GeneFormer V2 model.
    
    REQUIRES GeneFormer installation:
        pip install geneformer
    """
    
    def __init__(self, 
                 model_name: str = "gf-12L-95M-i4096",
                 embedding_dim: int = 256,
                 emb_layer: int = -1):
        """
        Initialize GeneFormer V2 embedder.
        
        Parameters:
            model_name: GeneFormer model path
            embedding_dim: Embedding dimension (256 for GeneFormer)
            emb_layer: Which layer to extract (-1 = final layer)
        """
        if not GENEFORMER_AVAILABLE:
            raise ImportError(
                "GeneFormer is required but not installed.\n"
                "Install with: pip install geneformer\n"
                "See: https://huggingface.co/ctheodoris/Geneformer"
            )
        
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.emb_layer = emb_layer
        
        print(f"Initializing GeneFormer V2: {model_name}")
        
        # Check if CUDA is available
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Using device: {self.device}")
        
        # Initialize GeneFormer components
        self.embedder = EmbExtractor(
            model_type="Pretrained",
            emb_layer=emb_layer,
            emb_label=None,
            max_ncells=None,
            forward_batch_size=50,  # Smaller batch for CPU
            nproc=1  # Single process
        )
        
        # Initialize tokenizer
        self.tokenizer = TranscriptomeTokenizer(
            nproc=1  # Single process for small datasets
        )
        
        print("✓ GeneFormer V2 initialized")
        print(f"  Model: {model_name}")
        print(f"  Embedding dim: {embedding_dim}")
        if self.device == "cpu":
            print(f"  ⚠ Running on CPU (GPU not available)")
    
    def embed_adata(self, 
                   adata: ad.AnnData, 
                   batch_size: int = 50) -> np.ndarray:
        """
        Generate embeddings using GeneFormer V2.
    
        Parameters:
            adata: AnnData object to embed
            batch_size: Batch size for processing
        
        Returns:
            Embeddings (n_cells × 256)
        """
        import tempfile
        import os
        import torch
    
        print(f"Generating GeneFormer embeddings for {adata.n_obs:,} cells...")
        
        # Force CPU usage if no CUDA available
        if not torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            print("  ⚠ No GPU detected - using CPU (this will be slower)")
        
        # Move model to CPU if needed
        if self.device == "cpu":
            torch.set_default_device("cpu")
    
        # Prepare data for GeneFormer
        adata_gf = adata.copy()
    
        # GeneFormer requires 'ensembl_id' in var and 'n_counts' in obs
        if 'ensembl_id' not in adata_gf.var.columns:
            if 'ENSID' in adata_gf.var.columns:
                adata_gf.var['ensembl_id'] = adata_gf.var['ENSID']
            elif 'gene_ids' in adata_gf.var.columns:
                adata_gf.var['ensembl_id'] = adata_gf.var['gene_ids']
            else:
                raise ValueError(
                    "GeneFormer requires Ensembl IDs.\n"
                    f"Available var columns: {list(adata_gf.var.columns)}\n"
                    "Need 'ENSID' or 'gene_ids' column."
                )
    
        if 'n_counts' not in adata_gf.obs.columns:
            if 'total_counts' in adata_gf.obs.columns:
                adata_gf.obs['n_counts'] = adata_gf.obs['total_counts']
            else:
                adata_gf.obs['n_counts'] = np.array(adata_gf.X.sum(axis=1)).flatten()
    
        # Create temp directory for GeneFormer intermediate files
        temp_dir = tempfile.mkdtemp()
        h5ad_dir = os.path.join(temp_dir, "input")
        os.makedirs(h5ad_dir, exist_ok=True)
        
        h5ad_path = os.path.join(h5ad_dir, "data.h5ad")
        output_dir = os.path.join(temp_dir, "embeddings")
        tokenized_dir = os.path.join(temp_dir, "tokenized")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(tokenized_dir, exist_ok=True)
    
        try:
            # Save h5ad
            adata_gf.write_h5ad(h5ad_path)
        
            # Tokenize - returns (tokenized_cells, cell_metadata, tokenized_counts)
            print("  Tokenizing...")
            tokenized_cells, cell_metadata, tokenized_counts = self.tokenizer.tokenize_anndata(h5ad_path)
            
            # Create dataset from tokenized cells
            print("  Creating dataset...")
            tokenized_dataset = self.tokenizer.create_dataset(
                tokenized_cells, 
                cell_metadata,
                tokenized_counts,
                use_generator=False
            )
            
            # Save to disk
            tokenized_path = os.path.join(temp_dir, "tokenized.dataset")
            tokenized_dataset.save_to_disk(tokenized_path)
        
            # Extract embeddings
            print("  Extracting embeddings...")
            self.embedder.extract_embs(
                model_directory=self.model_name,
                input_data_file=tokenized_path,
                output_directory=output_dir,
                output_prefix="emb",
                output_torch_embs=False,
                cell_state=None
            )
        
            # Load embeddings
            emb_file = os.path.join(output_dir, "emb.csv")
            if not os.path.exists(emb_file):
                # Try alternative output format
                emb_file = os.path.join(output_dir, "emb_embs.csv")
            
            if not os.path.exists(emb_file):
                raise FileNotFoundError(
                    f"Embedding file not found. Expected: {emb_file}"
                )
            
            embeddings_df = pd.read_csv(emb_file, index_col=0)
            embeddings = embeddings_df.values
        
        finally:
            # Clean up temp directory
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
        print(f"✓ Embeddings generated: {embeddings.shape}")
        return embeddings


class EmbeddingAnalyzer:
    """Analyze perturbation effects in embedding space."""
    
    def __init__(self, 
                 baseline_embeddings: np.ndarray,
                 perturbation_embeddings: Dict[str, np.ndarray],
                 metadata: pd.DataFrame):
        """
        Initialize analyzer.
        
        Parameters:
            baseline_embeddings: Original cell embeddings (n_cells × 256)
            perturbation_embeddings: Dict of perturbation_id -> embeddings
            metadata: Perturbation metadata DataFrame
        """
        self.baseline = baseline_embeddings
        self.perturbed = perturbation_embeddings
        self.metadata = metadata
        
        print(f"EmbeddingAnalyzer initialized")
        print(f"  Baseline: {baseline_embeddings.shape[0]:,} cells")
        print(f"  Perturbations: {len(perturbation_embeddings)}")
    
    def compute_centroid_shift(self, pert_id: str) -> float:
        """
        Compute shift of population centroid.
        
        Parameters:
            pert_id: Perturbation ID
            
        Returns:
            Euclidean distance between centroids
        """
        if pert_id not in self.perturbed:
            raise KeyError(f"Perturbation '{pert_id}' not found")
        
        baseline_centroid = np.mean(self.baseline, axis=0)
        perturbed_centroid = np.mean(self.perturbed[pert_id], axis=0)
        
        return euclidean(baseline_centroid, perturbed_centroid)

    def compute_cosine_shift(self, pert_id: str) -> float:
        """
        Compute cosine shift (directional change).
        
        Measures angular change between baseline and perturbed centroids.
        Returns cosine distance: 0 = same direction, 2 = opposite directions.
        
        Parameters:
            pert_id: Perturbation ID
            
        Returns:
            Cosine distance (0-2, lower = more similar direction)
        """
        if pert_id not in self.perturbed:
            raise KeyError(f"Perturbation '{pert_id}' not found")
        
        baseline_centroid = np.mean(self.baseline, axis=0)
        perturbed_centroid = np.mean(self.perturbed[pert_id], axis=0)
        
        # Cosine distance (0 = same direction, 2 = opposite)
        return cosine(baseline_centroid, perturbed_centroid)
    
    def compute_cell_shifts(self, pert_id: str) -> np.ndarray:
        """
        Compute per-cell embedding shifts.
        
        Parameters:
            pert_id: Perturbation ID
            
        Returns:
            Array of shift distances for each cell
        """
        if pert_id not in self.perturbed:
            raise KeyError(f"Perturbation '{pert_id}' not found")
        
        distances = cdist(self.baseline, self.perturbed[pert_id])
        return np.diag(distances)
    
    def compute_neighborhood_preservation(self, 
                                         pert_id: str, 
                                         k: int = 10) -> float:
        """
        Compute neighborhood preservation score.
        
        Measures how well local cell relationships are preserved.
        
        Parameters:
            pert_id: Perturbation ID
            k: Number of nearest neighbors
            
        Returns:
            Preservation score (0-1, higher is better)
        """
        if pert_id not in self.perturbed:
            raise KeyError(f"Perturbation '{pert_id}' not found")
        
        # Find k nearest neighbors in baseline
        baseline_dists = cdist(self.baseline, self.baseline)
        np.fill_diagonal(baseline_dists, np.inf)
        baseline_nn = np.argsort(baseline_dists, axis=1)[:, :k]
        
        # Find k nearest neighbors in perturbed
        perturbed_dists = cdist(self.perturbed[pert_id], self.perturbed[pert_id])
        np.fill_diagonal(perturbed_dists, np.inf)
        perturbed_nn = np.argsort(perturbed_dists, axis=1)[:, :k]
        
        # Compute overlap
        overlaps = []
        for i in range(len(baseline_nn)):
            overlap = len(set(baseline_nn[i]) & set(perturbed_nn[i]))
            overlaps.append(overlap / k)
        
        return np.mean(overlaps)
    
    def analyze_all(self, k_neighbors: int = 10) -> pd.DataFrame:
        """
        Compute all metrics for all perturbations.
        
        Parameters:
            k_neighbors: Number of neighbors for preservation analysis
            
        Returns:
            DataFrame with all metrics
        """
        print(f"Analyzing {len(self.perturbed)} perturbations...")
        
        results = []
        
        for i, pert_id in enumerate(self.perturbed.keys(), 1):
            print(f"  [{i}/{len(self.perturbed)}] {pert_id}")
            
            # Get metadata
            meta_rows = self.metadata[self.metadata['perturbation_id'] == pert_id]
            if len(meta_rows) == 0:
                print(f"    Warning: No metadata for {pert_id}, skipping")
                continue
            
            meta = meta_rows.iloc[0]
            
            try:
                # Compute metrics
                cell_shifts = self.compute_cell_shifts(pert_id)
                
                results.append({
                    'perturbation_id': pert_id,
                    'gene': meta['gene'],
                    'type': meta['type'],
                    'factor': meta.get('factor', None),
                    'centroid_shift': self.compute_centroid_shift(pert_id),
                    'cosine_shift': self.compute_cosine_shift(pert_id),
                    'mean_cell_shift': np.mean(cell_shifts),
                    'median_cell_shift': np.median(cell_shifts),
                    'std_cell_shift': np.std(cell_shifts),
                    'neighborhood_preservation': self.compute_neighborhood_preservation(
                        pert_id, k_neighbors
                    )
                })
                
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        print(f"✓ Analysis complete: {len(results)} perturbations")
        return pd.DataFrame(results)


if __name__ == "__main__":
    # Verify GeneFormer installation
    if GENEFORMER_AVAILABLE:
        print("✓ GeneFormer is installed and ready")
    else:
        print("✗ GeneFormer is NOT installed")
        print("\nTo install GeneFormer:")
        print("  pip install geneformer")
        print("\nDocumentation:")
        print("  https://huggingface.co/ctheodoris/Geneformer")
