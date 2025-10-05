# ============================================================================
# utils/perturbation.py - COMPLETE VERSION
# ============================================================================
"""
Gene Perturbation Simulation
============================

Simulate gene knock-down and knock-up experiments with biological realism.

Key features:
- Multiplicative perturbations (simple, biologically interpretable)
- Cell-type specific targeting
- Sparse matrix optimization (memory efficient)
- Log2 fold-change tracking
- Reproducible with seeding
"""

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from typing import List, Dict, Union, Optional
import warnings


class GenePerturbationWorkflow:
    """
    Simulate gene perturbations on single-cell data.
    
    This class provides multiplicative knock-down (reduce expression) and 
    knock-up (increase expression) with optional cell-type specificity.
    
    All perturbations are tracked with metadata for reproducibility.
    """
    
    def __init__(self, adata: ad.AnnData, copy: bool = True):
        """
        Initialize perturbation workflow.
        
        Parameters:
            adata: AnnData object with single-cell expression data
            copy: If True, create a copy to protect original data
        """
        # Always copy to protect original data
        self.adata = adata.copy() if copy else adata
        
        # Storage for perturbed datasets and logs
        self.perturbed_data = {}
        self.perturbation_log = []
        
        # Validate input (catches empty/malformed data early)
        self._validate_input_data()
        
        print(f"✓ GenePerturbationWorkflow initialized")
        print(f"  Cells: {self.adata.n_obs:,}")
        print(f"  Genes: {self.adata.n_vars:,}")
    
    def _validate_input_data(self):
        """
        Validate that input data is suitable for perturbations.
        
        NOTE: Even with data_io validation, this catches edge cases like:
        - Empty AnnData objects
        - Wrong data types
        - Extreme values that suggest preprocessing errors
        
        This is fast and prevents cryptic errors later.
        """
        # Check not empty
        if self.adata.n_obs == 0 or self.adata.n_vars == 0:
            raise ValueError("AnnData object is empty")
        
        # Check data type
        if not isinstance(self.adata.X, (np.ndarray, sparse.spmatrix)):
            raise TypeError(
                f"Expression matrix must be numpy array or sparse matrix, "
                f"got {type(self.adata.X)}"
            )
        
        # Check for reasonable values (catches preprocessing errors)
        if sparse.issparse(self.adata.X):
            max_val = self.adata.X.max()
        else:
            max_val = np.max(self.adata.X)
        
        if max_val > 1e6:
            warnings.warn(
                f"Very large expression values detected (max={max_val}). "
                "Consider normalizing data first."
            )
    
    def _get_gene_indices(self, genes: Union[str, List[str]]) -> np.ndarray:
        """
        Convert gene names to column indices in expression matrix.
        
        Parameters:
            genes: Single gene name or list of gene names
        
        Returns:
            Array of gene indices
        """
        if isinstance(genes, str):
            genes = [genes]
        
        gene_indices = []
        missing_genes = []
        
        for gene in genes:
            if gene in self.adata.var_names:
                idx = np.where(self.adata.var_names == gene)[0][0]
                gene_indices.append(idx)
            else:
                missing_genes.append(gene)
        
        # Report missing genes
        if missing_genes:
            print(f"⚠ Warning: {len(missing_genes)} genes not found")
            if len(missing_genes) <= 5:
                print(f"  Missing: {missing_genes}")
        
        # Must have at least one valid gene
        if len(gene_indices) == 0:
            raise ValueError("No valid genes found in dataset")
        
        return np.array(gene_indices)
    
    def knock_down(self,
                   genes: Union[str, List[str]],
                   reduction_factor: float = 0.5,
                   cell_subset: Optional[np.ndarray] = None,
                   perturbation_id: Optional[str] = None,
                   seed: Optional[int] = None) -> str:
        """
        Simulate gene knock-down (reduce expression).
        
        Uses multiplicative scaling: new_expr = old_expr × reduction_factor
        
        Parameters:
            genes: Gene(s) to knock down
            reduction_factor: Fraction of expression to KEEP (0-1)
                            0.3 = keep 30% (70% reduction, -1.74 log2FC)
                            0.5 = keep 50% (50% reduction, -1.0 log2FC)
                            0.7 = keep 70% (30% reduction, -0.51 log2FC)
            cell_subset: Boolean mask for specific cells (None = all cells)
            perturbation_id: Custom identifier (auto-generated if None)
            seed: Random seed for reproducibility (unused for knock-down, 
                  but kept for API consistency)
        
        Returns:
            Perturbation ID (use to retrieve: workflow.perturbed_data[id])
        
        Example:
            # Reduce SOD1 by 70% in diseased cells
            diseased = adata.obs['disease'] == 'ALS'
            pert_id = workflow.knock_down('SOD1', reduction_factor=0.3, 
                                         cell_subset=diseased)
            adata_kd = workflow.perturbed_data[pert_id]
        """
        # Validate parameters
        if not 0 <= reduction_factor <= 1:
            raise ValueError(
                f"reduction_factor must be 0-1 (fraction to keep), "
                f"got {reduction_factor}"
            )
        
        # Get gene indices
        gene_indices = self._get_gene_indices(genes)
        gene_list = genes if isinstance(genes, list) else [genes]
        
        # Create perturbed copy
        adata_perturbed = self.adata.copy()
        
        # Determine which cells to perturb
        if cell_subset is None:
            cell_subset = np.ones(adata_perturbed.n_obs, dtype=bool)
        n_perturbed_cells = np.sum(cell_subset)
        
        # Apply perturbation (memory-efficient for sparse)
        if sparse.issparse(adata_perturbed.X):
            # Operate on sparse matrix directly (memory efficient)
            X = adata_perturbed.X
            for gene_idx in gene_indices:
                # Extract column (remains sparse)
                gene_col = X[:, gene_idx].toarray().flatten().astype(float)
                # Scale down
                gene_col[cell_subset] *= reduction_factor
                # Put back
                X[:, gene_idx] = sparse.csr_matrix(gene_col.reshape(-1, 1))
        else:
            # Dense matrix
            X = adata_perturbed.X.astype(float)
            for gene_idx in gene_indices:
                X[cell_subset, gene_idx] *= reduction_factor
        
        # Calculate log2 fold change (biological standard)
        log2_effect = np.log2(reduction_factor) if reduction_factor > 0 else -np.inf
        
        # Create perturbation ID
        if perturbation_id is None:
            gene_str = '_'.join(gene_list)
            perturbation_id = f"KD_{gene_str}_{reduction_factor:.2f}"
        
        # Store comprehensive metadata
        adata_perturbed.uns['perturbation'] = {
            'type': 'knock_down',
            'genes': gene_list,
            'reduction_factor': reduction_factor,
            'log2_effect': log2_effect,  # Biological standard
            'method': 'multiplicative',
            'n_cells_perturbed': int(n_perturbed_cells),
            'n_cells_total': adata_perturbed.n_obs,
            'perturbation_id': perturbation_id,
        }
        
        # Log for summary
        self.perturbation_log.append({
            'id': perturbation_id,
            'type': 'knock_down',
            'genes': gene_list,
            'factor': reduction_factor,
            'log2_effect': log2_effect,
            'n_cells': int(n_perturbed_cells)
        })
        
        # Store perturbed dataset
        self.perturbed_data[perturbation_id] = adata_perturbed
        
        # Print summary
        print(f"✓ Knock-down: {len(gene_indices)} gene(s), "
              f"{n_perturbed_cells:,} cells")
        print(f"  Reduction: {(1-reduction_factor)*100:.0f}% "
              f"(log2FC = {log2_effect:.2f})")
        
        # Return ID for easy retrieval
        return perturbation_id
    
    def knock_up(self,
                 genes: Union[str, List[str]],
                 amplification_factor: float = 2.0,
                 cell_subset: Optional[np.ndarray] = None,
                 noise_level: float = 0.1,
                 min_expr: float = 0.1,
                 perturbation_id: Optional[str] = None,
                 seed: Optional[int] = None) -> str:
        """
        Simulate gene knock-up (increase expression).
        
        Uses multiplicative scaling with optional noise:
        new_expr = old_expr × amplification_factor × (1 + noise)
        
        Only amplifies genes with expression > min_expr to avoid unrealistic
        amplification of truly unexpressed genes.
        
        Parameters:
            genes: Gene(s) to amplify
            amplification_factor: Multiplication factor (>1)
                                1.5 = 50% increase (+0.58 log2FC)
                                2.0 = doubling (+1.0 log2FC)
                                3.0 = tripling (+1.58 log2FC)
            cell_subset: Boolean mask for specific cells (None = all cells)
            noise_level: Gaussian noise std (0-0.2 recommended)
                        0.1 = 10% cell-to-cell variability (biological realism)
            min_expr: Minimum expression to amplify (avoids amplifying zeros)
                     Default 0.1 = only amplify if baseline > 0.1
            perturbation_id: Custom identifier (auto-generated if None)
            seed: Random seed for reproducibility (use seed=0 in notebook)
        
        Returns:
            Perturbation ID (use to retrieve: workflow.perturbed_data[id])
        
        Example:
            # Double C9orf72 in healthy cells with 10% noise
            healthy = adata.obs['disease'] == 'Control'
            pert_id = workflow.knock_up('C9orf72', amplification_factor=2.0,
                                       cell_subset=healthy, seed=0)
            adata_ku = workflow.perturbed_data[pert_id]
        """
        # Validate parameters
        if amplification_factor < 1:
            raise ValueError(
                f"amplification_factor must be >=1 (fold increase), "
                f"got {amplification_factor}"
            )
        
        if not 0 <= noise_level <= 1:
            raise ValueError(
                f"noise_level must be 0-1, got {noise_level}"
            )
        
        # Set seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Get gene indices
        gene_indices = self._get_gene_indices(genes)
        gene_list = genes if isinstance(genes, list) else [genes]
        
        # Create perturbed copy
        adata_perturbed = self.adata.copy()
        
        # Determine which cells to perturb
        if cell_subset is None:
            cell_subset = np.ones(adata_perturbed.n_obs, dtype=bool)
        n_perturbed_cells = np.sum(cell_subset)
        
        # Apply perturbation (memory-efficient for sparse)
        if sparse.issparse(adata_perturbed.X):
            # Operate on sparse matrix directly
            X = adata_perturbed.X
            for gene_idx in gene_indices:
                # Extract column
                gene_col = X[:, gene_idx].toarray().flatten().astype(float)
                
                # Only amplify where expression > min_expr
                amplify_mask = cell_subset & (gene_col > min_expr)
                
                if np.any(amplify_mask):
                    # Add biological noise
                    noise = np.random.normal(1.0, noise_level, 
                                           size=np.sum(amplify_mask))
                    # Scale up with noise
                    gene_col[amplify_mask] *= amplification_factor * noise
                
                # Put back
                X[:, gene_idx] = sparse.csr_matrix(gene_col.reshape(-1, 1))
        else:
            # Dense matrix
            X = adata_perturbed.X.astype(float)
            for gene_idx in gene_indices:
                gene_col = X[:, gene_idx]
                
                # Only amplify where expression > min_expr
                amplify_mask = cell_subset & (gene_col > min_expr)
                
                if np.any(amplify_mask):
                    # Add biological noise
                    noise = np.random.normal(1.0, noise_level,
                                           size=np.sum(amplify_mask))
                    # Scale up with noise
                    X[amplify_mask, gene_idx] *= amplification_factor * noise
        
        # Calculate log2 fold change
        log2_effect = np.log2(amplification_factor)
        
        # Create perturbation ID
        if perturbation_id is None:
            gene_str = '_'.join(gene_list)
            perturbation_id = f"KU_{gene_str}_{amplification_factor:.1f}"
        
        # Store metadata
        adata_perturbed.uns['perturbation'] = {
            'type': 'knock_up',
            'genes': gene_list,
            'amplification_factor': amplification_factor,
            'log2_effect': log2_effect,  # Biological standard
            'method': 'multiplicative',
            'noise_level': noise_level,
            'min_expr': min_expr,
            'n_cells_perturbed': int(n_perturbed_cells),
            'n_cells_total': adata_perturbed.n_obs,
            'perturbation_id': perturbation_id,
            'seed': seed,
        }
        
        # Log
        self.perturbation_log.append({
            'id': perturbation_id,
            'type': 'knock_up',
            'genes': gene_list,
            'factor': amplification_factor,
            'log2_effect': log2_effect,
            'n_cells': int(n_perturbed_cells)
        })
        
        # Store
        self.perturbed_data[perturbation_id] = adata_perturbed
        
        # Print summary
        print(f"✓ Knock-up: {len(gene_indices)} gene(s), "
              f"{n_perturbed_cells:,} cells")
        print(f"  Amplification: {amplification_factor:.1f}x "
              f"(log2FC = {log2_effect:.2f})")
        
        # Return ID
        return perturbation_id

    def clear_perturbation(self, perturbation_id: str):
        """Remove perturbation from memory to free RAM."""
        if perturbation_id in self.perturbed_data:
            del self.perturbed_data[perturbation_id]
            import gc
            gc.collect()
    
    def get_perturbation_summary(self) -> pd.DataFrame:
        """
        Get summary DataFrame of all perturbations.
        
        Returns:
            DataFrame with perturbation details
        """
        if not self.perturbation_log:
            return pd.DataFrame()
        return pd.DataFrame(self.perturbation_log)
    
    def get_perturbation(self, perturbation_id: str) -> ad.AnnData:
        """
        Retrieve a specific perturbed dataset.
        
        Parameters:
            perturbation_id: ID returned from knock_down/knock_up
        
        Returns:
            Perturbed AnnData object
        """
        if perturbation_id not in self.perturbed_data:
            available = list(self.perturbed_data.keys())
            raise KeyError(
                f"Perturbation '{perturbation_id}' not found. "
                f"Available: {available}"
            )
        return self.perturbed_data[perturbation_id]


class PerturbationValidator:
    """
    Validate perturbation quality and biological plausibility.
    
    Use these two methods in Notebook 1 for quality control.
    """
    
    def validate_rank_shifts(
        self,
        adata_original: ad.AnnData,
        adata_perturbed: ad.AnnData,
        gene: str,
        threshold: int = 100
    ) -> Dict[str, any]:
        """
        Check if perturbation actually changed gene ranks (critical for GeneFormer).
    
        GeneFormer uses rank-value encoding - perturbations must shift ranks
        to be visible to the model.
        
        Parameters:
            adata_original: Original dataset
            adata_perturbed: Perturbed dataset
            gene: Gene that was perturbed
            threshold: Minimum mean rank shift to consider "effective"
                      (100 = moved 100 positions out of ~20k genes)
    
        Returns:
            Dict with rank-shift metrics and warning if ineffective
        """
        from scipy.sparse import issparse
    
        # Get gene index
        gene_idx = np.where(adata_original.var_names == gene)[0][0]
    
        # Extract expression
        X_orig = adata_original.X.toarray() if issparse(adata_original.X) else adata_original.X
        X_pert = adata_perturbed.X.toarray() if issparse(adata_perturbed.X) else adata_perturbed.X
    
        # Compute rank shifts
        rank_shifts = []
        for i in range(len(X_orig)):
            # Rank by expression (0 = highest)
            orig_sorted = np.argsort(-X_orig[i])
            pert_sorted = np.argsort(-X_pert[i])
        
            orig_rank = np.where(orig_sorted == gene_idx)[0][0]
            pert_rank = np.where(pert_sorted == gene_idx)[0][0]
        
            rank_shifts.append(orig_rank - pert_rank)  # Positive = moved up
    
        rank_shifts = np.array(rank_shifts)
    
        # Calculate metrics
        mean_shift = rank_shifts.mean()
        median_shift = np.median(rank_shifts)
        pct_changed = 100 * (rank_shifts != 0).sum() / len(rank_shifts)
        mean_abs_shift = np.abs(rank_shifts).mean()
    
        # Determine if effective
        is_effective = mean_abs_shift >= threshold
    
        return {
            'gene': gene,
            'mean_rank_shift': float(mean_shift),
            'median_rank_shift': float(median_shift),
            'mean_abs_rank_shift': float(mean_abs_shift),
            'pct_cells_changed_rank': float(pct_changed),
            'is_effective_for_geneformer': is_effective,
            'warning': None if is_effective else 
                      f"Low rank shift ({mean_abs_shift:.0f} positions) - "
                      f"perturbation may be invisible to GeneFormer. "
                      f"Consider stronger factors (10x up, 0.1x down) or percentile-based."
        }
    
    def validate_perturbation_strength(
        self,
        adata_original: ad.AnnData,
        adata_perturbed: ad.AnnData,
        gene: str
    ) -> Dict[str, float]:
        """
        Validate that perturbation achieved expected fold-change.
        
        This is your primary QC metric - use in Notebook 1.
        
        Parameters:
            adata_original: Original dataset
            adata_perturbed: Perturbed dataset
            gene: Gene that was perturbed
        
        Returns:
            Dict with per-gene fold-change metrics
        
        Example:
            metrics = validator.validate_perturbation_strength(
                adata, adata_perturbed, 'SOD1'
            )
            print(f"Fold change: {metrics['fold_change']:.2f}")
            print(f"Log2 FC: {metrics['log2_fold_change']:.2f}")
        """
        # Find gene index
        gene_idx = np.where(adata_original.var_names == gene)[0][0]
        
        # Extract expression (handle sparse)
        if sparse.issparse(adata_original.X):
            orig_expr = adata_original.X[:, gene_idx].toarray().flatten()
            pert_expr = adata_perturbed.X[:, gene_idx].toarray().flatten()
        else:
            orig_expr = adata_original.X[:, gene_idx]
            pert_expr = adata_perturbed.X[:, gene_idx]
        
        # Calculate metrics
        orig_mean = np.mean(orig_expr)
        pert_mean = np.mean(pert_expr)
        fold_change = pert_mean / (orig_mean + 1e-10)
        log2_fc = np.log2(fold_change) if fold_change > 0 else -np.inf
        
        return {
            'gene': gene,
            'original_mean': float(orig_mean),
            'perturbed_mean': float(pert_mean),
            'fold_change': float(fold_change),
            'log2_fold_change': float(log2_fc),
            'cells_affected': int(np.sum(orig_expr != pert_expr)),
            'correlation': float(np.corrcoef(orig_expr, pert_expr)[0, 1])
        }
    
    def check_biological_plausibility(
        self,
        adata_perturbed: ad.AnnData
    ) -> Dict[str, bool]:
        """
        Check that perturbed data maintains biological plausibility.
        
        This is your plausibility checklist - use in Notebook 1.
        
        Parameters:
            adata_perturbed: Perturbed dataset
        
        Returns:
            Dict with plausibility checks (True = passed)
        
        Example:
            checks = validator.check_biological_plausibility(adata_perturbed)
            for check, passed in checks.items():
                status = "✓" if passed else "✗"
                print(f"{status} {check}")
        """
        # Extract data (handle sparse)
        if sparse.issparse(adata_perturbed.X):
            X = adata_perturbed.X
            # For sparse, check differently
            checks = {
                'no_negative_values': X.min() >= 0,
                'no_nan_values': not np.isnan(X.data).any(),
                'no_inf_values': not np.isinf(X.data).any(),
                'maintains_sparsity': (X.nnz / X.size) < 0.7,  # <70% non-zero
                'realistic_max': X.max() < 1e6,
                'realistic_total_counts': X.sum(axis=1).min() > 10
            }
        else:
            X = adata_perturbed.X
            checks = {
                'no_negative_values': np.all(X >= 0),
                'no_nan_values': not np.any(np.isnan(X)),
                'no_inf_values': not np.any(np.isinf(X)),
                'maintains_sparsity': (np.sum(X == 0) / X.size) > 0.3,
                'realistic_max': np.max(X) < 1e6,
                'realistic_total_counts': np.all(X.sum(axis=1) > 10)
            }
        
        return checks


if __name__ == "__main__":
    print("Perturbation utilities loaded successfully!")
    print("\nKey classes:")
    print("  - GenePerturbationWorkflow: Simulate perturbations")
    print("  - PerturbationValidator: Quality control")
    print("\nRecommended usage:")
    print("  1. Set seed in notebook: np.random.seed(0)")
    print("  2. Use multiplicative method only")
    print("  3. Validate with fold-change + plausibility checks")
