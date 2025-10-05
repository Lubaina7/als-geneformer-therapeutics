"""
Smoke Tests - Quick Validation on Real Data Subset
==================================================

Quick validation that runs in 30 seconds on a tiny real data subset.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
import anndata as ad
from pathlib import Path
from typing import Optional


class SmokeTestRunner:
    """Quick smoke testing on real data subset."""
    
    def __init__(self, full_data_path: str):
        self.full_data_path = Path(full_data_path)
    
    def create_smoke_test_subset(self,
                                 n_cells: int = 50,
                                 n_genes: int = 100,
                                 save_path: Optional[str] = None) -> ad.AnnData:
        """
        Extract tiny subset of REAL data (not synthetic).
        
        Parameters:
            n_cells: Number of cells (default: 50)
            n_genes: Number of genes (default: 100)
            save_path: Where to save (reuses if exists)
        
        Returns:
            Small AnnData subset (~40KB)
        """
        # Reuse cached subset if it exists
        if save_path and Path(save_path).exists():
            return sc.read_h5ad(save_path)
        
        # Load full data in backed mode (memory efficient)
        adata_full = sc.read_h5ad(self.full_data_path, backed='r')
        
        # Sample random cells
        np.random.seed(42)
        n_cells = min(n_cells, adata_full.n_obs)
        cell_idx = np.random.choice(adata_full.n_obs, n_cells, replace=False)
        
        # **FIX: Subset cells FIRST while in backed mode**
        adata_cell_subset = adata_full[cell_idx, :].to_memory()
        
        # Select high variance genes (now working with small matrix)
        if adata_cell_subset.n_vars > n_genes:
            # Handle sparse or dense matrix safely
            if issparse(adata_cell_subset.X):
                X = adata_cell_subset.X.toarray()  # Small matrix, safe to convert
            else:
                X = adata_cell_subset.X
            
            # Compute variance across cells (axis=0)
            var = np.var(X, axis=0)
            gene_idx = np.argsort(var)[-n_genes:]
            
            # Final subset with selected genes
            adata_subset = adata_cell_subset[:, gene_idx].copy()
        else:
            adata_subset = adata_cell_subset.copy()
        
        # Save for reuse
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            adata_subset.write_h5ad(save_path)
        
        return adata_subset
    
    def _test_task1(self, adata: ad.AnnData) -> bool:
        """Test Task 1: Perturbations."""
        try:
            from utils.perturbation import GenePerturbationWorkflow
            
            workflow = GenePerturbationWorkflow(adata)
            gene = adata.var_names[0]
            
            # Test knock-down
            pert_id_kd = workflow.knock_down(gene, 0.5)
            adata_kd = workflow.perturbed_data[pert_id_kd]
            kd_ratio = adata_kd.X[:, 0].mean() / (adata.X[:, 0].mean() + 1e-10)
            if abs(kd_ratio - 0.5) > 0.15:
                return False
            
            # Test knock-up
            pert_id_ku = workflow.knock_up(gene, 2.0, seed=0)
            adata_ku = workflow.perturbed_data[pert_id_ku]
            ku_ratio = adata_ku.X[:, 0].mean() / (adata.X[:, 0].mean() + 1e-10)
            if abs(ku_ratio - 2.0) > 0.4:
                return False
            
            # Check no negative values
            if np.any(adata_kd.X < 0) or np.any(adata_ku.X < 0):
                return False
            
            return True
        except Exception as e:
            print(f"Task 1 failed: {e}")
            return False
    
    def _test_task2(self, adata: ad.AnnData) -> bool:
        """Test Task 2: Embeddings."""
        try:
            from utils.embeddings import GeneFormerEmbedder
            
            embedder = GeneFormerEmbedder()
            embeddings = embedder.embed_adata(adata)
            
            # Check shape
            if embeddings.shape[0] != adata.n_obs:
                return False
            
            # Check no NaN/Inf
            if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
                return False
            
            return True
        except Exception as e:
            print(f"Task 2 failed: {e}")
            return False
    
    def run_all_smoke_tests(self, 
                           n_cells: int = 50, 
                           n_genes: int = 100,
                           verbose: bool = False) -> bool:
        """
        Run all smoke tests.
        
        Parameters:
            n_cells: Number of cells for test
            n_genes: Number of genes for test
            verbose: Print detailed output
        
        Returns:
            True if all tests pass
        """
        # Create/load subset
        cache_path = "cache/smoke_test_subset.h5ad"
        adata = self.create_smoke_test_subset(n_cells, n_genes, cache_path)
        
        if verbose:
            print(f"Running smoke tests on {adata.n_obs} cells × {adata.n_vars} genes...")
        
        # Run tests
        results = {
            'Task 1 (Perturbations)': self._test_task1(adata),
            'Task 2 (Embeddings)': self._test_task2(adata)
        }
        
        # Report
        all_passed = all(results.values())
        
        if verbose or not all_passed:
            for task, passed in results.items():
                status = "✓" if passed else "✗"
                print(f"{status} {task}")
        
        if all_passed:
            if verbose:
                print("\n✓ ALL SMOKE TESTS PASSED")
        else:
            print("\n✗ SOME SMOKE TESTS FAILED")
        
        return all_passed


def quick_smoke_test(data_path: str, verbose: bool = False) -> bool:
    """
    One-liner smoke test.
    
    Usage:
        from utils.smoke_tests import quick_smoke_test
        quick_smoke_test("data/als_data.h5ad")
    
    Parameters:
        data_path: Path to full dataset
        verbose: Print detailed output
    
    Returns:
        True if all tests pass
    """
    runner = SmokeTestRunner(data_path)
    return runner.run_all_smoke_tests(verbose=verbose)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        verbose = "--verbose" in sys.argv or "-v" in sys.argv
        passed = quick_smoke_test(data_path, verbose=verbose)
        sys.exit(0 if passed else 1)
    else:
        print("Usage: python utils/smoke_tests.py <data_path> [--verbose]")
        print("Example: python utils/smoke_tests.py data/als_data.h5ad")
        sys.exit(1)
