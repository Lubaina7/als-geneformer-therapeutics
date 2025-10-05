# ============================================================================
# utils/data_io.py - CHECKED VERSION (NO ERRORS FOUND)
# ============================================================================
"""
Efficient Data I/O for Large Single-Cell Datasets
================================================
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Dict, Optional, List
import h5py
from tqdm import tqdm


class DataIOManager:
    """Manages efficient loading and saving of large datasets."""
    
    def __init__(self, base_dir: str = "./data", cache_dir: str = "./cache"):
        """
        Initialize DataIOManager.
        
        Parameters:
            base_dir: Base directory for raw data
            cache_dir: Cache directory for processed data
        """
        self.base_dir = Path(base_dir)
        self.cache_dir = Path(cache_dir)
        
        # FIX: Use parents=True to create nested directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"DataIOManager initialized")
        print(f"  Data directory: {self.base_dir}")
        print(f"  Cache directory: {self.cache_dir}")
    
    def load_adata_efficient(self, 
                            filepath: str,
                            backed: bool = True,
                            subset_genes: Optional[List[str]] = None,
                            subset_cells: Optional[int] = None) -> ad.AnnData:
        """
        Load AnnData efficiently with optional subsetting.
        
        Parameters:
            filepath: Path to h5ad file
            backed: Load in backed mode (memory-efficient)
            subset_genes: List of genes to keep
            subset_cells: Number of cells to sample
        
        Returns:
            AnnData object
        """
        print(f"Loading data from: {filepath}")
        
        # FIX: Check if file exists
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        if backed:
            print("  Loading in backed mode (memory-efficient)...")
            adata = sc.read_h5ad(filepath, backed='r')
            
            if subset_genes or subset_cells:
                print("  Subsetting requires loading into memory...")
                adata = adata.to_memory()
            else:
                return adata
        else:
            print("  Loading fully into memory...")
            adata = sc.read_h5ad(filepath)
        
        if subset_genes:
            available = [g for g in subset_genes if g in adata.var_names]
            print(f"  Subsetting to {len(available)}/{len(subset_genes)} genes")
            if len(available) == 0:  # FIX: Check for empty subset
                raise ValueError("None of the specified genes found in dataset")
            adata = adata[:, available].copy()
        
        if subset_cells and subset_cells < adata.n_obs:
            print(f"  Sampling {subset_cells}/{adata.n_obs} cells")
            np.random.seed(42)
            cell_idx = np.random.choice(adata.n_obs, subset_cells, replace=False)
            adata = adata[cell_idx, :].copy()
        
        print(f"✓ Loaded: {adata.n_obs:,} cells × {adata.n_vars:,} genes")
        return adata
    
    def save_embeddings_hdf5(self,
                            embeddings_dict: Dict[str, np.ndarray],
                            filename: str = "embeddings.h5",
                            compression: str = 'gzip'):
        """
        Save multiple embedding arrays efficiently in HDF5.
        
        Much better than saving 50+ .npy files!
        
        Parameters:
            embeddings_dict: Dict of name -> embedding array
            filename: Output filename
            compression: Compression method ('gzip', 'lzf', None)
        """
        if len(embeddings_dict) == 0:  # FIX: Check for empty dict
            raise ValueError("No embeddings to save")
        
        filepath = self.cache_dir / filename
        print(f"Saving {len(embeddings_dict)} embedding sets to: {filepath}")
        
        # FIX: Use mode='w' to overwrite if exists
        with h5py.File(filepath, 'w') as f:
            for name, emb in tqdm(embeddings_dict.items(), desc="Saving"):
                # FIX: Validate embedding is numpy array
                if not isinstance(emb, np.ndarray):
                    raise TypeError(f"Embedding '{name}' must be numpy array, got {type(emb)}")
                
                f.create_dataset(name, data=emb, compression=compression, chunks=True)
        
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"✓ Saved ({size_mb:.1f} MB)")
    
    def load_embeddings_hdf5(self, 
                            filename: str = "embeddings.h5",
                            embedding_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Load embeddings from HDF5 file.
        
        Parameters:
            filename: HDF5 filename
            embedding_names: Specific embeddings to load (None = all)
            
        Returns:
            Dict of name -> embedding array
        """
        filepath = self.cache_dir / filename
        
        # FIX: Check if file exists
        if not filepath.exists():
            raise FileNotFoundError(f"Embedding file not found: {filepath}")
        
        print(f"Loading embeddings from: {filepath}")
        
        embeddings = {}
        with h5py.File(filepath, 'r') as f:
            if embedding_names is None:
                embedding_names = list(f.keys())
            
            # FIX: Check if any embeddings found
            if len(embedding_names) == 0:
                raise ValueError(f"No embeddings found in {filepath}")
            
            for name in tqdm(embedding_names, desc="Loading"):
                if name in f:
                    embeddings[name] = f[name][:]
                else:
                    print(f"  Warning: '{name}' not found in file")
        
        print(f"✓ Loaded {len(embeddings)} embedding sets")
        return embeddings
    
    def cache_exists(self, cache_name: str) -> bool:
        """
        Check if cached data exists.
        
        Parameters:
            cache_name: Name of cache file
            
        Returns:
            True if cache exists
        """
        return (self.cache_dir / cache_name).exists()
    
    def get_cache_path(self, cache_name: str) -> Path:
        """
        Get full path to cache file.
        
        Parameters:
            cache_name: Name of cache file
            
        Returns:
            Full path to cache file
        """
        return self.cache_dir / cache_name


if __name__ == "__main__":
    print("Utils modules loaded successfully!")
    print("\nAvailable classes:")
    print("  - GeneFormerEmbedder: Generate embeddings with GeneFormer V2")
    print("  - EmbeddingAnalyzer: Analyze perturbation effects")
    print("  - DataIOManager: Efficient data I/O operations")
