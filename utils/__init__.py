# ============================================================================
# utils/__init__.py - UPDATED
# ============================================================================
"""
Gene Perturbation Analysis Utilities
====================================

Complete package for in-silico gene perturbation analysis using GeneFormer V2.

Main modules:
    - perturbation: GenePerturbationWorkflow, PerturbationValidator
    - embeddings: GeneFormerEmbedder, EmbeddingAnalyzer  
    - data_io: DataIOManager
    - smoke_tests: SmokeTestRunner, quick_smoke_test
    - visualization: Plotting utilities
"""

__version__ = "1.0.0"
__author__ = "Gene Perturbation Analysis Team"

# Core imports (always available)
from utils.perturbation import GenePerturbationWorkflow, PerturbationValidator
from utils.data_io import DataIOManager
from utils.smoke_tests import SmokeTestRunner, quick_smoke_test

# Optional imports (require GeneFormer)
try:
    from utils.embeddings import GeneFormerEmbedder, EmbeddingAnalyzer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Note: GeneFormer embeddings not available. Install with: pip install geneformer")

# Visualization (always available)
try:
    from utils import visualization
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Note: Visualization utilities require matplotlib, seaborn, umap-learn")

__all__ = [
    # Core classes
    'GenePerturbationWorkflow',
    'PerturbationValidator', 
    'DataIOManager',
    'SmokeTestRunner',
    'quick_smoke_test',
    
    # Embeddings
    'GeneFormerEmbedder',
    'EmbeddingAnalyzer',
    
    # Modules
    'visualization',
    
    # Flags
    'EMBEDDINGS_AVAILABLE',
    'VISUALIZATION_AVAILABLE',
]
