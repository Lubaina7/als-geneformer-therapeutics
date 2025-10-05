# ============================================================================
# utils/visualization.py - UPDATED with heatmap
# ============================================================================
"""
Visualization Utilities
======================

Plotting functions for perturbation analysis and embedding visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
from scipy import sparse

# Optional imports
try:
    from sklearn.decomposition import PCA
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def plot_perturbation_effect(adata_original, 
                             adata_perturbed, 
                             gene_name: str,
                             figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
    """
    Plot before/after perturbation for a gene.
    
    Shows histogram and scatter plot comparing original vs perturbed expression.
    
    Parameters:
        adata_original: Original AnnData
        adata_perturbed: Perturbed AnnData
        gene_name: Gene to visualize
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Get gene index
    gene_idx = np.where(adata_original.var_names == gene_name)[0][0]
    
    # Extract expression (handle sparse)
    if sparse.issparse(adata_original.X):
        orig = adata_original.X[:, gene_idx].toarray().flatten()
        pert = adata_perturbed.X[:, gene_idx].toarray().flatten()
    else:
        orig = adata_original.X[:, gene_idx]
        pert = adata_perturbed.X[:, gene_idx]

    # Get perturbation info from metadata
    pert_meta = adata_perturbed.uns['perturbation']  # ← YOU'RE MISSING THIS LINE
    pert_type = pert_meta.get('type', 'unknown')
    log2fc = pert_meta.get('log2_effect', None)

    # Get factor (check multiple possible key names)
    factor = next((pert_meta.get(k) for k in ['factor', 'amplification_factor', 'reduction_factor'] if k in pert_meta), None)

    # Generate description
    try:
        if factor and pert_type == 'knock_down':
            pert_desc = f'Knockdown to {float(factor)*100:.0f}%'
        elif factor and pert_type == 'knock_up':
            pert_desc = f'Knockup by {float(factor):.1f}x'
        else:
            pert_desc = pert_type
    except (ValueError, TypeError):
        pert_desc = pert_type
    
    
    # LEFT: Histogram
    ax = axes[0]
    ax.hist(orig, bins=30, alpha=0.6, label='Before', color='#3498db', edgecolor='black')
    ax.hist(pert, bins=30, alpha=0.6, label='After', color='#e74c3c', edgecolor='black')
    ax.set_xlabel(f'{gene_name} Expression (counts)', fontsize=11)
    ax.set_ylabel('Number of Cells', fontsize=11)
    ax.legend(fontsize=10)
    
    title = f'{gene_name} Expression: Before vs After\n{pert_desc}'
    if log2fc is not None:
        title += f' (log2FC = {log2fc:.2f})'
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # RIGHT: Scatter
    ax = axes[1]
    ax.scatter(orig, pert, alpha=0.5, s=30, edgecolors='black', linewidth=0.5)
    
    # Identity line
    max_val = max(orig.max(), pert.max())
    ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2, 
           label='No change (identity)')
    
    ax.set_xlabel('Original Expression (counts)', fontsize=11)
    ax.set_ylabel('Perturbed Expression (counts)', fontsize=11)
    ax.set_title(f'Per-Cell Effect\n(Points below line = reduced expression)', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Add equal aspect ratio for better visualization
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig

def plot_embedding_shift(baseline_emb, 
                         perturbed_emb, 
                         title: str = "Embedding Shift",
                         method: str = 'umap',
                         figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
    """
    Visualize embedding space shift.
    
    Parameters:
        baseline_emb: Baseline embeddings (n_cells × dim)
        perturbed_emb: Perturbed embeddings (n_cells × dim)
        title: Plot title
        method: 'umap' or 'pca'
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Reduce to 2D
    if method == 'umap':
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        
        reducer = umap.UMAP(n_components=2, random_state=42)
        baseline_2d = reducer.fit_transform(baseline_emb)
        perturbed_2d = reducer.transform(perturbed_emb)
        xlabel, ylabel = 'UMAP 1', 'UMAP 2'
    else:  # PCA
        pca = PCA(n_components=2)
        baseline_2d = pca.fit_transform(baseline_emb)
        perturbed_2d = pca.transform(perturbed_emb)
        xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.1%})'
        ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'
    
    # Plot baseline
    axes[0].scatter(baseline_2d[:, 0], baseline_2d[:, 1], 
                   c='lightblue', s=20, alpha=0.5)
    axes[0].set_title('Baseline State')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].grid(alpha=0.3)
    
    # Plot overlay
    axes[1].scatter(baseline_2d[:, 0], baseline_2d[:, 1], 
                   c='lightgray', s=20, alpha=0.3, label='Baseline')
    axes[1].scatter(perturbed_2d[:, 0], perturbed_2d[:, 1],
                   c='red', s=20, alpha=0.5, label='Perturbed')
    axes[1].set_title(title)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_shift_vectors(baseline_emb,
                       perturbed_emb,
                       n_sample: int = 50,
                       method: str = 'umap',
                       title: str = "Perturbation Shift Vectors",
                       figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot shift vectors showing cell movement in embedding space.
    
    Parameters:
        baseline_emb: Baseline embeddings
        perturbed_emb: Perturbed embeddings
        n_sample: Number of cells to show vectors for
        method: 'umap' or 'pca'
        title: Plot title
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Reduce to 2D
    if method == 'umap':
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        reducer = umap.UMAP(n_components=2, random_state=42)
        baseline_2d = reducer.fit_transform(baseline_emb)
        perturbed_2d = reducer.transform(perturbed_emb)
        xlabel, ylabel = 'UMAP 1', 'UMAP 2'
    else:
        pca = PCA(n_components=2)
        baseline_2d = pca.fit_transform(baseline_emb)
        perturbed_2d = pca.transform(perturbed_emb)
        xlabel, ylabel = 'PC 1', 'PC 2'
    
    # Sample cells
    n_cells = len(baseline_2d)
    sample_idx = np.random.choice(n_cells, min(n_sample, n_cells), replace=False)
    
    # Plot all cells (gray)
    ax.scatter(baseline_2d[:, 0], baseline_2d[:, 1],
              c='lightgray', s=10, alpha=0.3, label='All cells')
    
    # Plot sampled start points
    ax.scatter(baseline_2d[sample_idx, 0], baseline_2d[sample_idx, 1],
              c='blue', s=40, alpha=0.6, marker='o', 
              edgecolors='black', linewidth=0.5, label='Baseline')
    
    # Draw vectors
    for idx in sample_idx:
        ax.arrow(baseline_2d[idx, 0], baseline_2d[idx, 1],
                perturbed_2d[idx, 0] - baseline_2d[idx, 0],
                perturbed_2d[idx, 1] - baseline_2d[idx, 1],
                head_width=0.2, head_length=0.2, 
                fc='red', ec='red', alpha=0.4)
    
    # Plot endpoints
    ax.scatter(perturbed_2d[sample_idx, 0], perturbed_2d[sample_idx, 1],
              c='red', s=40, alpha=0.6, marker='s',
              edgecolors='black', linewidth=0.5, label='Perturbed')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_metrics_heatmap(results_df: pd.DataFrame,
                         metrics: Optional[List[str]] = None,
                         figsize: Tuple[int, int] = (12, 8),
                         cmap: str = 'RdYlGn') -> plt.Figure:
    """
    Plot heatmap of metrics across all perturbations.
    
    Parameters:
        results_df: DataFrame with perturbation metrics
        metrics: List of metric columns to include (None = auto-select)
        figsize: Figure size
        cmap: Colormap (RdYlGn = red (low) to green (high))
    
    Returns:
        Matplotlib figure
    """
    # Auto-select metrics if not provided
    if metrics is None:
        metrics = [
            'centroid_shift',
            'mean_cell_shift',
            'neighborhood_preservation',
            'pct_changed_cluster'
        ]
    
    # Filter to available metrics
    available_metrics = [m for m in metrics if m in results_df.columns]
    
    if len(available_metrics) == 0:
        raise ValueError(f"No valid metrics found. Available: {results_df.columns.tolist()}")
    
    # Prepare data - IMPROVED SORTING
    df = results_df.copy()
    
    # Parse perturbation components for better sorting
    df['gene'] = df['perturbation_id'].str.extract(r'(?:KD|KU)_([A-Z0-9]+)_')[0].fillna('Unknown')
    df['type_label'] = df['perturbation_id'].str.extract(r'^(KD|KU)_')[0].fillna('KD')
    df['factor'] = df['perturbation_id'].str.extract(r'_(\d+\.?\d*)_')[0]
    df['factor'] = pd.to_numeric(df['factor'], errors='coerce').fillna(0.0)
    
    # Sort by gene → type → factor (groups related perturbations together)
    df = df.sort_values(['gene', 'type_label', 'factor'])
    
    # Create better display labels with KD/KU prefix
    df['display_label'] = (df['type_label'] + '_' + df['gene'] + '_' + 
                       df['factor'].astype(str))
    
    # Set index for heatmap
    data = df.set_index('display_label')[available_metrics]
    
    # Normalize each metric to 0-1 for better color scale
    data_norm = (data - data.min()) / (data.max() - data.min())
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap with IMPROVED colorbar
    sns.heatmap(data_norm, 
                annot=data.values,  # Show actual values
                fmt='.2f', 
                cmap=cmap, 
                ax=ax, 
                cbar_kws={
                    'label': 'Normalized Score\n(Red = Low, Green = High)',
                    'shrink': 0.8
                })
    
    # Improve labels
    ax.set_xlabel('Metrics', fontweight='bold', fontsize=11)
    ax.set_ylabel('Perturbations (Gene_Type)', fontweight='bold', fontsize=11)
    ax.set_title('Perturbation Metrics Heatmap\n(Sorted by Gene → Type → Factor)', 
                 fontweight='bold', fontsize=13, pad=15)
    
    # Rotate x-axis labels for readability
    ax.set_xticklabels([m.replace('_', '\n') for m in available_metrics], 
                       rotation=45, ha='right', fontsize=10)
    
    # Keep y-axis labels horizontal
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_metric_comparison(analysis_df: pd.DataFrame,
                           metric: str = 'centroid_shift',
                           group_by: str = 'gene',
                           figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Compare metrics across perturbations.
    
    Parameters:
        analysis_df: DataFrame from EmbeddingAnalyzer.analyze_all()
        metric: Metric to visualize
        group_by: How to group ('gene' or 'type')
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Group by gene and type
    pivot = analysis_df.pivot_table(
        values=metric,
        index=group_by,
        columns='type',
        aggfunc='mean'
    )
    
    pivot.plot(kind='bar', ax=ax, rot=45)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_xlabel(group_by.title())
    ax.set_title(f'{metric.replace("_", " ").title()} by {group_by.title()} and Type')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(title='Perturbation Type')
    
    plt.tight_layout()
    return fig


def plot_log2fc_summary(workflow, validation_df=None, figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
    """
    Plot log2 fold-change summary for all perturbations.
    Shows actual vs expected values, grouped by perturbation type.
    
    Parameters:
        workflow: GenePerturbationWorkflow instance
        validation_df: DataFrame with actual_log2fc values (optional)
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    # Get summary
    summary = workflow.get_perturbation_summary()
    
    if len(summary) == 0:
        raise ValueError("No perturbations found")
    
    # Merge with validation data if provided
    if validation_df is not None:
        summary = summary.merge(
            validation_df[['perturbation_id', 'actual_log2fc']], 
            left_on='id', 
            right_on='perturbation_id', 
            how='left'
        )
    
    # Separate knockdowns and knockups
    kd_data = summary[summary['type'] == 'knock_down'].copy()
    ku_data = summary[summary['type'] == 'knock_up'].copy()
    
    # Sort each group by factor
    kd_data = kd_data.sort_values('factor')
    ku_data = ku_data.sort_values('factor')
    
    # Combine: knockdowns first, then knockups
    plot_data = pd.concat([kd_data, ku_data], ignore_index=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(plot_data))
    
    # Determine what to plot
    if 'actual_log2fc' in plot_data.columns and plot_data['actual_log2fc'].notna().any():
        # Plot ACTUAL values as bars
        bar_values = plot_data['actual_log2fc'].values
        expected_values = plot_data['log2_effect'].values
        show_expected = True
    else:
        # No validation data - just show expected
        bar_values = plot_data['log2_effect'].values
        show_expected = False
    
    # Color bars
    colors = ['#e74c3c' if t == 'knock_down' else '#3498db' 
              for t in plot_data['type']]
    
    bars = ax.bar(x_pos, bar_values, color=colors, alpha=0.7, 
                  edgecolor='black', linewidth=1, label='Actual log2FC')
    
    # Overlay expected values as markers
    if show_expected:
        ax.scatter(x_pos, expected_values, color='gold', s=100, 
                  marker='D', edgecolors='black', linewidth=1.5, 
                  label='Expected (Target)', zorder=5)
        
        # Connect expected values with a line for each group
        kd_count = len(kd_data)
        if kd_count > 0:
            ax.plot(range(kd_count), expected_values[:kd_count], 
                   'k--', alpha=0.3, linewidth=1)
        if len(ku_data) > 0:
            ax.plot(range(kd_count, len(plot_data)), expected_values[kd_count:], 
                   'k--', alpha=0.3, linewidth=1)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1.5)
    
    # Add separator between knockdowns and knockups
    if len(kd_data) > 0 and len(ku_data) > 0:
        separator_pos = len(kd_data) - 0.5
        ax.axvline(x=separator_pos, color='gray', linestyle='--', 
                  alpha=0.5, linewidth=2)
        
        # Add labels for sections
        ax.text(separator_pos / 2, ax.get_ylim()[0] * 0.9, 
               'KNOCKDOWNS', ha='center', fontsize=11, 
               fontweight='bold', color='#e74c3c')
        ax.text(separator_pos + (len(ku_data) / 2), ax.get_ylim()[0] * 0.9, 
               'KNOCKUPS', ha='center', fontsize=11, 
               fontweight='bold', color='#3498db')
    
    # Create informative labels
    labels = []
    for _, row in plot_data.iterrows():
        gene = row['genes'][0]
        factor = f"{row['factor']:.1f}x"
        
        # Check if condition-specific
        condition = ''
        if 'ALS' in row['id']:
            condition = '\n(ALS)'
        elif 'PN' in row['id']:
            condition = '\n(PN)'
        
        label = f"{gene}\n{factor}{condition}"
        labels.append(label)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    
    ax.set_ylabel('log2 Fold Change', fontsize=12)
    ax.set_xlabel('Perturbation (Gene, Factor, Condition)', fontsize=12)
    
    title = 'Perturbation Effect Sizes: Actual vs Expected'
    if not show_expected:
        title = 'Perturbation Effect Sizes (Expected)'
    ax.set_title(title, fontsize=13, fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3)
    
    # Legend
    if show_expected:
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Add value labels on bars (actual values)
    for i, (bar, val) in enumerate(zip(bars, bar_values)):
        height = bar.get_height()
        va = 'bottom' if height > 0 else 'top'
        offset = 0.1 if height > 0 else -0.1
        ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                f'{val:.2f}', ha='center', va=va, fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    return fig

# Compatibility check
def check_compatibility():
    """Check if all visualization dependencies are available."""
    missing = []
    
    try:
        import matplotlib
    except ImportError:
        missing.append('matplotlib')
    
    try:
        import seaborn
    except ImportError:
        missing.append('seaborn')
    
    try:
        import umap
    except ImportError:
        missing.append('umap-learn')
    
    if missing:
        print(f"Missing visualization dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True


if __name__ == "__main__":
    if check_compatibility():
        print("✓ All visualization dependencies available")
    else:
        print("✗ Some dependencies missing")
