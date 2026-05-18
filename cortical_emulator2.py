"""
GN-v18 CORTICAL STATE BANK LABORATORY — Multi-Electrode Offline Analysis
========================================================================
Loads any EDF file, extracts Laplace-Beltrami manifolds from ALL electrodes,
builds a searchable state bank, and provides analytical tools for:
- Manifold browser
- Inter-electrode distance matrix
- Hierarchical clustering of brain regions
- Basin depth testing
- Cross-state interpolation

PerceptionLab Helsinki — May 2026
"""

import numpy as np
import scipy.io.wavfile as wav
import scipy.signal as sp_signal
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import tempfile
import os
import json
import hashlib
import warnings
from datetime import datetime
import gradio as gr

try:
    import mne
    MNE_OK = True
except ImportError:
    MNE_OK = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# STYLE & PALETTE
# ─────────────────────────────────────────────────────────────────────────────
BG, PANEL = "#0d0d1a", "#13132a"
TEAL, AMBER, CORAL, PURPLE, GOLD, CYAN = "#00d4aa", "#f0a500", "#ff6b6b", "#9d7bea", "#ffd700", "#00e5ff"
DIM, WHITE = "#5a5a7a", "#e8e8f0"

def ax_style(ax, title, xlabel="", ylabel="", is_3d=False):
    ax.set_facecolor(PANEL if not is_3d else BG)
    ax.set_title(title, color=TEAL, fontsize=11, pad=10)
    if not is_3d:
        for sp in ax.spines.values(): sp.set_edgecolor(DIM)
        ax.tick_params(colors=DIM, labelsize=8)
        if xlabel: ax.set_xlabel(xlabel, color=DIM, fontsize=9)
        if ylabel: ax.set_ylabel(ylabel, color=DIM, fontsize=9)
    else:
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor(BG); ax.yaxis.pane.set_edgecolor(BG); ax.zaxis.pane.set_edgecolor(BG)
        ax.tick_params(colors=DIM, labelsize=8)

# ─────────────────────────────────────────────────────────────────────────────
# CORE MATHEMATICS (Unchanged from GN-v17)
# ─────────────────────────────────────────────────────────────────────────────
def compute_diffusion_map(V, eps_scale=0.2, num_modes=3):
    N = V.shape[0]
    sq_dists = squareform(pdist(V, metric='sqeuclidean'))
    epsilon = np.median(sq_dists) * eps_scale + 1e-9
    K = np.exp(-sq_dists / epsilon)
    
    D = np.sum(K, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
    L_sym = np.eye(N) - (D_inv_sqrt @ K @ D_inv_sqrt)
    
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
    idx = np.argsort(eigenvalues)
    true_modes = eigenvectors[:, idx[1:num_modes+1]]
    return K, true_modes

def estimate_tangent_bundle(modes_3d, k_neighbors=10):
    N = modes_3d.shape[0]
    tangents = np.zeros_like(modes_3d)
    
    for i in range(N):
        dists = np.sum((modes_3d - modes_3d[i])**2, axis=1)
        nearest_idx = np.argsort(dists)[:k_neighbors]
        
        local_centered = modes_3d[nearest_idx] - np.mean(modes_3d[nearest_idx], axis=0)
        _, _, Vt = np.linalg.svd(local_centered)
        v = Vt[0]
        
        if i < N - 1:
            temporal_flow = modes_3d[i+1] - modes_3d[i]
            if np.dot(v, temporal_flow) < 0:
                v = -v
        tangents[i] = v
    return tangents

def extract_manifold_from_signal(signal, tau=1, dim=16, eps=0.2, k_neighbors=10, max_samples=2000):
    """Extract manifold and tangent bundle from a 1D signal"""
    sig = signal.copy()
    if len(sig) > max_samples:
        sig = sp_signal.resample(sig, max_samples)
    
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-9)
    min_len = (int(dim) - 1) * int(tau) + 1
    
    if len(sig) < min_len:
        return None, None
    
    N_emb = len(sig) - min_len + 1
    V = np.zeros((N_emb, int(dim)))
    for i in range(int(dim)):
        V[:, i] = sig[i * int(tau) : i * int(tau) + N_emb]
    
    _, modes = compute_diffusion_map(V, eps, num_modes=3)
    tangents = estimate_tangent_bundle(modes, k_neighbors)
    
    return modes, tangents

def manifold_distance(modes_A, modes_B):
    """Chamfer distance between two manifolds"""
    if modes_A is None or modes_B is None:
        return float('inf')
    
    # Ensure same length by truncating to shorter
    min_len = min(len(modes_A), len(modes_B))
    A = modes_A[:min_len]
    B = modes_B[:min_len]
    
    # Nearest neighbor distances
    dist_AB = np.mean([np.min(np.sum((A - p)**2, axis=1)) for p in B])
    dist_BA = np.mean([np.min(np.sum((B - p)**2, axis=1)) for p in A])
    return np.sqrt((dist_AB + dist_BA) / 2)

def basin_depth(modes, tangents, shock_magnitudes=[0.1, 0.2, 0.5, 1.0], sim_steps=500):
    """Measure how stable the manifold is under perturbation"""
    results = []
    for mag in shock_magnitudes:
        # Simple simulation: perturb and see if it returns
        from scipy.spatial import KDTree
        tree = KDTree(modes)
        
        # Start at manifold center
        start_idx = len(modes) // 2
        state = modes[start_idx].copy() + np.random.randn(3) * mag
        
        final_dist = 0.0
        for _ in range(sim_steps):
            dist, idx = tree.query(state)
            if dist < 0.01:
                final_dist = dist
                break
            # Move along tangent direction
            state += tangents[idx] * 0.1
        else:
            final_dist = dist
        
        results.append({
            'shock_magnitude': mag,
            'final_distance': final_dist,
            'recovered': final_dist < 0.05
        })
    
    # Compute basin depth metric: max shock that still recovers
    max_recoverable = 0.0
    for r in results:
        if r['recovered'] and r['shock_magnitude'] > max_recoverable:
            max_recoverable = r['shock_magnitude']
    
    return max_recoverable, results

# ─────────────────────────────────────────────────────────────────────────────
# STATE BANK: Process entire EDF into searchable database
# ─────────────────────────────────────────────────────────────────────────────
class CorticalStateBank:
    def __init__(self):
        self.states = {}  # channel_name -> {modes, tangents, metadata}
        self.distance_matrix = None
        self.channel_list = []
    
    def process_edf(self, edf_path, tau=1, dim=16, eps=0.2, k_neighbors=8, duration_sec=3):
        """Extract manifolds from every channel in the EDF"""
        if not MNE_OK:
            raise ImportError("MNE not installed")
        
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        fs = int(raw.info['sfreq'])
        n_samples = int(fs * duration_sec)
        
        results = {}
        channels = raw.ch_names
        
        for ch in channels:
            try:
                sig = raw.get_data(picks=[ch], start=0, stop=n_samples)[0]
                modes, tangents = extract_manifold_from_signal(
                    sig, tau, dim, eps, k_neighbors
                )
                if modes is not None:
                    results[ch] = {
                        'modes': modes,
                        'tangents': tangents,
                        'metadata': {
                            'channel': ch,
                            'duration_sec': duration_sec,
                            'sample_rate': fs,
                            'tau': tau,
                            'dim': dim,
                            'eps': eps
                        }
                    }
            except Exception as e:
                print(f"Failed on {ch}: {e}")
        
        self.states = results
        self.channel_list = list(results.keys())
        self._compute_distance_matrix()
        return results
    
    def _compute_distance_matrix(self):
        """Compute pairwise distances between all channels"""
        n = len(self.channel_list)
        self.distance_matrix = np.zeros((n, n))
        for i, ch1 in enumerate(self.channel_list):
            for j, ch2 in enumerate(self.channel_list):
                if i <= j:
                    d = manifold_distance(
                        self.states[ch1]['modes'],
                        self.states[ch2]['modes']
                    )
                    self.distance_matrix[i, j] = d
                    self.distance_matrix[j, i] = d
    
    def get_clustering(self, method='ward', t=0.5):
        """Hierarchical clustering of electrodes by manifold geometry"""
        if self.distance_matrix is None:
            return None
        
        # Condensed distance matrix for linkage
        condensed = squareform(self.distance_matrix)
        Z = linkage(condensed, method=method)
        
        # Assign clusters
        clusters = fcluster(Z, t=t, criterion='distance')
        
        return Z, clusters
    
    def get_basin_depths(self, shock_magnitudes=[0.1, 0.2, 0.5, 1.0]):
        """Compute basin depth for every channel"""
        depths = {}
        for ch in self.channel_list:
            state = self.states[ch]
            depth, _ = basin_depth(
                state['modes'], 
                state['tangents'],
                shock_magnitudes
            )
            depths[ch] = depth
        return depths

# Global bank instance
state_bank = CorticalStateBank()

# ─────────────────────────────────────────────────────────────────────────────
# GRADIO UI FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def process_full_edf(edf_file, tau, dim, eps, k_neigh, duration):
    """Load EDF and extract all state machines"""
    if edf_file is None:
        return None, "No file uploaded", "", None
    
    try:
        state_bank.process_edf(
            edf_file.name, 
            tau=int(tau), 
            dim=int(dim), 
            eps=eps, 
            k_neighbors=int(k_neigh),
            duration_sec=duration
        )
        
        n_channels = len(state_bank.channel_list)
        summary = f"✅ Processed {n_channels} electrodes successfully.\n\nChannels: {', '.join(state_bank.channel_list[:10])}"
        if n_channels > 10:
            summary += f"\n... and {n_channels - 10} more"
        
        # Create distance matrix heatmap
        fig = plt.figure(figsize=(12, 10), facecolor=BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(PANEL)
        im = ax.imshow(state_bank.distance_matrix, cmap='plasma', aspect='auto')
        plt.colorbar(im, ax=ax, label='Manifold Distance')
        ax.set_title('Inter-Electrode Geometric Distance Matrix', color=TEAL, fontsize=14)
        ax.set_xlabel('Channel Index', color=DIM)
        ax.set_ylabel('Channel Index', color=DIM)
        
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        fig.savefig(tmp.name, facecolor=BG, bbox_inches='tight')
        plt.close(fig)
        
        return tmp.name, summary, f"Distance matrix shape: {state_bank.distance_matrix.shape}", state_bank.distance_matrix.tolist()
    
    except Exception as e:
        import traceback
        return None, f"Error: {str(e)}", traceback.format_exc(), None

def render_channel_manifold(channel_name):
    """Display the 3D manifold for a specific channel"""
    if channel_name not in state_bank.states:
        return None, f"Channel '{channel_name}' not found"
    
    state = state_bank.states[channel_name]
    modes = state['modes']
    tangents = state['tangents']
    
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    
    # 3D manifold
    ax1 = fig.add_subplot(121, projection='3d')
    ax_style(ax1, f"Manifold: {channel_name}", is_3d=True)
    ax1.plot(modes[:, 0], modes[:, 1], modes[:, 2], color=TEAL, lw=1.0, alpha=0.8)
    
    # Subsampled tangent field
    step = max(1, len(modes) // 50)
    ax1.quiver(modes[::step, 0], modes[::step, 1], modes[::step, 2],
               tangents[::step, 0], tangents[::step, 1], tangents[::step, 2],
               length=0.03, color=AMBER, alpha=0.6, normalize=True)
    
    # 2D projection (mode1 vs mode2)
    ax2 = fig.add_subplot(122)
    ax_style(ax2, f"Phase Projection: {channel_name}", xlabel="Mode 1", ylabel="Mode 2")
    ax2.plot(modes[:, 0], modes[:, 1], color=CORAL, lw=0.8, alpha=0.7)
    ax2.scatter(modes[0, 0], modes[0, 1], color=WHITE, s=30, label='Start')
    ax2.scatter(modes[-1, 0], modes[-1, 1], color=GOLD, s=30, label='End')
    ax2.legend(facecolor=PANEL, labelcolor=WHITE)
    
    fig.suptitle(f"Cortical State Machine: {channel_name}", color=WHITE, fontsize=14)
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    fig.savefig(tmp.name, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    
    metadata = json.dumps(state['metadata'], indent=2)
    return tmp.name, metadata

def compute_clustering_plot(method, threshold):
    """Generate dendrogram of electrode relationships"""
    if state_bank.distance_matrix is None:
        return None, "No data loaded"
    
    Z, clusters = state_bank.get_clustering(method=method, t=threshold)
    
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(PANEL)
    
    # Plot dendrogram
    dendrogram(Z, labels=state_bank.channel_list, ax=ax, leaf_rotation=90, leaf_font_size=8)
    ax.set_title(f'Electrode Clustering by Manifold Geometry (method={method})', color=TEAL, fontsize=12)
    ax.set_xlabel('Electrode', color=DIM)
    ax.set_ylabel('Geometric Distance', color=DIM)
    ax.tick_params(colors=DIM)
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    fig.savefig(tmp.name, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    
    # Format cluster assignment
    cluster_map = {}
    for ch, c in zip(state_bank.channel_list, clusters):
        if c not in cluster_map:
            cluster_map[c] = []
        cluster_map[c].append(ch)
    
    summary = f"Found {len(cluster_map)} clusters at threshold {threshold}:\n"
    for c, chs in cluster_map.items():
        summary += f"\nCluster {c}: {', '.join(chs[:5])}"
        if len(chs) > 5:
            summary += f" and {len(chs)-5} more"
    
    return tmp.name, summary

def compare_two_channels(ch1, ch2):
    """Compare two channels and compute their relationship"""
    if ch1 not in state_bank.states or ch2 not in state_bank.states:
        return None, "One or both channels not found"
    
    modes1 = state_bank.states[ch1]['modes']
    modes2 = state_bank.states[ch2]['modes']
    
    distance = manifold_distance(modes1, modes2)
    
    # Create comparison plot
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax_style(ax1, f"{ch1} (Teal)", is_3d=True)
    ax1.plot(modes1[:, 0], modes1[:, 1], modes1[:, 2], color=TEAL, lw=1.0)
    
    ax2 = fig.add_subplot(122, projection='3d')
    ax_style(ax2, f"{ch2} (Coral)", is_3d=True)
    ax2.plot(modes2[:, 0], modes2[:, 1], modes2[:, 2], color=CORAL, lw=1.0)
    
    fig.suptitle(f"Geometric Distance: {distance:.4f}", color=WHITE, fontsize=14)
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    fig.savefig(tmp.name, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    
    return tmp.name, f"Manifold distance between {ch1} and {ch2}: {distance:.6f}"

def compute_basin_depths_table():
    """Compute and display basin depths for all channels"""
    depths = state_bank.get_basin_depths()
    
    if not depths:
        return "No data loaded", None
    
    # Sort by depth
    sorted_depths = sorted(depths.items(), key=lambda x: x[1], reverse=True)
    
    # Create visualization
    fig = plt.figure(figsize=(12, 6), facecolor=BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(PANEL)
    
    channels = [x[0] for x in sorted_depths[:20]]  # Top 20
    depth_vals = [x[1] for x in sorted_depths[:20]]
    
    colors = [TEAL if d > 0.5 else AMBER if d > 0.2 else CORAL for d in depth_vals]
    ax.barh(channels, depth_vals, color=colors)
    ax.set_xlabel('Basin Depth (Max Recoverable Shock)', color=DIM)
    ax.set_title('Cognitive Stability by Electrode', color=TEAL, fontsize=12)
    ax.tick_params(colors=DIM)
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    fig.savefig(tmp.name, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    
    # Text summary
    summary = "Basin Depths (higher = more stable):\n"
    for ch, d in sorted_depths[:10]:
        stability = "STABLE" if d > 0.5 else "MODERATE" if d > 0.2 else "FRAGILE"
        summary += f"\n{ch}: {d:.3f} [{stability}]"
    
    return summary, tmp.name

# ─────────────────────────────────────────────────────────────────────────────
# GRADIO INTERFACE
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Monochrome(), title="GN-v18 Cortical State Bank") as demo:
    gr.Markdown("""
    # 🧠 GN-v18 Cortical State Bank Laboratory
    ### Extract, store, and study geometric brain states from every electrode simultaneously
    
    **Workflow:** Load EDF → Process All Electrodes → Browse States → Analyze Relationships
    """)
    
    with gr.Tabs():
        
        # TAB 1: LOAD & PROCESS
        with gr.TabItem("📁 1. Load EDF & Extract States"):
            with gr.Row():
                with gr.Column(scale=1):
                    edf_input = gr.File(label="Upload EDF File", file_types=[".edf"])
                    gr.Markdown("### Extraction Parameters")
                    tau_sl = gr.Slider(1, 50, 1, step=1, label="τ (Delay Spacing)")
                    dim_sl = gr.Slider(2, 64, 16, step=1, label="d (Embedding Dimension)")
                    eps_sl = gr.Slider(0.1, 2.0, 0.2, label="ε (Kernel Bandwidth)")
                    k_sl = gr.Slider(5, 30, 8, step=1, label="Tangent SVD Window")
                    dur_sl = gr.Slider(1, 10, 3, step=1, label="Duration (seconds)")
                    
                    process_btn = gr.Button("🔥 Extract All Electrode States", variant="primary")
                    
                with gr.Column(scale=2):
                    dist_matrix_plot = gr.Image(label="Inter-Electrode Distance Matrix", type="filepath")
                    process_summary = gr.Textbox(label="Extraction Summary", lines=15)
                    debug_output = gr.Textbox(label="Debug", visible=False)
                    distance_data = gr.State()
            
            process_btn.click(
                fn=process_full_edf,
                inputs=[edf_input, tau_sl, dim_sl, eps_sl, k_sl, dur_sl],
                outputs=[dist_matrix_plot, process_summary, debug_output, distance_data]
            )
        
        # TAB 2: BROWSE INDIVIDUAL STATES
        with gr.TabItem("🔍 2. Browse Electrode Manifolds"):
            with gr.Row():
                with gr.Column(scale=1):
                    channel_dropdown = gr.Dropdown(label="Select Electrode", choices=[])
                    refresh_btn = gr.Button("Refresh Channel List")
                    view_btn = gr.Button("Render Manifold", variant="primary")
                    
                with gr.Column(scale=2):
                    manifold_plot = gr.Image(label="Manifold Geometry + Tangent Field", type="filepath")
                    channel_metadata = gr.Textbox(label="Extraction Parameters", lines=10)
            
            def refresh_channels():
                return gr.update(choices=state_bank.channel_list)
            
            refresh_btn.click(fn=refresh_channels, outputs=[channel_dropdown])
            view_btn.click(fn=render_channel_manifold, inputs=[channel_dropdown], outputs=[manifold_plot, channel_metadata])
        
        # TAB 3: CLUSTER ANALYSIS
        with gr.TabItem("📊 3. Cluster Electrodes"):
            with gr.Row():
                with gr.Column(scale=1):
                    cluster_method = gr.Radio(["ward", "complete", "average", "single"], label="Clustering Method", value="ward")
                    cluster_thresh = gr.Slider(0.1, 2.0, 0.8, step=0.05, label="Distance Threshold")
                    cluster_btn = gr.Button("Compute Clustering", variant="primary")
                    
                with gr.Column(scale=2):
                    dendrogram_plot = gr.Image(label="Electrode Dendrogram", type="filepath")
                    cluster_summary = gr.Textbox(label="Cluster Assignments", lines=15)
            
            cluster_btn.click(fn=compute_clustering_plot, inputs=[cluster_method, cluster_thresh], outputs=[dendrogram_plot, cluster_summary])
        
        # TAB 4: COMPARE CHANNELS
        with gr.TabItem("⚖️ 4. Compare Two Electrodes"):
            with gr.Row():
                with gr.Column(scale=1):
                    ch1_drop = gr.Dropdown(label="Electrode A", choices=[])
                    ch2_drop = gr.Dropdown(label="Electrode B", choices=[])
                    compare_btn = gr.Button("Compare", variant="primary")
                    
                with gr.Column(scale=2):
                    compare_plot = gr.Image(label="Side-by-Side Comparison", type="filepath")
                    compare_text = gr.Textbox(label="Distance Metric", lines=5)
            
            def refresh_compare():
                choices = state_bank.channel_list
                return gr.update(choices=choices), gr.update(choices=choices)
            
            compare_btn.click(fn=compare_two_channels, inputs=[ch1_drop, ch2_drop], outputs=[compare_plot, compare_text])
            
            # Auto-refresh dropdowns when state changes
            def update_choices():
                return gr.update(choices=state_bank.channel_list), gr.update(choices=state_bank.channel_list)
            
            process_btn.click(fn=update_choices, outputs=[ch1_drop, ch2_drop])
            refresh_btn.click(fn=update_choices, outputs=[ch1_drop, ch2_drop])
        
        # TAB 5: BASIN DEPTH ANALYSIS (Cognitive Stability)
        with gr.TabItem("📈 5. Cognitive Stability (Basin Depth)"):
            with gr.Row():
                with gr.Column(scale=1):
                    depth_btn = gr.Button("Compute Stability Scores", variant="primary")
                    
                with gr.Column(scale=2):
                    depth_plot = gr.Image(label="Basin Depth by Electrode", type="filepath")
                    depth_summary = gr.Textbox(label="Stability Rankings", lines=15)
            
            depth_btn.click(fn=compute_basin_depths_table, outputs=[depth_summary, depth_plot])
        
        # TAB 6: THEORY & INTERPRETATION
        with gr.TabItem("📖 6. Interpretation Guide"):
            gr.Markdown("""
            ## What These Geometric Measurements Mean
            
            ### Basin Depth (Cognitive Stability)
            - **>0.5**: Very stable state — highly practiced/automatic behavior
            - **0.2-0.5**: Moderate stability — flexible, attention-dependent
            - **<0.2**: Fragile — easily perturbed, high plasticity
            
            ### Manifold Distance Between Electrodes
            - **<0.1**: Nearly identical geometry — highly synchronized regions
            - **0.1-0.5**: Moderately related — functional coupling
            - **>0.5**: Very different geometry — independent processing
            
            ### Clustering Patterns
            - Tight clusters = functional networks
            - Isolated electrodes = specialized or artifact-prone regions
            - Hierarchical structure = nested functional organization
            
            ### Tangent Field (Teal Arrows)
            - Laminar flow = deterministic, predictable dynamics
            - Turbulent/frayed = high entropy, exploratory state
            - Abrupt direction changes = phase transitions
            
            ### What You Can Discover
            1. **Subject fingerprint**: Each person has unique manifold geometry
            2. **State classification**: Different cognitive tasks produce distinct attractors
            3. **Pathology markers**: Abnormal basin depths may indicate neurological conditions
            4. **Development tracking**: Manifold complexity increases with age/learning
            """)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0")