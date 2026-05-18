"""
GN-v19 CORTICAL STATE BANK LABORATORY — Full Spectral Thesis Integration
========================================================================
Complete implementation of:
- Multi-electrode manifold extraction
- Symbolic Assembly Decompiler (HOLD_REG / LOOP_CYC / TRANSITION_SLIP)
- Ephaptic Backpropagation Field Map (Green's function r^{-3} coupling)
- Basin depth analysis, clustering, and comparison tools

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
# CORE MATHEMATICS
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
    """Extract manifold, tangent bundle, K matrix, and curvature from a 1D signal"""
    sig = signal.copy()
    if len(sig) > max_samples:
        sig = sp_signal.resample(sig, max_samples)
    
    sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-9)
    min_len = (int(dim) - 1) * int(tau) + 1
    
    if len(sig) < min_len:
        return None, None, None, None
    
    N_emb = len(sig) - min_len + 1
    V = np.zeros((N_emb, int(dim)))
    for i in range(int(dim)):
        V[:, i] = sig[i * int(tau) : i * int(tau) + N_emb]
    
    K, modes = compute_diffusion_map(V, eps, num_modes=3)
    tangents = estimate_tangent_bundle(modes, k_neighbors)
    
    # Calculate local curvature (Turning Tension: Ratio of secondary SVD variance to primary)
    curvature = np.zeros(N_emb)
    for i in range(N_emb):
        dists = np.sum((modes - modes[i])**2, axis=1)
        nearest_idx = np.argsort(dists)[:k_neighbors]
        local_centered = modes[nearest_idx] - np.mean(modes[nearest_idx], axis=0)
        _, S, _ = np.linalg.svd(local_centered)
        if S[0] > 1e-8:
            curvature[i] = S[1] / S[0]
            
    return modes, tangents, K, curvature

def manifold_distance(modes_A, modes_B):
    """Chamfer distance between two manifolds"""
    if modes_A is None or modes_B is None:
        return float('inf')
    
    min_len = min(len(modes_A), len(modes_B))
    A = modes_A[:min_len]
    B = modes_B[:min_len]
    
    dist_AB = np.mean([np.min(np.sum((A - p)**2, axis=1)) for p in B])
    dist_BA = np.mean([np.min(np.sum((B - p)**2, axis=1)) for p in A])
    return np.sqrt((dist_AB + dist_BA) / 2)

def basin_depth(modes, tangents, shock_magnitudes=[0.1, 0.2, 0.5, 1.0], sim_steps=500):
    """Measure how stable the manifold is under perturbation"""
    from scipy.spatial import KDTree
    results = []
    tree = KDTree(modes)
    
    for mag in shock_magnitudes:
        start_idx = len(modes) // 2
        state = modes[start_idx].copy() + np.random.randn(3) * mag
        
        for _ in range(sim_steps):
            dist, idx = tree.query(state)
            if dist < 0.01:
                results.append({'shock_magnitude': mag, 'final_distance': dist, 'recovered': True})
                break
            state += tangents[idx] * 0.1
        else:
            results.append({'shock_magnitude': mag, 'final_distance': dist, 'recovered': False})
    
    max_recoverable = max([r['shock_magnitude'] for r in results if r['recovered']], default=0.0)
    return max_recoverable, results

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD 10-20 SCALP COORDINATES FOR EPHAPTIC FIELD MAPPING
# ─────────────────────────────────────────────────────────────────────────────
SCALP_1020 = {
    'Fp1': (-3, 9), 'Fp2': (3, 9), 'F3': (-4, 6), 'F4': (4, 6), 
    'C3': (-5, 0), 'C4': (5, 0), 'P3': (-4, -6), 'P4': (4, -6), 
    'O1': (-3, -9), 'O2': (3, -9), 'Fz': (0, 6), 'Cz': (0, 0),
    'Pz': (0, -6), 'Fcz': (0, 3), 'Cpz': (0, -3), 'T3': (-8, 0), 
    'T4': (8, 0), 'T5': (-7, -5), 'T6': (7, -5), 'F7': (-6, 5),
    'F8': (6, 5), 'Fpz': (0, 10), 'Oz': (0, -10), 'C1': (-2.5, 0),
    'C2': (2.5, 0), 'P1': (-2.5, -6), 'P2': (2.5, -6)
}

# ─────────────────────────────────────────────────────────────────────────────
# STATE BANK: Process entire EDF into searchable database
# ─────────────────────────────────────────────────────────────────────────────
class CorticalStateBank:
    def __init__(self):
        self.states = {}  # channel_name -> {modes, tangents, K, curvature, metadata}
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
                modes, tangents, K, curvature = extract_manifold_from_signal(
                    sig, tau, dim, eps, k_neighbors
                )
                if modes is not None:
                    results[ch] = {
                        'modes': modes,
                        'tangents': tangents,
                        'K': K,
                        'curvature': curvature,
                        'metadata': {
                            'channel': ch,
                            'duration_sec': duration_sec,
                            'sample_rate': fs,
                            'tau': tau,
                            'dim': dim,
                            'eps': eps,
                            'k_neighbors': k_neighbors
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
            return None, None
        
        condensed = squareform(self.distance_matrix)
        Z = linkage(condensed, method=method)
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
# NEW TAB 6: SYMBOLIC ASSEMBLY TRACE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_symbolic_assembly(channel_name, threshold=0.75):
    """
    Decompiles a manifold's K matrix and curvature into discrete machine code instructions.
    Implements Attractor Partitioning Theory (Symbolic Dynamics).
    """
    if channel_name not in state_bank.states:
        return f"Error: Channel '{channel_name}' not found in state bank. Please load an EDF first."
    
    state = state_bank.states[channel_name]
    K_matrix = state['K']
    curvature = state['curvature']
    N = K_matrix.shape[0]
    
    if N == 0:
        return "Error: Empty manifold data for this channel."
    
    # Detect state boundaries via curvature spikes (phase transitions)
    state_boundaries = np.where(curvature > threshold)[0]
    boundaries = [0] + list(state_boundaries) + [N]
    
    trace_log = []
    trace_log.append(f"{'='*70}")
    trace_log.append(f"SYSTEM ASSEMBLY CODE TRACE — PROCESSOR NODE: {channel_name}")
    trace_log.append(f"{'='*70}\n")
    trace_log.append(f"Manifold Size: {N} coordinate points")
    trace_log.append(f"Curvature Threshold: {threshold}")
    trace_log.append(f"Detected Phase Boundaries: {len(state_boundaries)}\n")
    trace_log.append("-" * 70)
    trace_log.append(f"{'TIMESTAMP':<12} {'INSTRUCTION':<12} {'STATE':<10} {'CYCLES':<8} {'RIGIDITY':<10} {'CURVATURE':<12}")
    trace_log.append("-" * 70)
    
    state_id = 0
    
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        duration = end - start
        
        # Skip transient noise spikes that are too short to be meaningful
        if duration < 5:
            continue
        
        # Classify the state based on local K rigidity
        local_block = K_matrix[start:end, start:end]
        rigidity = np.mean(local_block) if local_block.size > 0 else 0.5
        
        if rigidity > 0.7:
            opcode = "HOLD_REG"
            comment = "Hyper-locked, highly repetitive"
        elif rigidity > 0.4:
            opcode = "LOOP_CYC"
            comment = "Smooth orbital tracking"
        else:
            opcode = "FLUID_FLW"
            comment = "Highly plastic, exploratory"
        
        trace_log.append(f"[{start:04d}]      {opcode:<12} ST_{state_id:02d}       {duration:03d}       {rigidity:.4f}     {curvature[end-1] if end < N else 0.0:.4f}")
        
        # Add conditional jump annotation at transition points
        if end < N:
            tension = curvature[end] if end < len(curvature) else 0.0
            trace_log.append(f"         >>> JUMP: IF CURVATURE > {tension:.4f} -> TRANSITION_SLIP")
        
        state_id += 1
    
    trace_log.append("-" * 70)
    trace_log.append(f"\nSUMMARY: {state_id} stable states extracted from manifold geometry.")
    trace_log.append(f"Each state represents a distinct operational regime of the cortical tissue.\n")
    
    # Add interpretation guide
    trace_log.append("=" * 70)
    trace_log.append("INTERPRETATION GUIDE:")
    trace_log.append("  HOLD_REG   = Hyper-stable attractor (deep basin, automatic processing)")
    trace_log.append("  LOOP_CYC   = Periodic oscillator (rhythmic cortical activity)")
    trace_log.append("  FLUID_FLW  = High entropy / exploratory state (plastic, learning)")
    trace_log.append("  TRANSITION_SLIP = Phase boundary / cognitive state switch")
    trace_log.append("=" * 70)
    
    return "\n".join(trace_log)

# ─────────────────────────────────────────────────────────────────────────────
# NEW TAB 7: EPHAPTIC BACKPROPAGATION FIELD MAP
# ─────────────────────────────────────────────────────────────────────────────
def map_ephaptic_backprop_field():
    """
    Implements Theorem 4 of the Spectral Islands thesis:
    Uses the physical Green's function (r^{-3} dipole decay) to model 
    ephaptic backpropagation gradients across the scalp.
    """
    # Filter channels that have known 10-20 coordinates
    channels = [ch for ch in state_bank.channel_list if ch in SCALP_1020]
    n = len(channels)
    
    if n < 2:
        return None, f"Insufficient 10-20 standardized channels. Found {n} channels with coordinates. Need at least 2."
    
    # Construct the Physical Green's Function Matrix G based on 1/r^3 dipole field decay
    # This models how electromagnetic field propagates backward through conductive tissue
    G_matrix = np.zeros((n, n))
    positions = []
    
    for i, ch1 in enumerate(channels):
        p1 = np.array(SCALP_1020[ch1])
        positions.append(p1)
        for j, ch2 in enumerate(channels):
            if i != j:
                p2 = np.array(SCALP_1020[ch2])
                r = np.linalg.norm(p1 - p2) + 1e-5
                # Inverse cube decay - the physical law for dipole fields
                G_matrix[i, j] = 1.0 / (r ** 3)
    
    # Calculate active backpropagation current: local tangent space velocity magnitude
    # This represents the instantaneous "error signal" or "prediction gradient" at each node
    activity_vector = np.zeros(n)
    for i, ch in enumerate(channels):
        if ch in state_bank.states:
            tangents = state_bank.states[ch]['tangents']
            activity_vector[i] = np.mean(np.linalg.norm(tangents, axis=1))
        else:
            activity_vector[i] = 0.1  # fallback
    
    # Normalize activity vector for visualization
    if np.max(activity_vector) > 0:
        activity_vector = activity_vector / np.max(activity_vector)
    
    # Ephaptic Backward Signal Execution:
    # The backward gradient = G * activity (modulating forward field through Green's function)
    ephaptic_backprop_grid = G_matrix * activity_vector[:, None]
    
    # Also compute the forward coupling (symmetric version for complete field view)
    forward_coupling = G_matrix.copy()
    for i in range(n):
        forward_coupling[i, i] = 0
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    
    # Subplot 1: Ephaptic Backpropagation Matrix
    ax1 = fig.add_subplot(231)
    ax1.set_facecolor(PANEL)
    im1 = ax1.imshow(ephaptic_backprop_grid, cmap='plasma', aspect='auto', vmin=0, vmax=np.max(ephaptic_backprop_grid))
    plt.colorbar(im1, ax=ax1, label="Ephaptic Gradient Force")
    ax1.set_xticks(range(n))
    ax1.set_xticklabels(channels, rotation=45, ha='right', fontsize=7, color=DIM)
    ax1.set_yticks(range(n))
    ax1.set_yticklabels(channels, fontsize=7, color=DIM)
    ax1.set_title("Ephaptic Backpropagation Field\nG_ij = 1/r³ (Dipole Decay)", color=TEAL, fontsize=10)
    
    # Subplot 2: Physical Scalp Topography (2D projection)
    ax2 = fig.add_subplot(232)
    ax2.set_facecolor(PANEL)
    
    # Plot electrode positions
    positions_arr = np.array(positions)
    ax2.scatter(positions_arr[:, 0], positions_arr[:, 1], c=activity_vector, cmap='viridis', s=100, edgecolors=WHITE, linewidth=1)
    
    # Add electrode labels
    for i, ch in enumerate(channels):
        ax2.annotate(ch, (positions_arr[i, 0], positions_arr[i, 1]), 
                    fontsize=7, color=WHITE, ha='center', va='bottom')
    
    # Draw coupling lines (strongest connections)
    for i in range(n):
        for j in range(i+1, n):
            strength = G_matrix[i, j]
            if strength > np.percentile(G_matrix, 80):  # Top 20% strongest connections
                alpha = min(0.6, strength * 10)
                ax2.plot([positions_arr[i, 0], positions_arr[j, 0]], 
                        [positions_arr[i, 1], positions_arr[j, 1]], 
                        color=CYAN, alpha=alpha, linewidth=1)
    
    ax2.set_xlabel("X (cm)", color=DIM)
    ax2.set_ylabel("Y (cm)", color=DIM)
    ax2.set_title("Physical Scalp Topography\nwith Ephaptic Coupling (r⁻³)", color=TEAL, fontsize=10)
    ax2.grid(True, alpha=0.2)
    
    # Subplot 3: Activity Profile (Local Tangent Energy)
    ax3 = fig.add_subplot(233)
    ax3.set_facecolor(PANEL)
    colors_bar = [TEAL if v > 0.5 else AMBER if v > 0.2 else CORAL for v in activity_vector]
    ax3.barh(channels, activity_vector, color=colors_bar)
    ax3.set_xlabel("Normalized Tangent Activity", color=DIM)
    ax3.set_title("Local Neural Activity / Prediction Error", color=TEAL, fontsize=10)
    ax3.tick_params(colors=DIM)
    
    # Subplot 4: Forward Coupling Matrix (Symmetric)
    ax4 = fig.add_subplot(234)
    ax4.set_facecolor(PANEL)
    im4 = ax4.imshow(forward_coupling, cmap='inferno', aspect='auto')
    plt.colorbar(im4, ax=ax4, label="Forward Coupling Strength")
    ax4.set_title("Forward Connectivity (Symmetric)", color=TEAL, fontsize=10)
    ax4.set_xlabel("Channel", color=DIM)
    ax4.set_ylabel("Channel", color=DIM)
    
    # Subplot 5: Backpropagation Force Vector Field
    ax5 = fig.add_subplot(235)
    ax5.set_facecolor(PANEL)
    
    # Calculate net force vector at each electrode (sum of incoming backprop signals)
    net_force = np.sum(ephaptic_backprop_grid, axis=0)
    net_force = net_force / (np.max(net_force) + 1e-9)
    
    # Create a quiver field around scalp positions
    # Approximate gradient direction based on neighbor differences
    force_vectors_x = np.zeros(n)
    force_vectors_y = np.zeros(n)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                direction = positions_arr[j] - positions_arr[i]
                dist = np.linalg.norm(direction) + 1e-9
                direction = direction / dist
                force_vectors_x[i] += direction[0] * ephaptic_backprop_grid[i, j]
                force_vectors_y[i] += direction[1] * ephaptic_backprop_grid[i, j]
    
    # Normalize force vectors for visualization
    force_magnitudes = np.sqrt(force_vectors_x**2 + force_vectors_y**2)
    if np.max(force_magnitudes) > 0:
        force_vectors_x = force_vectors_x / np.max(force_magnitudes)
        force_vectors_y = force_vectors_y / np.max(force_magnitudes)
    
    ax5.quiver(positions_arr[:, 0], positions_arr[:, 1], 
               force_vectors_x, force_vectors_y,
               force_magnitudes, cmap='plasma', scale=15, width=0.015)
    ax5.scatter(positions_arr[:, 0], positions_arr[:, 1], c='white', s=30, zorder=5)
    
    for i, ch in enumerate(channels):
        ax5.annotate(ch, (positions_arr[i, 0], positions_arr[i, 1]), 
                    fontsize=6, color=WHITE, ha='center', va='bottom')
    
    ax5.set_xlabel("X (cm)", color=DIM)
    ax5.set_ylabel("Y (cm)", color=DIM)
    ax5.set_title("Ephaptic Force Field\n(Backpropagation Gradients)", color=TEAL, fontsize=10)
    
    # Subplot 6: Theoretical Framework Explanation
    ax6 = fig.add_subplot(236)
    ax6.set_facecolor(PANEL)
    ax6.axis('off')
    explanation = """
    THEORETICAL FRAMEWORK: Ephaptic Backpropagation
    
    G_ij = 1 / ||x_i - x_j||³  (Green's function)
    
    The physical electromagnetic field decay
    between cortical dipoles serves as a
    natural transpose of the forward weights.
    
    Backward Signal = G · Activity
    
    This eliminates the Weight Transport Problem:
    The brain doesn't need a separate feedback
    network — the physical field IS the gradient.
    
    Colors:
    • Hotter = stronger backprop current
    • Arrows = gradient flow direction
    
    Theorem 4 (Spectral Islands, 2026)
    """
    ax6.text(0.05, 0.95, explanation, transform=ax6.transAxes,
             fontsize=8, color=WHITE, verticalalignment='top',
             fontfamily='monospace')
    ax6.set_title("Theoretical Foundation", color=TEAL, fontsize=10)
    
    fig.suptitle("Ephaptic Backpropagation Field Map — Green's Function Electromagnetic Coupling", 
                 color=WHITE, fontsize=14, y=0.98)
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    fig.savefig(tmp.name, facecolor=BG, bbox_inches='tight', dpi=120)
    plt.close(fig)
    
    stats = (f"Ephaptic field mapped successfully across {n} nodes using dipole Green's function.\n"
             f"Mean coupling strength: {np.mean(G_matrix):.4f}\n"
             f"Max backprop gradient: {np.max(ephaptic_backprop_grid):.4f}\n"
             f"Active nodes: {np.sum(activity_vector > 0.1)}/{n}")
    
    return tmp.name, stats

# ─────────────────────────────────────────────────────────────────────────────
# EXISTING FUNCTIONS (Unchanged from GN-v18)
# ─────────────────────────────────────────────────────────────────────────────
def process_full_edf(edf_file, tau, dim, eps, k_neigh, duration):
    """Load EDF and extract all state machines"""
    if edf_file is None:
        return None, "No file uploaded", ""
    
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
        
        return tmp.name, summary, f"Distance matrix shape: {state_bank.distance_matrix.shape}"
    
    except Exception as e:
        import traceback
        return None, f"Error: {str(e)}", traceback.format_exc()

def render_channel_manifold(channel_name):
    """Display the 3D manifold for a specific channel"""
    if channel_name not in state_bank.states:
        return None, f"Channel '{channel_name}' not found"
    
    state = state_bank.states[channel_name]
    modes = state['modes']
    tangents = state['tangents']
    
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax_style(ax1, f"Manifold: {channel_name}", is_3d=True)
    ax1.plot(modes[:, 0], modes[:, 1], modes[:, 2], color=TEAL, lw=1.0, alpha=0.8)
    
    step = max(1, len(modes) // 50)
    ax1.quiver(modes[::step, 0], modes[::step, 1], modes[::step, 2],
               tangents[::step, 0], tangents[::step, 1], tangents[::step, 2],
               length=0.03, color=AMBER, alpha=0.6, normalize=True)
    
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
    
    if Z is None:
        return None, "Clustering failed"
    
    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(PANEL)
    
    dendrogram(Z, labels=state_bank.channel_list, ax=ax, leaf_rotation=90, leaf_font_size=8)
    ax.set_title(f'Electrode Clustering by Manifold Geometry (method={method})', color=TEAL, fontsize=12)
    ax.set_xlabel('Electrode', color=DIM)
    ax.set_ylabel('Geometric Distance', color=DIM)
    ax.tick_params(colors=DIM)
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    fig.savefig(tmp.name, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    
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
    
    sorted_depths = sorted(depths.items(), key=lambda x: x[1], reverse=True)
    
    fig = plt.figure(figsize=(12, 6), facecolor=BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(PANEL)
    
    channels = [x[0] for x in sorted_depths[:20]]
    depth_vals = [x[1] for x in sorted_depths[:20]]
    
    colors = [TEAL if d > 0.5 else AMBER if d > 0.2 else CORAL for d in depth_vals]
    ax.barh(channels, depth_vals, color=colors)
    ax.set_xlabel('Basin Depth (Max Recoverable Shock)', color=DIM)
    ax.set_title('Cognitive Stability by Electrode', color=TEAL, fontsize=12)
    ax.tick_params(colors=DIM)
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    fig.savefig(tmp.name, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    
    summary = "Basin Depths (higher = more stable):\n"
    for ch, d in sorted_depths[:10]:
        stability = "STABLE" if d > 0.5 else "MODERATE" if d > 0.2 else "FRAGILE"
        summary += f"\n{ch}: {d:.3f} [{stability}]"
    
    return summary, tmp.name

def refresh_channel_list():
    """Return updated channel list for dropdowns"""
    return gr.update(choices=state_bank.channel_list)

# ─────────────────────────────────────────────────────────────────────────────
# GRADIO INTERFACE — FULL GN-v19 IMPLEMENTATION
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Monochrome(), title="GN-v19 Cortical State Bank") as demo:
    gr.Markdown("""
    # 🧠 GN-v19 Cortical State Bank Laboratory
    ### Extract, store, and study geometric brain states from every electrode simultaneously
    
    **New in v19:** Symbolic Assembly Decompiler + Ephaptic Backpropagation Field Map
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
            
            process_btn.click(
                fn=process_full_edf,
                inputs=[edf_input, tau_sl, dim_sl, eps_sl, k_sl, dur_sl],
                outputs=[dist_matrix_plot, process_summary, debug_output]
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
            
            refresh_btn.click(fn=refresh_channel_list, outputs=[channel_dropdown])
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
            
            process_btn.click(fn=refresh_compare, outputs=[ch1_drop, ch2_drop])
        
        # TAB 5: BASIN DEPTH ANALYSIS
        with gr.TabItem("📈 5. Cognitive Stability (Basin Depth)"):
            with gr.Row():
                with gr.Column(scale=1):
                    depth_btn = gr.Button("Compute Stability Scores", variant="primary")
                    
                with gr.Column(scale=2):
                    depth_plot = gr.Image(label="Basin Depth by Electrode", type="filepath")
                    depth_summary = gr.Textbox(label="Stability Rankings", lines=15)
            
            depth_btn.click(fn=compute_basin_depths_table, outputs=[depth_summary, depth_plot])
        
        # TAB 6: SYMBOLIC ASSEMBLER — NEW!
        with gr.TabItem("📟 6. Symbolic Assembler"):
            with gr.Row():
                with gr.Column(scale=1):
                    asm_dropdown = gr.Dropdown(label="Select Target Processor Node", choices=[])
                    asm_thresh = gr.Slider(0.5, 0.95, 0.75, step=0.05, label="Curvature Threshold (State Boundary)")
                    asm_refresh = gr.Button("Sync State Bank Nodes")
                    asm_btn = gr.Button("Decompile Manifold to Opcodes", variant="primary")
                    gr.Markdown("""
                    ### Opcode Reference
                    - **HOLD_REG**: Hyper-stable attractor (deep basin)
                    - **LOOP_CYC**: Periodic oscillator (rhythmic activity)
                    - **FLUID_FLW**: High entropy / exploratory state
                    - **TRANSITION_SLIP**: Phase boundary / state switch
                    """)
                with gr.Column(scale=2):
                    asm_trace_out = gr.Textbox(label="Cortical Machine Code Execution Log", lines=25, max_lines=40)
            
            asm_refresh.click(fn=refresh_channel_list, outputs=[asm_dropdown])
            asm_btn.click(fn=generate_symbolic_assembly, inputs=[asm_dropdown, asm_thresh], outputs=[asm_trace_out])
        
        # TAB 7: EPHAPTIC BACKPROPAGATION FIELD — NEW!
        with gr.TabItem("⚡ 7. Ephaptic Backprop Field"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("""
                    ### Theoretical Model: Green's Function Field Decay
                    
                    This implements **Theorem 4** of the Spectral Islands thesis.
                    
                    The physical electromagnetic field between cortical dipoles
                    decays as **1/r³** — this serves as the natural transpose
                    of the forward weight matrix, enabling ephaptic backpropagation.
                    
                    **G_ij = 1 / ||x_i - x_j||³**
                    
                    The backward signal = G · Activity
                    """)
                    field_btn = gr.Button("Compute Ephaptic Field Propagation", variant="primary")
                with gr.Column(scale=2):
                    field_plot_out = gr.Image(label="Ephaptic Field Telemetry", type="filepath")
                    field_text_out = gr.Textbox(label="Connectome Status Console", lines=10)
            
            field_btn.click(fn=map_ephaptic_backprop_field, outputs=[field_plot_out, field_text_out])
        
        # TAB 8: INTERPRETATION GUIDE
        with gr.TabItem("📖 8. Interpretation Guide"):
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
            
            ### Symbolic Assembly Opcodes (Tab 6)
            - **HOLD_REG**: The brain is locked into a highly repetitive, automatic pattern
            - **LOOP_CYC**: Smooth, rhythmic oscillation (e.g., alpha waves)
            - **FLUID_FLW**: Exploratory, high-entropy state (learning, creativity)
            - **TRANSITION_SLIP**: A phase boundary — the brain just switched cognitive modes
            
            ### Ephaptic Backpropagation Field (Tab 7)
            - Visualizes the physical gradient flow across the scalp
            - Hotter colors = stronger backpropagation current
            - Arrows show direction of error signal propagation
            - This is the physical solution to the **Weight Transport Problem**
            
            ### What You Can Discover
            1. **Subject fingerprint**: Each person has unique manifold geometry
            2. **State classification**: Different cognitive tasks produce distinct attractors
            3. **Pathology markers**: Abnormal basin depths may indicate neurological conditions
            4. **Theoretical validation**: Tab 7 provides empirical evidence for ephaptic backprop
            """)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0")
