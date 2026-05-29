#[derive(Debug, Clone)]
pub enum Aggregation {
    Sum,
    Mean,
    Max,
}

#[derive(Debug, Clone)]
pub enum ReadoutMethod {
    Sum,
    Mean,
    Attention(Vec<f64>),
}

/// Graph Neural Network with spectral primitives.
pub struct GraphNeural {
    /// Adjacency matrix (n x n)
    adj: Vec<Vec<f64>>,
    /// Number of nodes
    n: usize,
    /// Feature dimension
    feature_dim: usize,
    /// Normalized Laplacian eigenvectors (cached, computed lazily)
    eigenvectors: Option<Vec<Vec<f64>>>,
    /// Eigenvalues (cached)
    eigenvalues: Option<Vec<f64>>,
}

impl GraphNeural {
    /// Create a new GraphNeural from an adjacency matrix and feature dimension.
    pub fn from_adjacency(adj: Vec<Vec<f64>>, feature_dim: usize) -> Self {
        let n = adj.len();
        assert!(n > 0, "Adjacency matrix must not be empty");
        for row in &adj {
            assert_eq!(row.len(), n, "Adjacency matrix must be square");
        }
        Self {
            adj,
            n,
            feature_dim,
            eigenvectors: None,
            eigenvalues: None,
        }
    }

    /// Compute normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    fn normalized_laplacian(&self) -> Vec<Vec<f64>> {
        let n = self.n;
        let mut degree = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                degree[i] += self.adj[i][j];
            }
        }
        let d_inv_sqrt: Vec<f64> = degree
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
            .collect();

        let mut lap = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    lap[i][j] = 1.0;
                }
                if self.adj[i][j] != 0.0 {
                    lap[i][j] -= d_inv_sqrt[i] * self.adj[i][j] * d_inv_sqrt[j];
                }
            }
        }
        lap
    }

    /// Compute and cache eigen decomposition using Jacobi method.
    fn ensure_eigen(&mut self) {
        if self.eigenvectors.is_some() {
            return;
        }
        let lap = self.normalized_laplacian();
        let (vals, vecs) = jacobi_eigen(&lap, 100, 1e-10);
        self.eigenvalues = Some(vals);
        self.eigenvectors = Some(vecs);
    }

    /// Spectral convolution: filter features in spectral domain.
    /// features: (n x feature_dim), filter_weights: (n x feature_dim)
    /// Output: filtered features (n x feature_dim)
    pub fn spectral_conv(
        &mut self,
        features: &[Vec<f64>],
        filter_weights: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        assert_eq!(features.len(), self.n);
        assert_eq!(filter_weights.len(), self.n);
        self.ensure_eigen();

        let u = self.eigenvectors.as_ref().unwrap();
        let ut = transpose(u);

        // Transform to spectral domain: \hat{F} = U^T * F
        let f_hat = mat_mul(&ut, features);

        // Apply filter element-wise: \hat{H} = \hat{F} \odot W
        let mut h_hat = f_hat;
        for i in 0..self.n {
            for j in 0..self.feature_dim {
                h_hat[i][j] *= filter_weights[i][j];
            }
        }

        // Transform back: H = U * \hat{H}
        mat_mul(u, &h_hat)
    }

    /// Chebyshev polynomial filter of order K.
    /// Uses recurrence T_k(L) = 2*L_tilde*T_{k-1} - T_{k-2}
    /// coefficients: K+1 coefficients (one per polynomial order 0..=K)
    pub fn chebyshev_filter(
        &mut self,
        features: &[Vec<f64>],
        coefficients: &[f64],
        k: usize,
    ) -> Vec<Vec<f64>> {
        assert_eq!(features.len(), self.n);
        assert!(
            coefficients.len() >= k + 1,
            "Need K+1 coefficients for order K"
        );

        // Compute scaled Laplacian: L_tilde = (2/lambda_max) * L - I
        self.ensure_eigen();
        let lambda_max = self
            .eigenvalues
            .as_ref()
            .unwrap()
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let lambda_max = if lambda_max < 1e-12 { 2.0 } else { lambda_max };

        let lap = self.normalized_laplacian();
        let n = self.n;

        // L_tilde = (2.0 / lambda_max) * L - I
        let mut l_tilde = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                l_tilde[i][j] = (2.0 / lambda_max) * lap[i][j];
                if i == j {
                    l_tilde[i][j] -= 1.0;
                }
            }
        }

        // T_0 = I (identity on features)
        let mut t_prev = features.to_vec(); // T_0 * features = features
        let mut result = vec![vec![0.0; self.feature_dim]; n];

        // Add c_0 * T_0 * features
        for i in 0..n {
            for j in 0..self.feature_dim {
                result[i][j] = coefficients[0] * t_prev[i][j];
            }
        }

        if k == 0 {
            return result;
        }

        // T_1 = L_tilde, so T_1 * features = L_tilde * features
        let mut t_curr = mat_mul(&l_tilde, features);

        // Add c_1 * T_1 * features
        for i in 0..n {
            for j in 0..self.feature_dim {
                result[i][j] += coefficients[1] * t_curr[i][j];
            }
        }

        // Higher orders
        for order in 2..=k {
            // T_order = 2 * L_tilde * T_{order-1} - T_{order-2}
            let l_times_curr = mat_mul(&l_tilde, &t_curr);
            let mut t_next = vec![vec![0.0; self.feature_dim]; n];
            for i in 0..n {
                for j in 0..self.feature_dim {
                    t_next[i][j] = 2.0 * l_times_curr[i][j] - t_prev[i][j];
                }
            }

            for i in 0..n {
                for j in 0..self.feature_dim {
                    result[i][j] += coefficients[order] * t_next[i][j];
                }
            }

            t_prev = t_curr;
            t_curr = t_next;
        }

        result
    }

    /// One round of message passing with learned weight matrix.
    /// features: (n x feature_dim), weight_matrix: (feature_dim x feature_dim)
    pub fn message_passing(
        &self,
        features: &[Vec<f64>],
        weight_matrix: &[Vec<f64>],
        aggregation: Aggregation,
    ) -> Vec<Vec<f64>> {
        assert_eq!(features.len(), self.n);
        assert_eq!(weight_matrix.len(), self.feature_dim);

        let d = self.feature_dim;
        let n = self.n;
        let mut output = vec![vec![0.0; d]; n];

        for i in 0..n {
            // Collect neighbor features
            let mut neighbors: Vec<usize> = Vec::new();
            for j in 0..n {
                if self.adj[i][j] != 0.0 || i == j {
                    neighbors.push(j);
                }
            }

            // Aggregate neighbor features
            let mut agg = vec![0.0; d];
            for &j in &neighbors {
                for k in 0..d {
                    agg[k] += features[j][k];
                }
            }

            match aggregation {
                Aggregation::Mean => {
                    let count = neighbors.len() as f64;
                    if count > 0.0 {
                        for k in 0..d {
                            agg[k] /= count;
                        }
                    }
                }
                Aggregation::Max => {
                    // For max: take element-wise max across neighbors
                    agg = vec![f64::NEG_INFINITY; d];
                    for &j in &neighbors {
                        for k in 0..d {
                            agg[k] = agg[k].max(features[j][k]);
                        }
                    }
                }
                Aggregation::Sum => {}
            }

            // Apply weight matrix: output[i] = W * agg
            for k in 0..d {
                for l in 0..d {
                    output[i][k] += weight_matrix[k][l] * agg[l];
                }
            }
        }

        output
    }

    /// Graph-level readout: aggregate all node features into a single vector.
    pub fn graph_readout(features: &[Vec<f64>], method: ReadoutMethod) -> Vec<f64> {
        if features.is_empty() {
            return Vec::new();
        }
        let d = features[0].len();
        let n = features.len();

        match method {
            ReadoutMethod::Sum => {
                let mut out = vec![0.0; d];
                for node in features {
                    for (j, val) in node.iter().enumerate() {
                        out[j] += val;
                    }
                }
                out
            }
            ReadoutMethod::Mean => {
                let mut out = vec![0.0; d];
                for node in features {
                    for (j, val) in node.iter().enumerate() {
                        out[j] += val;
                    }
                }
                for v in out.iter_mut() {
                    *v /= n as f64;
                }
                out
            }
            ReadoutMethod::Attention(weights) => {
                assert_eq!(weights.len(), n, "Attention weights must match node count");
                let mut out = vec![0.0; d];
                for (i, node) in features.iter().enumerate() {
                    for (j, val) in node.iter().enumerate() {
                        out[j] += weights[i] * val;
                    }
                }
                out
            }
        }
    }

    /// Spectral pooling: group nodes by eigenvector similarity using k-means.
    /// Returns (assignment vector, pooled features matrix of size k x feature_dim)
    pub fn spectral_pool(
        &mut self,
        features: &[Vec<f64>],
        k: usize,
    ) -> (Vec<usize>, Vec<Vec<f64>>) {
        assert!(k > 0 && k <= self.n);
        self.ensure_eigen();

        // Use first few eigenvectors as features for clustering
        let u = self.eigenvectors.as_ref().unwrap();
        let num_eigvecs = k.min(self.n);
        let mut node_spectral = vec![vec![0.0; num_eigvecs]; self.n];
        for i in 0..self.n {
            for j in 0..num_eigvecs {
                node_spectral[i][j] = u[j][i]; // eigenvector j, component i
            }
        }

        // K-means clustering
        let assignment = kmeans(&node_spectral, k, 50);

        // Compute pooled features: mean of features per cluster
        let d = features[0].len();
        let mut pooled = vec![vec![0.0; d]; k];
        let mut counts = vec![0usize; k];

        for (i, &cluster) in assignment.iter().enumerate() {
            for j in 0..d {
                pooled[cluster][j] += features[i][j];
            }
            counts[cluster] += 1;
        }

        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..d {
                    pooled[c][j] /= counts[c] as f64;
                }
            }
        }

        (assignment, pooled)
    }
}

// --- Linear Algebra Utilities ---

fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if a.is_empty() {
        return Vec::new();
    }
    let m = a.len();
    let n = a[0].len();
    let mut t = vec![vec![0.0; m]; n];
    for i in 0..m {
        for j in 0..n {
            t[j][i] = a[i][j];
        }
    }
    t
}

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let n = b[0].len();
    let p = b.len();
    let mut c = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..p {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }
    c
}

/// Jacobi eigenvalue decomposition for symmetric matrices.
/// Returns (eigenvalues, eigenvectors as columns).
fn jacobi_eigen(a: &[Vec<f64>], max_iter: usize, tol: f64) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    let mut v = vec![vec![0.0; n]; n];
    for i in 0..n {
        v[i][i] = 1.0;
    }
    let mut s = a.to_vec();

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if s[i][j].abs() > max_val {
                    max_val = s[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < tol {
            break;
        }

        // Compute rotation
        let app = s[p][p];
        let aqq = s[q][q];
        let apq = s[p][q];

        let theta = if (app - aqq).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Update S
        // We need to update rows/cols p and q
        let mut new_s = s.clone();

        for i in 0..n {
            if i != p && i != q {
                let sip = s[i][p];
                let siq = s[i][q];
                new_s[i][p] = cos_t * sip + sin_t * siq;
                new_s[p][i] = new_s[i][p];
                new_s[i][q] = -sin_t * sip + cos_t * siq;
                new_s[q][i] = new_s[i][q];
            }
        }

        new_s[p][p] = cos_t * cos_t * app + 2.0 * sin_t * cos_t * apq + sin_t * sin_t * aqq;
        new_s[q][q] = sin_t * sin_t * app - 2.0 * sin_t * cos_t * apq + cos_t * cos_t * aqq;
        new_s[p][q] = 0.0;
        new_s[q][p] = 0.0;

        s = new_s;

        // Update eigenvectors
        let mut new_v = v.clone();
        for i in 0..n {
            let vip = v[i][p];
            let viq = v[i][q];
            new_v[i][p] = cos_t * vip + sin_t * viq;
            new_v[i][q] = -sin_t * vip + cos_t * viq;
        }
        v = new_v;
    }

    // Extract eigenvalues
    let mut eigenvalues: Vec<f64> = (0..n).map(|i| s[i][i]).collect();

    // Sort by eigenvalue
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());

    let sorted_vals: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
    // Eigenvectors as columns (row-major representation: vecs[col_index][row_index])
    let sorted_vecs: Vec<Vec<f64>> = indices
        .iter()
        .map(|&col| (0..n).map(|row| v[row][col]).collect())
        .collect();

    (sorted_vals, sorted_vecs)
}

/// K-means clustering.
fn kmeans(data: &[Vec<f64>], k: usize, max_iter: usize) -> Vec<usize> {
    let n = data.len();
    let d = data[0].len();

    // Initialize centroids: pick first k data points (simple but deterministic)
    let mut centroids: Vec<Vec<f64>> = data.iter().take(k).cloned().collect();
    if centroids.len() < k {
        // Pad with zeros if fewer data points than k
        while centroids.len() < k {
            centroids.push(vec![0.0; d]);
        }
    }

    let mut assignment = vec![0usize; n];

    for _ in 0..max_iter {
        // Assign points to nearest centroid
        let mut changed = false;
        for i in 0..n {
            let mut best = 0;
            let mut best_dist = f64::INFINITY;
            for c in 0..k {
                let dist = euclidean_dist_sq(&data[i], &centroids[c]);
                if dist < best_dist {
                    best_dist = dist;
                    best = c;
                }
            }
            if assignment[i] != best {
                changed = true;
                assignment[i] = best;
            }
        }

        if !changed {
            break;
        }

        // Update centroids
        let mut new_centroids = vec![vec![0.0; d]; k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assignment[i];
            counts[c] += 1;
            for j in 0..d {
                new_centroids[c][j] += data[i][j];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..d {
                    new_centroids[c][j] /= counts[c] as f64;
                }
            } else {
                new_centroids[c] = centroids[c].clone();
            }
        }
        centroids = new_centroids;
    }

    assignment
}

fn euclidean_dist_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: 3-node path graph adjacency
    fn path_graph_adj() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
        ]
    }

    fn identity_features(n: usize, d: usize) -> Vec<Vec<f64>> {
        let mut f = vec![vec![0.0; d]; n];
        for i in 0..n.min(d) {
            f[i][i] = 1.0;
        }
        f
    }

    #[test]
    fn test_spectral_conv_output_dim() {
        let adj = path_graph_adj();
        let d = 3;
        let mut gnn = GraphNeural::from_adjacency(adj, d);
        let features = identity_features(3, d);
        let filter = vec![vec![1.0; d]; 3]; // identity filter
        let out = gnn.spectral_conv(&features, &filter);
        assert_eq!(out.len(), 3, "Should have 3 nodes");
        assert_eq!(out[0].len(), d, "Feature dim should match");
    }

    #[test]
    fn test_chebyshev_k0_identity() {
        let adj = path_graph_adj();
        let d = 2;
        let mut gnn = GraphNeural::from_adjacency(adj, d);
        let features = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let coeffs = vec![1.0]; // K=0, coefficient = 1
        let out = gnn.chebyshev_filter(&features, &coeffs, 0);
        // K=0 should return c_0 * features (identity when c_0 = 1)
        for i in 0..3 {
            for j in 0..d {
                assert!(
                    (out[i][j] - features[i][j]).abs() < 1e-6,
                    "K=0 should be identity transform"
                );
            }
        }
    }

    #[test]
    fn test_chebyshev_k1_adds_neighbors() {
        let adj = path_graph_adj();
        let d = 2;
        let mut gnn = GraphNeural::from_adjacency(adj, d);
        let features = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        // K=1: output = c_0 * T_0 * f + c_1 * T_1 * f
        // With c_0=1, c_1=1 this incorporates neighbor information
        let coeffs = vec![1.0, 1.0];
        let out = gnn.chebyshev_filter(&features, &coeffs, 1);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].len(), d);
        // Output should differ from plain features since T_1 adds Laplacian contribution
    }

    #[test]
    fn test_message_passing_updates_all() {
        let adj = path_graph_adj();
        let d = 2;
        let gnn = GraphNeural::from_adjacency(adj, d);
        let features = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let w = vec![vec![1.0, 0.0], vec![0.0, 1.0]]; // identity weight
        let out = gnn.message_passing(&features, &w, Aggregation::Sum);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].len(), d);
        // Node 0 should aggregate self + node 1
        assert!((out[0][0] - (1.0 + 0.0)).abs() < 1e-6); // 1+0 from self+neighbor
        assert!((out[0][1] - (0.0 + 1.0)).abs() < 1e-6); // 0+1 from self+neighbor
    }

    #[test]
    fn test_readout_sum() {
        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let out = GraphNeural::graph_readout(&features, ReadoutMethod::Sum);
        assert_eq!(out, vec![9.0, 12.0]);
    }

    #[test]
    fn test_readout_mean() {
        let features = vec![vec![3.0, 6.0], vec![3.0, 6.0]];
        let out = GraphNeural::graph_readout(&features, ReadoutMethod::Mean);
        assert!((out[0] - 3.0).abs() < 1e-10);
        assert!((out[1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_readout_attention() {
        let features = vec![vec![1.0], vec![3.0]];
        let weights = vec![0.5, 0.5];
        let out = GraphNeural::graph_readout(&features, ReadoutMethod::Attention(weights));
        assert!((out[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_pooling_reduces_nodes() {
        let adj = path_graph_adj();
        let d = 2;
        let mut gnn = GraphNeural::from_adjacency(adj, d);
        let features = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let (assignment, pooled) = gnn.spectral_pool(&features, 2);
        assert_eq!(assignment.len(), 3, "Assignment for each node");
        assert_eq!(pooled.len(), 2, "Pooled to 2 clusters");
        for &c in &assignment {
            assert!(c < 2, "Cluster index in range");
        }
    }

    #[test]
    fn test_full_forward_pass() {
        let adj = path_graph_adj();
        let d = 3;
        let mut gnn = GraphNeural::from_adjacency(adj, d);
        let features = identity_features(3, d);

        // Step 1: spectral conv
        let filter = vec![vec![1.0; d]; 3];
        let h = gnn.spectral_conv(&features, &filter);
        assert_eq!(h.len(), 3);

        // Step 2: message passing
        let w = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];
        let h2 = gnn.message_passing(&h, &w, Aggregation::Sum);
        assert_eq!(h2.len(), 3);

        // Step 3: readout
        let graph_vec = GraphNeural::graph_readout(&h2, ReadoutMethod::Sum);
        assert_eq!(graph_vec.len(), d);
        // Should be finite
        for &v in &graph_vec {
            assert!(v.is_finite(), "Readout should produce finite values");
        }
    }
}
