//! Integration tests for graph-neural

use graph_neural::*;

fn triangle_adj() -> Vec<Vec<f64>> {
    vec![
        vec![0.0, 1.0, 1.0],
        vec![1.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0],
    ]
}

fn path_adj() -> Vec<Vec<f64>> {
    vec![
        vec![0.0, 1.0, 0.0],
        vec![1.0, 0.0, 1.0],
        vec![0.0, 1.0, 0.0],
    ]
}

#[test]
fn test_from_adjacency_no_panic() {
    let _gnn = GraphNeural::from_adjacency(triangle_adj(), 2);
}

#[test]
#[should_panic(expected = "Adjacency matrix must not be empty")]
fn test_empty_adjacency_panics() {
    let _ = GraphNeural::from_adjacency(vec![], 2);
}

#[test]
fn test_spectral_conv_output_shape() {
    let mut gnn = GraphNeural::from_adjacency(triangle_adj(), 4);
    let features = vec![vec![1.0, 0.0, 0.0, 0.0]; 3];
    let filter = vec![vec![1.0; 4]; 3];
    let out = gnn.spectral_conv(&features, &filter);
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].len(), 4);
}

#[test]
fn test_message_passing_sum() {
    let gnn = GraphNeural::from_adjacency(triangle_adj(), 2);
    let features = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let w = vec![vec![1.0, 0.0], vec![0.0, 1.0]]; // identity
    let out = gnn.message_passing(&features, &w, Aggregation::Sum);
    assert_eq!(out.len(), 3);
    assert!((out[0][0] - 2.0).abs() < 1e-6, "expected 2.0 got {}", out[0][0]);
    assert!((out[0][1] - 2.0).abs() < 1e-6, "expected 2.0 got {}", out[0][1]);
}

#[test]
fn test_message_passing_mean() {
    let gnn = GraphNeural::from_adjacency(triangle_adj(), 2);
    let features = vec![vec![3.0, 0.0], vec![0.0, 3.0], vec![3.0, 3.0]];
    let w = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let out = gnn.message_passing(&features, &w, Aggregation::Mean);
    assert!((out[0][0] - 2.0).abs() < 1e-6);
    assert!((out[0][1] - 2.0).abs() < 1e-6);
}

#[test]
fn test_chebyshev_k0() {
    let mut gnn = GraphNeural::from_adjacency(path_adj(), 2);
    let features = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    let out = gnn.chebyshev_filter(&features, &[2.0], 0);
    for i in 0..3 {
        for j in 0..2 {
            assert!((out[i][j] - 2.0 * features[i][j]).abs() < 1e-6);
        }
    }
}

#[test]
fn test_graph_readout_sum() {
    let features = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let out = GraphNeural::graph_readout(&features, ReadoutMethod::Sum);
    assert_eq!(out, vec![4.0, 6.0]);
}

#[test]
fn test_graph_readout_mean() {
    let features = vec![vec![2.0, 4.0], vec![6.0, 8.0]];
    let out = GraphNeural::graph_readout(&features, ReadoutMethod::Mean);
    assert!((out[0] - 4.0).abs() < 1e-10);
    assert!((out[1] - 6.0).abs() < 1e-10);
}

#[test]
fn test_graph_readout_attention() {
    let features = vec![vec![2.0], vec![4.0], vec![6.0]];
    let weights = vec![0.0, 1.0, 0.0];
    let out = GraphNeural::graph_readout(&features, ReadoutMethod::Attention(weights));
    assert!((out[0] - 4.0).abs() < 1e-10);
}

#[test]
fn test_spectral_pool() {
    let mut gnn = GraphNeural::from_adjacency(triangle_adj(), 2);
    let features = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
    let (assignment, pooled) = gnn.spectral_pool(&features, 2);
    assert_eq!(assignment.len(), 3);
    assert_eq!(pooled.len(), 2);
    for &c in &assignment {
        assert!(c < 2);
    }
}
