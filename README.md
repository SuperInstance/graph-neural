# graph-neural

**Spectral graph neural network primitives — normalized Laplacian, spectral filters, and graph-level readout in pure Rust.**

Implements the core building blocks for spectral GNNs: normalized Laplacian computation (L = I - D⁻½AD⁻½), eigendecomposition via power iteration with deflation, spectral convolution filters, message-passing aggregation, and graph-level readout with attention.

## What This Gives You

- **Normalized Laplacian** — L = I - D⁻½AD⁻½, eigenvalues in [0, 2]
- **Spectral convolution** — filter node features in the eigenbasis
- **Message passing** — sum, mean, or max aggregation over neighborhoods
- **Graph readout** — sum, mean, or attention-weighted pooling to graph-level representation
- **Power iteration eigendecomposition** — no external dependencies
- **Cheeger constant estimation** — from Fiedler vector

## Quick Start

```rust
use graph_neural::{GraphNeural, Aggregation, ReadoutMethod};

let adj = vec![
    vec![0.0, 1.0, 1.0],
    vec![1.0, 0.0, 1.0],
    vec![1.0, 1.0, 0.0],
];

let mut gnn = GraphNeural::from_adjacency(adj, 4); // 4-dim features

// Spectral filter
let filtered = gnn.spectral_filter(&features, |lambda| (1.0 - lambda).max(0.0));

// Message passing
let aggregated = gnn.message_pass(&features, Aggregation::Mean);

// Graph-level readout
let graph_vec = gnn.readout(&features, ReadoutMethod::Attention(weights));
```

## API Reference

### `GraphNeural`

| Method | Description |
|--------|-------------|
| `from_adjacency(adj, feature_dim)` | Create from adjacency matrix |
| `spectral_filter(features, kernel)` | Filter features in eigenbasis |
| `message_pass(features, aggregation)` | Aggregate neighborhood features |
| `readout(features, method)` | Pool node features to graph vector |
| `eigenvalues()` | Cached eigenvalue computation |
| `fiedler_vector()` | Algebraic connectivity eigenvector |
| `cheeger_constant()` | Graph cut estimate |

### Aggregation

`Sum` · `Mean` · `Max`

### ReadoutMethod

`Sum` · `Mean` · `Attention(Vec<f64>)`

## Testing

```bash
cargo test
cargo run
```

## Installation

```toml
[dependencies]
graph-neural = { git = "https://github.com/SuperInstance/graph-neural" }
```

## How It Fits

Part of the SuperInstance ecosystem:

- **graph-neural** — Spectral GNN primitives (this repo)
- **[heat-spectral](https://github.com/SuperInstance/heat-spectral)** — Heat diffusion on graphs
- **[wave-conservation](https://github.com/SuperInstance/wave-conservation)** — Wave propagation on graphs

## License

MIT
