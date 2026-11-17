// Project : nQIST
// File    : src/xca/topology.rs
// Purpose : connected chord-diagram topologies with fermionic parity
//           for the arbitrary-order (XCA) hybridization expansion
//
// Port of TRIQS/xca python/triqs_xca/diag.py (`all_pairings`,
// `is_crossing`, `is_connected`, pinned SHA 35b0c59): the pairings of
// the 2m backbone vertices with the crossing-connectivity filter are
// the skeleton topologies of order m (connected chord diagrams,
// C(1)=1, C(2)=1, C(3)=4, C(4)=27, C(5)=248), and the recursion
// parity is the diagram sign used together with the per-channel
// statistics factors.

/// One skeleton topology: `m` hybridization lines pairing the
/// `2m` backbone vertices, plus the fermionic parity sign (+-1).
///
/// `pairs[0]` always contains vertex 0 (the line eliminated at the
/// time-zero vertex for self-energy diagrams; the external operator
/// pair for single-particle Green's function diagrams).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Topology {
    pub pairs: Vec<(usize, usize)>,
    /// Fermionic Wick parity of the pairing (+-1, diag.py `_recursive_pairing`).
    pub parity: f64,
}

/// Strict interleave test of two vertex pairs (diag.py `is_crossing`).
fn is_crossing(p1: (usize, usize), p2: (usize, usize)) -> bool {
    (p1.0 < p2.0 && p2.0 < p1.1 && p1.1 < p2.1) || (p2.0 < p1.0 && p1.0 < p2.1 && p2.1 < p1.1)
}

/// Crossing-connectivity of a pairing (diag.py `is_connected` with
/// `ks = [1]`): a pair is a connector iff it touches vertex 0, and
/// pairs propagate connectivity by crossing.
fn is_connected(pairs: &[(usize, usize)]) -> bool {
    let mut stack: Vec<(usize, usize)> = Vec::new();
    let mut rest: Vec<(usize, usize)> = Vec::new();
    for &p in pairs {
        if p.0 == 0 {
            stack.push(p);
        } else {
            rest.push(p);
        }
    }
    while let Some(pc) = stack.pop() {
        let mut i = 0;
        while i < rest.len() {
            if is_crossing(pc, rest[i]) {
                stack.push(rest.remove(i));
            } else {
                i += 1;
            }
        }
    }
    rest.is_empty()
}

/// Recursive pairing of `unpaired` (diag.py `_recursive_pairing`):
/// pair element 0 with element `i` and flip the parity when `i` is
/// even (the fermionic permutation sign of moving the two operators
/// together).
fn recurse(
    parity: f64,
    partial: &mut Vec<(usize, usize)>,
    unpaired: &[usize],
    out: &mut Vec<Topology>,
) {
    if unpaired.is_empty() {
        let pairs = partial.clone();
        if is_connected(&pairs) {
            out.push(Topology { pairs, parity });
        }
        return;
    }
    for i in 1..unpaired.len() {
        let p = (unpaired[0], unpaired[i]);
        let new_parity = if i % 2 == 0 { -parity } else { parity };
        let mut rest: Vec<usize> = Vec::with_capacity(unpaired.len() - 2);
        rest.extend_from_slice(&unpaired[1..i]);
        rest.extend_from_slice(&unpaired[i + 1..]);
        partial.push(p);
        recurse(new_parity, partial, &rest, out);
        partial.pop();
    }
}

/// All connected chord-diagram topologies of order `m` with their
/// parity signs (`all_connected_pairings(m)`).
pub(crate) fn connected_pairings(m: usize) -> Vec<Topology> {
    assert!(m >= 1, "connected_pairings: m must be >= 1");
    let unpaired: Vec<usize> = (0..2 * m).collect();
    let mut out = Vec::new();
    recurse(1.0, &mut Vec::new(), &unpaired, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_counts_match_connected_chord_diagrams() {
        // OEIS A000699: connected chord diagrams 1, 1, 4, 27, 248
        let counts = [1usize, 1, 4, 27, 248];
        for (m, &want) in counts.iter().enumerate() {
            let got = connected_pairings(m + 1).len();
            assert_eq!(got, want, "order {}", m + 1);
        }
    }

    #[test]
    fn order3_topologies_and_signs_match_reference() {
        // Hand-verified against TRIQS/xca examples/dense_backbone_third_order.cpp
        let topos = connected_pairings(3);
        let mut table: std::collections::HashSet<(Vec<(usize, usize)>, i8)> = topos
            .iter()
            .map(|t| (t.pairs.clone(), t.parity as i8))
            .collect();
        let expect: [(Vec<(usize, usize)>, i8); 4] = [
            (vec![(0, 2), (1, 4), (3, 5)], 1),
            (vec![(0, 3), (1, 5), (2, 4)], 1),
            (vec![(0, 4), (1, 3), (2, 5)], 1),
            (vec![(0, 3), (1, 4), (2, 5)], -1),
        ];
        for (pairs, parity) in expect {
            let key = (pairs.clone(), parity);
            assert!(table.remove(&key), "missing {pairs:?} {parity}");
        }
        assert!(table.is_empty(), "unexpected extra topologies");
    }

    #[test]
    fn first_pair_touches_vertex_zero() {
        for m in 1..=4usize {
            for t in connected_pairings(m) {
                assert_eq!(t.pairs[0].0, 0, "order {m}");
                assert_eq!(t.pairs.len(), m);
            }
        }
    }
}
