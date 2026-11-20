// Mixed fermion/boson channel plumbing for the real-time XCA
// evaluator (#5 Holstein m >= 3 groundwork): the channel view, the
// mixed-sector chain builder (boson lines are same-sector diagonal
// vertices), and the boson Matsubara-kernel access used by the
// chord branches. The statistical rules mirror the equilibrium
// anchors: boson channels carry NO backward minus, the topology
// parity stays the pure pairing parity, and the per-line statistics
// factor only fires for backward FERMION channels.
use crate::hilbert::HilbertSpace;
use crate::nca::Hybridization;
use kadanoffbaym::linalg::DenseMat;
use kadanoffbaym::types::ContourGreen;

use super::evalm::RtChain;

/// A real-time evaluation channel: either a fermionic
/// hybridization (the flavor-operator sector walks of the
/// fermionic machinery) or a bosonic phonon line (the
/// same-sector diagonal `docc` vertex blocks).
pub(crate) enum RtChan<'a> {
    Fermion(&'a Hybridization<'a>),
    Boson {
        delta: &'a ContourGreen,
        backward: bool,
        blocks: &'a [(usize, DenseMat)],
    },
}

impl RtChan<'_> {
    #[inline]
    pub(crate) fn delta(&self) -> &ContourGreen {
        match self {
            RtChan::Fermion(h) => h.delta,
            RtChan::Boson { delta, .. } => delta,
        }
    }
    #[inline]
    pub(crate) fn backward(&self) -> bool {
        match self {
            RtChan::Fermion(h) => h.backward,
            RtChan::Boson { backward, .. } => *backward,
        }
    }
    /// Boson lines never carry the fermionic backward minus.
    #[inline]
    pub(crate) fn fermion(&self) -> bool {
        matches!(self, RtChan::Fermion(_))
    }
}

/// The 4 fermion channels extended with the optional bath's boson
/// channels (forward `d0`, backward `d0_cc`) - the real-time mirror
/// of `oca::channels_with_bath`.
pub(crate) fn chans_with_bath<'a>(
    pp_ints: &'a [Hybridization<'a>],
    bath: Option<&'a crate::nca::BosonBath>,
) -> Vec<RtChan<'a>> {
    let mut chans: Vec<RtChan<'a>> = pp_ints.iter().map(|h| RtChan::Fermion(h)).collect();
    if let Some(b) = bath {
        chans.push(RtChan::Boson {
            delta: &b.d0,
            backward: false,
            blocks: &b.blocks,
        });
        chans.push(RtChan::Boson {
            delta: &b.d0_cc,
            backward: true,
            blocks: &b.blocks,
        });
    }
    chans
}

/// Walk the vertices for channel assignment `cs` from sector `s0`;
/// `None` when an operator block is missing or the chain does not
/// close. Fermion vertices transition sectors through the flavor
/// ops exactly as `build_rt_chain`; boson vertices are SAME-SECTOR
/// diagonal insertions (the docc block of the current sector; a
/// sector without a block does not close the chain). Scalar
/// prototype scope as the fermionic builder.
pub(crate) fn build_rt_chain_mixed(
    hs: &HilbertSpace,
    pairs: &[(usize, usize)],
    cs: &[usize],
    chans: &[RtChan],
    s0: usize,
) -> Option<RtChain> {
    let m = pairs.len();
    let nv = 2 * m;
    let mut owner = vec![(0usize, false); nv];
    for (li, &(a, b)) in pairs.iter().enumerate() {
        owner[a] = (li, false);
        owner[b] = (li, true);
    }
    let mut sectors = vec![0usize; nv + 1];
    sectors[0] = s0;
    let mut ops = Vec::with_capacity(nv);
    for v in 0..nv {
        let (li, later) = owner[v];
        match &chans[cs[li]] {
            RtChan::Fermion(h) => {
                let flav = if later { h.op_cre_fl } else { h.op_ann_fl };
                let (target, blk) = hs.ops[flav].block(sectors[v])?;
                if blk.raw().len() != 1 {
                    // scalar prototype scope
                    return None;
                }
                sectors[v + 1] = *target;
                ops.push(blk.get(0, 0));
            }
            RtChan::Boson { blocks, .. } => {
                let ss = sectors[v];
                let Some((_, blk)) = blocks.iter().find(|(s, _)| *s == ss) else {
                    return None;
                };
                if blk.raw().len() != 1 {
                    // scalar prototype scope
                    return None;
                }
                sectors[v + 1] = ss;
                ops.push(blk.get(0, 0));
            }
        }
    }
    if sectors[nv] != s0 {
        return None;
    }
    Some(RtChain { sectors, ops })
}

/// MATRIX walk of [`build_rt_chain_mixed`]: identical sector
/// semantics (fermion vertices transition through the flavor ops,
/// boson vertices are same-sector diagonal insertions) but storing
/// the FULL operator blocks. The scalar builder stays the all-1x1
/// route.
pub(crate) fn build_rt_chain_mixed_m(
    hs: &HilbertSpace,
    pairs: &[(usize, usize)],
    cs: &[usize],
    chans: &[RtChan],
    s0: usize,
) -> Option<super::evalm::RtChainM> {
    let m = pairs.len();
    let nv = 2 * m;
    let mut owner = vec![(0usize, false); nv];
    for (li, &(a, b)) in pairs.iter().enumerate() {
        owner[a] = (li, false);
        owner[b] = (li, true);
    }
    let mut sectors = vec![0usize; nv + 1];
    sectors[0] = s0;
    let mut ops = Vec::with_capacity(nv);
    for v in 0..nv {
        let (li, later) = owner[v];
        match &chans[cs[li]] {
            RtChan::Fermion(h) => {
                let flav = if later { h.op_cre_fl } else { h.op_ann_fl };
                let (target, blk) = hs.ops[flav].block(sectors[v])?;
                sectors[v + 1] = *target;
                ops.push(blk.clone());
            }
            RtChan::Boson { blocks, .. } => {
                let ss = sectors[v];
                let Some((_, blk)) = blocks.iter().find(|(s, _)| *s == ss) else {
                    return None;
                };
                sectors[v + 1] = ss;
                ops.push(blk.clone());
            }
        }
    }
    if sectors[nv] != s0 {
        return None;
    }
    Some(super::evalm::RtChainM { sectors, ops })
}

/// Bosonic Matsubara kernel B(tau, w) = -e^{-w tau} / (1 - e^{-b w})
/// (overflow-safe; the reflected branch for w < 0). Statistical
/// limit: b*w*B -> -1 at w -> 0 (the Bose pole; NOT the fermion
/// K's finite -1/2).
pub(crate) fn boson_k_tau(tau: f64, w: f64, beta: f64) -> f64 {
    if w >= 0.0 {
        -(-w * tau).exp() / (1.0 - (-beta * w).exp())
    } else {
        -(-w * (tau - beta)).exp() / ((beta * w).exp() - 1.0)
    }
}
