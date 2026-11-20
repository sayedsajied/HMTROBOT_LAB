#![allow(dead_code)]

use crate::types::SymbolId;

pub const NUM_REGIMES: usize = 4;

/// 4-State Market Regime classified by Layer-1 Hidden Markov Model (HMM)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AiRegime {
    CalmRange = 0,
    Trending = 1,
    VolatileBurst = 2,
    ToxicCrisis = 3,
}

impl AiRegime {
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub const fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::CalmRange,
            1 => Self::Trending,
            2 => Self::VolatileBurst,
            _ => Self::ToxicCrisis,
        }
    }
}

/// Zero-allocation output produced by the 3-Layer Adaptive AI Router every tick
#[derive(Debug, Clone, Copy)]
pub struct AiDecision {
    pub symbol_id: SymbolId,
    pub regime: AiRegime,
    pub regime_probs: [f64; NUM_REGIMES],
    pub alpha_score: f64,      // [-1.0, 1.0] Bandit-weighted consensus
    pub confidence: f64,       // [0.0, 1.0]
    pub kelly_fraction: f64,   // [0.0, 0.5] RL-tuned Fractional Kelly
    pub sl_mult: f64,          // [0.8, 3.0] RL-tuned Stop-Loss multiplier
    pub tp_skew: f64,          // [0.5, 2.5] RL-tuned Take-Profit skew
    pub safe_mode: bool,       // True if Watchdog halts new entries
    pub active_engines: usize,
    pub isolated_engines: usize,
}

impl AiDecision {
    pub const fn neutral(symbol_id: SymbolId, safe_mode: bool) -> Self {
        Self {
            symbol_id,
            regime: AiRegime::CalmRange,
            regime_probs: [1.0, 0.0, 0.0, 0.0],
            alpha_score: 0.0,
            confidence: 0.0,
            kelly_fraction: 0.0,
            sl_mult: 1.5,
            tp_skew: 1.5,
            safe_mode,
            active_engines: 0,
            isolated_engines: 0,
        }
    }
}

/// Layer 1: Pure-Rust 4-State Hidden Markov Model (HMM) Regime Detector
#[derive(Debug, Clone, Copy)]
pub struct HmmRegimeDetector {
    pub probs: [f64; NUM_REGIMES],
    pub current: AiRegime,
}

impl HmmRegimeDetector {
    /// Sticky Markov transition matrix (rows = prev state, cols = next state)
    const TRANS: [[f64; NUM_REGIMES]; NUM_REGIMES] = [
        [0.88, 0.07, 0.04, 0.01], // CalmRange
        [0.06, 0.86, 0.06, 0.02], // Trending
        [0.04, 0.08, 0.80, 0.08], // VolatileBurst
        [0.02, 0.03, 0.15, 0.80], // ToxicCrisis
    ];

    pub const fn new() -> Self {
        Self {
            probs: [0.55, 0.25, 0.15, 0.05],
            current: AiRegime::CalmRange,
        }
    }

    #[inline]
    fn gauss_log_lik(x: f64, mean: f64, std: f64) -> f64 {
        let z = (x - mean) / std.max(1e-6);
        -0.5 * z * z
    }

    /// Bayesian forward-filter update from (vol_bps, |trend|, spread_bps, dispersion)
    pub fn update(&mut self, vol_bps: f64, trend_abs: f64, spread_bps: f64, dispersion: f64) -> AiRegime {
        let v = if vol_bps.is_finite() { vol_bps.clamp(0.0, 50.0) } else { 1.0 };
        let t = if trend_abs.is_finite() { trend_abs.clamp(0.0, 1.0) } else { 0.0 };
        let s = if spread_bps.is_finite() { spread_bps.clamp(0.0, 25.0) } else { 0.5 };
        let d = if dispersion.is_finite() { dispersion.clamp(0.0, 1.0) } else { 0.3 };

        // 1. Predict step: prior = probs * TRANS
        let mut prior = [0.0_f64; NUM_REGIMES];
        for j in 0..NUM_REGIMES {
            let mut acc = 0.0;
            for i in 0..NUM_REGIMES {
                acc += self.probs[i] * Self::TRANS[i][j];
            }
            prior[j] = acc.max(1e-9);
        }

        // 2. Emission log-likelihoods for [CalmRange, Trending, VolatileBurst, ToxicCrisis]
        let log_em = [
            Self::gauss_log_lik(v, 0.8, 0.9) + Self::gauss_log_lik(t, 0.15, 0.20) + Self::gauss_log_lik(s, 0.4, 0.6),
            Self::gauss_log_lik(v, 2.2, 1.5) + Self::gauss_log_lik(t, 0.75, 0.25) + Self::gauss_log_lik(d, 0.25, 0.20),
            Self::gauss_log_lik(v, 6.0, 3.0) + Self::gauss_log_lik(t, 0.50, 0.35) + Self::gauss_log_lik(s, 1.8, 1.5),
            Self::gauss_log_lik(v, 14.0, 6.0) + Self::gauss_log_lik(s, 5.5, 3.0) + Self::gauss_log_lik(d, 0.85, 0.25),
        ];

        // 3. Log-sum-exp stabilized Bayesian posterior
        let mut log_post = [0.0_f64; NUM_REGIMES];
        let mut max_lp = f64::NEG_INFINITY;
        for j in 0..NUM_REGIMES {
            log_post[j] = prior[j].ln() + log_em[j];
            if log_post[j] > max_lp {
                max_lp = log_post[j];
            }
        }

        let mut sum = 0.0;
        for j in 0..NUM_REGIMES {
            self.probs[j] = (log_post[j] - max_lp).exp();
            sum += self.probs[j];
        }

        let mut best_idx = 0usize;
        let mut best_p = -1.0;
        let inv_sum = 1.0 / sum.max(1e-12);
        for j in 0..NUM_REGIMES {
            self.probs[j] *= inv_sum;
            if self.probs[j] > best_p {
                best_p = self.probs[j];
                best_idx = j;
            }
        }

        self.current = AiRegime::from_index(best_idx);
        self.current
    }
}
