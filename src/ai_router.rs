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
use crate::math_kernel::{AlphaBatch, NUM_ALPHA_ENGINES};

/// Layer 2: Contextual EXP3 + Beta-Thompson Bandit across 4 Regimes x 86 Engines
#[derive(Debug, Clone, Copy)]
pub struct ContextualBandit86 {
    pub weights: [[f64; NUM_ALPHA_ENGINES]; NUM_REGIMES],
    pub alpha_hits: [[f64; NUM_ALPHA_ENGINES]; NUM_REGIMES],
    pub beta_misses: [[f64; NUM_ALPHA_ENGINES]; NUM_REGIMES],
    pub muted: [[bool; NUM_ALPHA_ENGINES]; NUM_REGIMES],
    pub eta: f64,
    pub gamma: f64,
}

impl ContextualBandit86 {
    pub const fn new() -> Self {
        let init_w = 1.0 / (NUM_ALPHA_ENGINES as f64);
        Self {
            weights: [[init_w; NUM_ALPHA_ENGINES]; NUM_REGIMES],
            alpha_hits: [[2.0; NUM_ALPHA_ENGINES]; NUM_REGIMES],
            beta_misses: [[2.0; NUM_ALPHA_ENGINES]; NUM_REGIMES],
            muted: [[false; NUM_ALPHA_ENGINES]; NUM_REGIMES],
            eta: 0.15,
            gamma: 0.05,
        }
    }

    /// Aggregate all unmuted, non-isolated engines into (alpha_score, confidence, active_count)
    pub fn aggregate(
        &self,
        regime: AiRegime,
        batch: &AlphaBatch,
        isolated: &[bool; NUM_ALPHA_ENGINES],
    ) -> (f64, f64, usize) {
        let r = regime.index();
        let mut weighted_score = 0.0;
        let mut weight_sum = 0.0;
        let mut conf_acc = 0.0;
        let mut active = 0usize;

        for i in 0..NUM_ALPHA_ENGINES {
            if isolated[i] || self.muted[r][i] {
                continue;
            }
            let sig = batch.signals[i];
            if !sig.score.is_finite() || !sig.confidence.is_finite() || sig.confidence <= 1e-6 {
                continue;
            }
            let w_exp3 = self.weights[r][i];
            if !w_exp3.is_finite() || w_exp3 <= 1e-9 {
                continue;
            }

            let a = self.alpha_hits[r][i];
            let b = self.beta_misses[r][i];
            let post_mean = (a / (a + b).max(1e-6)).clamp(0.1, 0.99);
            let eff_w = w_exp3 * post_mean * sig.confidence;

            weighted_score += eff_w * sig.score;
            weight_sum += eff_w;
            conf_acc += sig.confidence;
            active += 1;
        }

        if active == 0 || weight_sum <= 1e-12 {
            return (0.0, 0.0, 0);
        }

        let alpha = (weighted_score / weight_sum).clamp(-1.0, 1.0);
        let conf = (conf_acc / (active as f64)).clamp(0.0, 1.0);
        (alpha, conf, active)
    }

    /// Online EXP3 + Beta posterior update from realized post-fee return (in bps)
    pub fn update_from_outcome(
        &mut self,
        regime: AiRegime,
        prev_scores: &[f64; NUM_ALPHA_ENGINES],
        realized_ret_bps: f64,
        isolated: &[bool; NUM_ALPHA_ENGINES],
    ) {
        if !realized_ret_bps.is_finite() || realized_ret_bps.abs() < 1e-6 {
            return;
        }
        let r = regime.index();
        let mut w_sum = 0.0;
        let mut unmuted_count = 0usize;

        for i in 0..NUM_ALPHA_ENGINES {
            if isolated[i] {
                self.weights[r][i] = 0.0;
                continue;
            }
            let s = prev_scores[i];
            if !s.is_finite() || s.abs() < 0.05 {
                if !self.muted[r][i] {
                    w_sum += self.weights[r][i];
                    unmuted_count += 1;
                }
                continue;
            }

            // Directional edge reward in [-1.0, 1.0]
            let aligned = s * realized_ret_bps;
            let reward = (aligned / 4.0).tanh().clamp(-1.0, 1.0);

            // Exponentially decayed Beta posterior (adapts to non-stationary crypto)
            self.alpha_hits[r][i] = (self.alpha_hits[r][i] * 0.985) + reward.max(0.0);
            self.beta_misses[r][i] = (self.beta_misses[r][i] * 0.985) + (-reward).max(0.0);

            let eff_obs = self.alpha_hits[r][i] + self.beta_misses[r][i];
            let win_rate = self.alpha_hits[r][i] / eff_obs.max(1e-6);

            // Auto-mute persistently failing engines to 0% weight
            if eff_obs >= 12.0 && win_rate < 0.36 {
                self.muted[r][i] = true;
                self.weights[r][i] = 0.0;
                continue;
            }

            self.muted[r][i] = false;
            let mult = (self.eta * reward).clamp(-0.75, 0.75).exp();
            self.weights[r][i] = (self.weights[r][i] * mult).clamp(1e-5, 10.0);
            w_sum += self.weights[r][i];
            unmuted_count += 1;
        }

        // Fallback self-heal: if all engines got muted, reset regime weights uniformly
        if unmuted_count == 0 || w_sum <= 1e-12 {
            let uniform = 1.0 / (NUM_ALPHA_ENGINES as f64);
            for i in 0..NUM_ALPHA_ENGINES {
                self.muted[r][i] = false;
                self.weights[r][i] = if isolated[i] { 0.0 } else { uniform };
            }
            return;
        }

        // Normalize with small EXP3 exploration floor (gamma)
        let floor = self.gamma / (unmuted_count as f64);
        let scale = (1.0 - self.gamma) / w_sum;
        for i in 0..NUM_ALPHA_ENGINES {
            if !isolated[i] && !self.muted[r][i] {
                self.weights[r][i] = (self.weights[r][i] * scale) + floor;
            }
        }
    }

    /// Returns 1-indexed ID of the top-weighted engine in the active regime (for HUD)
    #[inline]
    pub fn top_engine_id(&self, regime: AiRegime) -> u16 {
        let r = regime.index();
        let mut best_i = 0usize;
        let mut best_w = -1.0;
        for i in 0..NUM_ALPHA_ENGINES {
            if !self.muted[r][i] && self.weights[r][i] > best_w {
                best_w = self.weights[r][i];
                best_i = i;
            }
        }
        (best_i + 1) as u16
    }
}
/// Layer 3: Online Actor-Critic RL Controller (Fractional Kelly, SL Width, TP Skew)
#[derive(Debug, Clone, Copy)]
pub struct ActorCriticController {
    pub value_est: [f64; NUM_REGIMES],      // Critic baseline expected reward (bps)
    pub kelly_param: [f64; NUM_REGIMES],    // Actor: Fractional Kelly [0.02, 0.50]
    pub sl_param: [f64; NUM_REGIMES],       // Actor: SL vol multiplier [0.80, 3.00]
    pub tp_param: [f64; NUM_REGIMES],       // Actor: TP skew multiplier [0.50, 2.50]
    pub lr_critic: f64,
    pub lr_actor: f64,
}

impl ActorCriticController {
    pub const fn new() -> Self {
        Self {
            value_est: [0.0; NUM_REGIMES],
            kelly_param: [0.15, 0.25, 0.12, 0.04],
            sl_param: [1.20, 1.80, 2.20, 1.00],
            tp_param: [1.10, 2.00, 1.60, 0.80],
            lr_critic: 0.10,
            lr_actor: 0.04,
        }
    }

    /// Returns RL-tuned (kelly_fraction, sl_mult, tp_skew) scaled by Watchdog drawdown factor
    #[inline]
    pub fn policy(&self, regime: AiRegime, size_scale: f64) -> (f64, f64, f64) {
        let r = regime.index();
        let scale = if size_scale.is_finite() { size_scale.clamp(0.0, 1.0) } else { 0.0 };
        let kelly = (self.kelly_param[r] * scale).clamp(0.0, 0.50);
        let sl = self.sl_param[r].clamp(0.80, 3.00);
        let tp = self.tp_param[r].clamp(0.50, 2.50);
        (kelly, sl, tp)
    }

    /// Online Actor-Critic update from realized post-fee trade outcome (bps)
    pub fn update_from_trade(
        &mut self,
        regime: AiRegime,
        post_fee_pnl_bps: f64,
        hit_sl: bool,
        hit_tp: bool,
    ) {
        if !post_fee_pnl_bps.is_finite() {
            return;
        }
        let r = regime.index();
        let reward = (post_fee_pnl_bps / 10.0).clamp(-2.0, 2.0);
        let td_error = reward - self.value_est[r];

        // Critic update
        self.value_est[r] = (self.value_est[r] + (self.lr_critic * td_error)).clamp(-2.0, 2.0);

        // Actor update: Kelly sizing rises on positive TD-error, contracts on negative
        let next_kelly = self.kelly_param[r] + (self.lr_actor * 0.25 * td_error);
        self.kelly_param[r] = next_kelly.clamp(0.02, 0.50);

        // Actor update: SL widens slightly if whipsawed by SL on negative TD, tightens on clean wins
        let sl_grad = if hit_sl && td_error < 0.0 {
            0.08
        } else if hit_tp && td_error > 0.0 {
            -0.03
        } else {
            -0.02 * td_error
        };
        self.sl_param[r] = (self.sl_param[r] + (self.lr_actor * sl_grad * 4.0)).clamp(0.80, 3.00);

        // Actor update: TP skew expands on positive momentum wins, contracts on reversals
        let tp_grad = if hit_tp && td_error > 0.0 {
            0.10
        } else if td_error < 0.0 {
            -0.08
        } else {
            0.03 * td_error
        };
        self.tp_param[r] = (self.tp_param[r] + (self.lr_actor * tp_grad * 4.0)).clamp(0.50, 2.50);
    }
}

/// Self-Healing Fault Isolator, Drawdown Halver & Stale-Feed / Daily-Loss Safe-Mode Watchdog
#[derive(Debug, Clone, Copy)]
pub struct SelfHealingWatchdog {
    pub isolated: [bool; NUM_ALPHA_ENGINES],
    pub fault_counts: [u16; NUM_ALPHA_ENGINES],
    pub loss_streak: u32,
    pub size_scale: f64,
    pub stale_limit_ms: u64,
    pub max_daily_loss_pct: f64,
    pub daily_loss_breached: bool,
    pub feed_stale: bool,
    pub safe_mode: bool,
}

impl SelfHealingWatchdog {
    pub const fn new(stale_limit_ms: u64, max_daily_loss_pct: f64) -> Self {
        Self {
            isolated: [false; NUM_ALPHA_ENGINES],
            fault_counts: [0; NUM_ALPHA_ENGINES],
            loss_streak: 0,
            size_scale: 1.0,
            stale_limit_ms,
            max_daily_loss_pct,
            daily_loss_breached: false,
            feed_stale: false,
            safe_mode: false,
        }
    }

    /// Inspect every engine output for NaN/Inf/out-of-bounds and isolate repeat offenders
    pub fn inspect_batch(&mut self, batch: &AlphaBatch) {
        for i in 0..NUM_ALPHA_ENGINES {
            let sig = batch.signals[i];
            let faulty = !sig.score.is_finite()
                || !sig.confidence.is_finite()
                || sig.score < -1.0001
                || sig.score > 1.0001
                || sig.confidence < -0.0001
                || sig.confidence > 1.0001;

            if faulty {
                self.fault_counts[i] = self.fault_counts[i].saturating_add(1);
                if self.fault_counts[i] >= 3 {
                    self.isolated[i] = true;
                }
            }
        }
    }

    /// Halve position size on consecutive loss streaks and lock Safe-Mode on daily loss breach
    pub fn record_trade_outcome(&mut self, post_fee_pnl_bps: f64, daily_pnl_pct: f64) {
        if post_fee_pnl_bps.is_finite() {
            if post_fee_pnl_bps < 0.0 {
                self.loss_streak = self.loss_streak.saturating_add(1);
                let exp = self.loss_streak.min(4) as i32;
                self.size_scale = (0.5_f64).powi(exp).clamp(0.125, 1.0);
            } else if post_fee_pnl_bps > 0.0 {
                self.loss_streak = 0;
                self.size_scale = (self.size_scale * 1.5).clamp(0.25, 1.0);
            }
        }

        if daily_pnl_pct.is_finite() && daily_pnl_pct <= -self.max_daily_loss_pct.abs() {
            self.daily_loss_breached = true;
        }
        self.recompute_safe_mode();
    }

    /// Check >3s stale market data feed and update Safe-Mode flag
    pub fn check_feed_health(&mut self, now_ms: u64, last_feed_ms: u64) -> bool {
        let age = now_ms.saturating_sub(last_feed_ms);
        self.feed_stale = last_feed_ms == 0 || age > self.stale_limit_ms;
        self.recompute_safe_mode();
        self.safe_mode
    }

    #[inline]
    pub fn isolated_count(&self) -> usize {
        let mut count = 0usize;
        for &flag in self.isolated.iter() {
            if flag {
                count += 1;
            }
        }
        count
    }

    #[inline]
    fn recompute_safe_mode(&mut self) {
        let too_many_faults = self.isolated_count() >= (NUM_ALPHA_ENGINES / 2);
        self.safe_mode = self.feed_stale || self.daily_loss_breached || too_many_faults;
    }
}
