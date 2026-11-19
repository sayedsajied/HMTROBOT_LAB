#![allow(dead_code)]

use crate::types::{FundingBasisSnapshot, L2OrderBook, OhlcvBar, TradeTick};

/// Stack-allocated fixed-size rolling window (zero heap allocation, O(1) push)
#[derive(Debug, Clone, Copy)]
pub struct RollingWindow<const N: usize> {
    buf: [f64; N],
    cursor: usize,
    count: usize,
}

impl<const N: usize> RollingWindow<N> {
    pub const fn new() -> Self {
        Self {
            buf: [0.0; N],
            cursor: 0,
            count: 0,
        }
    }

    #[inline]
    pub fn push(&mut self, value: f64) {
        if N == 0 || !value.is_finite() {
            return;
        }
        self.buf[self.cursor] = value;
        self.cursor = (self.cursor + 1) % N;
        if self.count < N {
            self.count += 1;
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[inline]
    pub fn latest(&self) -> Option<f64> {
        if self.count == 0 || N == 0 {
            return None;
        }
        let idx = if self.cursor == 0 { N - 1 } else { self.cursor - 1 };
        Some(self.buf[idx])
    }

    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let mut sum = 0.0;
        for i in 0..self.count {
            sum += self.buf[i];
        }
        let m = sum / (self.count as f64);
        if m.is_finite() { m } else { 0.0 }
    }

    pub fn std_dev(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let mut acc = 0.0;
        for i in 0..self.count {
            let d = self.buf[i] - mean;
            acc += d * d;
        }
        let var = acc / ((self.count - 1) as f64);
        if var.is_finite() && var > 0.0 {
            var.sqrt()
        } else {
            0.0
        }
    }

    /// Z-score of `value` relative to the rolling window, clamped to [-6.0, 6.0]
    #[inline]
    pub fn z_score(&self, value: f64) -> f64 {
        if !value.is_finite() || self.count < 2 {
            return 0.0;
        }
        let sd = self.std_dev();
        if !sd.is_finite() || sd <= 1e-12 {
            return 0.0;
        }
        ((value - self.mean()) / sd).clamp(-6.0, 6.0)
    }
}

/// Real-time EWMA realized volatility estimator in basis points (bps)
#[derive(Debug, Clone, Copy)]
pub struct EwmaVolatility {
    alpha: f64,
    prev_price: f64,
    ewma_var_bps2: f64,
    last_bar_range_bps: f64,
    samples: u64,
}

impl EwmaVolatility {
    pub const fn new(alpha: f64) -> Self {
        Self {
            alpha,
            prev_price: 0.0,
            ewma_var_bps2: 0.0,
            last_bar_range_bps: 0.0,
            samples: 0,
        }
    }

    #[inline]
    pub fn on_price(&mut self, price: f64) -> f64 {
        if !price.is_finite() || price <= 0.0 {
            return self.vol_bps();
        }
        if self.prev_price > 0.0 {
            let ret_bps = ((price - self.prev_price) / self.prev_price) * 10_000.0;
            if ret_bps.is_finite() {
                let sq = (ret_bps * ret_bps).min(2_500_000.0);
                let a = self.alpha.clamp(0.01, 0.95);
                if self.samples == 0 {
                    self.ewma_var_bps2 = sq;
                } else {
                    self.ewma_var_bps2 = (a * sq) + ((1.0 - a) * self.ewma_var_bps2);
                }
                self.samples = self.samples.saturating_add(1);
            }
        }
        self.prev_price = price;
        self.vol_bps()
    }

    #[inline]
    pub fn on_bar(&mut self, bar: &OhlcvBar) {
        let r = bar.range_bps();
        if r.is_finite() && r >= 0.0 {
            self.last_bar_range_bps = r;
        }
    }

    #[inline]
    pub fn vol_bps(&self) -> f64 {
        if self.ewma_var_bps2.is_finite() && self.ewma_var_bps2 > 0.0 {
            self.ewma_var_bps2.sqrt()
        } else {
            0.0
        }
    }

    /// Combines tick EWMA volatility and latest bar range for Vol-Spike Veto
    #[inline]
    pub fn effective_vol_bps(&self) -> f64 {
        self.vol_bps().max(self.last_bar_range_bps * 0.5)
    }
}

/// 1D Institutional Kalman Filter anchoring Microprice, Trade Tape & Basis into Fair Value
#[derive(Debug, Clone, Copy)]
pub struct KalmanFairValue {
    estimate: f64,
    error_cov: f64,
    process_noise_q: f64,
    meas_noise_r: f64,
    initialized: bool,
    last_ts_ms: u64,
}

impl KalmanFairValue {
    pub const fn new(process_noise_q: f64, meas_noise_r: f64) -> Self {
        Self {
            estimate: 0.0,
            error_cov: 1.0,
            process_noise_q,
            meas_noise_r,
            initialized: false,
            last_ts_ms: 0,
        }
    }

    #[inline]
    pub fn update(&mut self, measurement: f64, noise_multiplier: f64, now_ms: u64) -> f64 {
        if !measurement.is_finite() || measurement <= 0.0 {
            return self.estimate;
        }
        if !self.initialized {
            self.estimate = measurement;
            self.error_cov = 1.0;
            self.initialized = true;
            self.last_ts_ms = now_ms;
            return self.estimate;
        }

        let q = self.process_noise_q.max(1e-12);
        let r = (self.meas_noise_r * noise_multiplier.max(0.1)).max(1e-12);
        let p_pred = self.error_cov + q;
        let denom = p_pred + r;
        let gain = if denom.is_finite() && denom > 1e-15 {
            (p_pred / denom).clamp(0.0, 1.0)
        } else {
            0.0
        };

        self.estimate += gain * (measurement - self.estimate);
        self.error_cov = ((1.0 - gain) * p_pred).max(1e-12);
        self.last_ts_ms = now_ms;
        self.estimate
    }

    #[inline]
    pub fn on_book(&mut self, book: &L2OrderBook) -> Option<f64> {
        let micro = book.microprice()?;
        Some(self.update(micro, 1.0, book.local_ts_ms))
    }

    #[inline]
    pub fn on_trade(&mut self, tick: &TradeTick) -> Option<f64> {
        if !tick.is_valid() {
            return None;
        }
        // Trades are slightly noisier than L2 microprice (bid-ask bounce)
        Some(self.update(tick.price, 2.5, tick.local_ts_ms))
    }

    #[inline]
    pub fn on_funding_basis(&mut self, fb: &FundingBasisSnapshot) -> Option<f64> {
        if !fb.index_price.is_finite() || fb.index_price <= 0.0 {
            return None;
        }
        // Spot index acts as a slow macro gravity anchor
        Some(self.update(fb.index_price, 6.0, fb.local_ts_ms))
    }

    #[inline]
    pub fn value(&self) -> Option<f64> {
        if self.initialized && self.estimate.is_finite() && self.estimate > 0.0 {
            Some(self.estimate)
        } else {
            None
        }
    }

    /// Expected fair-value edge in bps relative to current reference price
    #[inline]
    pub fn edge_bps(&self, ref_price: f64) -> f64 {
        if !self.initialized || !ref_price.is_finite() || ref_price <= 0.0 {
            return 0.0;
        }
        let edge = ((self.estimate - ref_price) / ref_price) * 10_000.0;
        if edge.is_finite() { edge } else { 0.0 }
    }
}

/// Institutional Order Flow Imbalance (Cont-Kukanov-Stoikov OFI) tracker
#[derive(Debug, Clone, Copy)]
pub struct OrderFlowImbalance {
    prev_bid_px: f64,
    prev_bid_qty: f64,
    prev_ask_px: f64,
    prev_ask_qty: f64,
    ewma_ofi: f64,
    ewma_abs_ofi: f64,
    alpha: f64,
    initialized: bool,
}

impl OrderFlowImbalance {
    pub const fn new(alpha: f64) -> Self {
        Self {
            prev_bid_px: 0.0,
            prev_bid_qty: 0.0,
            prev_ask_px: 0.0,
            prev_ask_qty: 0.0,
            ewma_ofi: 0.0,
            ewma_abs_ofi: 0.0,
            alpha,
            initialized: false,
        }
    }

    /// Computes normalized OFI in [-1.0, 1.0] from consecutive L2 snapshots
    pub fn on_book(&mut self, book: &L2OrderBook) -> f64 {
        let (b, a) = match (book.best_bid(), book.best_ask()) {
            (Some(b), Some(a)) if b.price < a.price => (b, a),
            _ => return self.normalized(),
        };

        if !self.initialized {
            self.prev_bid_px = b.price;
            self.prev_bid_qty = b.qty;
            self.prev_ask_px = a.price;
            self.prev_ask_qty = a.qty;
            self.initialized = true;
            return 0.0;
        }

        let delta_bid = if b.price > self.prev_bid_px {
            b.qty
        } else if b.price < self.prev_bid_px {
            -self.prev_bid_qty
        } else {
            b.qty - self.prev_bid_qty
        };

        let delta_ask = if a.price > self.prev_ask_px {
            -self.prev_ask_qty
        } else if a.price < self.prev_ask_px {
            a.qty
        } else {
            a.qty - self.prev_ask_qty
        };

        let raw_ofi = delta_bid - delta_ask;
        if raw_ofi.is_finite() {
            let a_w = self.alpha.clamp(0.02, 0.95);
            self.ewma_ofi = (a_w * raw_ofi) + ((1.0 - a_w) * self.ewma_ofi);
            self.ewma_abs_ofi = (a_w * raw_ofi.abs()) + ((1.0 - a_w) * self.ewma_abs_ofi);
        }

        self.prev_bid_px = b.price;
        self.prev_bid_qty = b.qty;
        self.prev_ask_px = a.price;
        self.prev_ask_qty = a.qty;

        self.normalized()
    }

    #[inline]
    pub fn normalized(&self) -> f64 {
        if !self.ewma_abs_ofi.is_finite() || self.ewma_abs_ofi <= 1e-12 {
            return 0.0;
        }
        (self.ewma_ofi / self.ewma_abs_ofi).clamp(-1.0, 1.0)
    }
}
