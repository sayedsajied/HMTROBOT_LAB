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
use crate::types::{EngineSignal, SymbolId, Timeframe};

pub const NUM_ALPHA_ENGINES: usize = 86;

const ZERO_SIGNAL: EngineSignal = EngineSignal {
    engine_id: 0,
    timeframe: Timeframe::Sec1,
    score: 0.0,
    confidence: 0.0,
};

#[inline]
pub(crate) fn tanh_s(val: f64, scale: f64) -> f64 {
    if !val.is_finite() || !scale.is_finite() || scale <= 1e-12 {
        return 0.0;
    }
    (val / scale).tanh().clamp(-1.0, 1.0)
}

/// Zero-allocation output of all 86 silent alpha engines + consensus diagnostics
#[derive(Debug, Clone, Copy)]
pub struct AlphaBatch {
    pub symbol_id: SymbolId,
    pub signals: [EngineSignal; NUM_ALPHA_ENGINES],
    pub active_count: usize,
    pub consensus_score: f64,      // [-1.0, 1.0]
    pub consensus_confidence: f64, // [0.0, 1.0]
    pub dispersion: f64,           // [0.0, 1.0]
    pub timestamp_ms: u64,
}

impl AlphaBatch {
    pub const fn empty(symbol_id: SymbolId, timestamp_ms: u64) -> Self {
        Self {
            symbol_id,
            signals: [ZERO_SIGNAL; NUM_ALPHA_ENGINES],
            active_count: 0,
            consensus_score: 0.0,
            consensus_confidence: 0.0,
            dispersion: 0.0,
            timestamp_ms,
        }
    }

    /// Confidence-weighted consensus and inter-engine dispersion across all 86 engines
    pub fn finalize(&mut self) {
        let mut weighted_sum = 0.0;
        let mut conf_sum = 0.0;
        let mut active = 0usize;

        for sig in self.signals.iter() {
            if sig.confidence > 1e-6 && sig.score.is_finite() && sig.confidence.is_finite() {
                weighted_sum += sig.weighted_score();
                conf_sum += sig.confidence;
                active += 1;
            }
        }

        self.active_count = active;
        if active == 0 || conf_sum <= 1e-12 {
            self.consensus_score = 0.0;
            self.consensus_confidence = 0.0;
            self.dispersion = 0.0;
            return;
        }

        let mean_score = (weighted_sum / conf_sum).clamp(-1.0, 1.0);
        let mut var_acc = 0.0;
        for sig in self.signals.iter() {
            if sig.confidence > 1e-6 && sig.score.is_finite() {
                let d = sig.score - mean_score;
                var_acc += sig.confidence * (d * d);
            }
        }

        let std_dev = (var_acc / conf_sum).max(0.0).sqrt();
        self.consensus_score = mean_score;
        self.consensus_confidence = (conf_sum / (active as f64)).clamp(0.0, 1.0);
        self.dispersion = std_dev.clamp(0.0, 1.0);
    }
}
/// Stack-allocated rolling state for a single timeframe (1s, 15s, 1m, or 5m)
#[derive(Debug, Clone, Copy)]
pub struct TimeframeState {
    pub closes: RollingWindow<32>,
    pub ranges_bps: RollingWindow<32>,
    pub vol_deltas: RollingWindow<32>,
    pub ema_fast: f64,
    pub ema_slow: f64,
    pub last_bar: Option<OhlcvBar>,
    pub bar_count: u64,
}

impl TimeframeState {
    pub const fn new() -> Self {
        Self {
            closes: RollingWindow::new(),
            ranges_bps: RollingWindow::new(),
            vol_deltas: RollingWindow::new(),
            ema_fast: 0.0,
            ema_slow: 0.0,
            last_bar: None,
            bar_count: 0,
        }
    }

    pub fn on_bar(&mut self, bar: OhlcvBar) {
        if !bar.close.is_finite() || bar.close <= 0.0 {
            return;
        }
        self.closes.push(bar.close);
        self.ranges_bps.push(bar.range_bps());
        self.vol_deltas.push(bar.volume_delta_ratio());

        if self.bar_count == 0 {
            self.ema_fast = bar.close;
            self.ema_slow = bar.close;
        } else {
            self.ema_fast = (0.35 * bar.close) + (0.65 * self.ema_fast);
            self.ema_slow = (0.12 * bar.close) + (0.88 * self.ema_slow);
        }

        self.last_bar = Some(bar);
        self.bar_count = self.bar_count.saturating_add(1);
    }

    #[inline]
    pub fn ema_spread_bps(&self) -> f64 {
        if self.bar_count < 2 || self.ema_slow <= 0.0 {
            return 0.0;
        }
        let s = ((self.ema_fast - self.ema_slow) / self.ema_slow) * 10_000.0;
        if s.is_finite() { s } else { 0.0 }
    }
}

/// Pre-parsed zero-allocation tick input shared across all 86 engines (BTC / ETH / SOL)
#[derive(Debug, Clone, Copy)]
pub struct EngineTickInput {
    pub price: f64,
    pub bid: f64,
    pub ask: f64,
    pub bid_sz: f64,
    pub ask_sz: f64,
    pub top3_bid_vol: f64,
    pub top3_ask_vol: f64,
    pub top5_bid_vol: f64,
    pub top5_ask_vol: f64,
    pub b2_sz: f64,
    pub a2_sz: f64,
    pub spot_price: f64,
    pub funding_rate: f64,
    pub btc_anchor_ret_bps: f64,
    pub dt_sec: f64,
    pub has_book: bool,
}

impl EngineTickInput {
    pub fn from_snapshot(
        book: &L2OrderBook,
        fb: &FundingBasisSnapshot,
        btc_anchor_ret_bps: f64,
        dt_sec: f64,
    ) -> Self {
        let has_book = book.is_valid();
        let (bid, bid_sz) = book
            .best_bid()
            .map(|l| (l.price, l.qty))
            .unwrap_or((fb.mark_price, 1.0));
        let (ask, ask_sz) = book
            .best_ask()
            .map(|l| (l.price, l.qty))
            .unwrap_or((fb.mark_price, 1.0));

        let price = if fb.mark_price.is_finite() && fb.mark_price > 0.0 {
            fb.mark_price
        } else if has_book {
            (bid + ask) * 0.5
        } else {
            0.0
        };

        let (mut top3_b, mut top3_a, mut top5_b, mut top5_a) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..5 {
            if i < book.bid_count && book.bids[i].is_valid() {
                top5_b += book.bids[i].qty;
                if i < 3 { top3_b += book.bids[i].qty; }
            }
            if i < book.ask_count && book.asks[i].is_valid() {
                top5_a += book.asks[i].qty;
                if i < 3 { top3_a += book.asks[i].qty; }
            }
        }

        let b2_sz = if book.bid_count >= 2 && book.bids[1].is_valid() { book.bids[1].qty } else { bid_sz };
        let a2_sz = if book.ask_count >= 2 && book.asks[1].is_valid() { book.asks[1].qty } else { ask_sz };

        Self {
            price,
            bid,
            ask,
            bid_sz: bid_sz.max(0.0),
            ask_sz: ask_sz.max(0.0),
            top3_bid_vol: top3_b.max(bid_sz),
            top3_ask_vol: top3_a.max(ask_sz),
            top5_bid_vol: top5_b.max(bid_sz),
            top5_ask_vol: top5_a.max(ask_sz),
            b2_sz: b2_sz.max(0.0),
            a2_sz: a2_sz.max(0.0),
            spot_price: if fb.index_price.is_finite() && fb.index_price > 0.0 { fb.index_price } else { price },
            funding_rate: if fb.funding_rate.is_finite() { fb.funding_rate } else { 0.0 },
            btc_anchor_ret_bps: if btc_anchor_ret_bps.is_finite() { btc_anchor_ret_bps } else { 0.0 },
            dt_sec: dt_sec.clamp(0.001, 60.0),
            has_book,
        }
    }
}
/// Stack-only state for custom Microstructure Engines 01..15 (Zero Mutex, Zero Heap, Silent)
#[derive(Debug, Clone, Copy)]
pub struct Engines01To15 {
    last_price: f64,
    vpin_buy: f64,
    vpin_sell: f64,
    hawkes_lam: f64,
    kyle_last_mid: f64,
    kyle_lam: f64,
    roll_prev_dp: f64,
    roll_cov: f64,
    ofi_pb_p: f64,
    ofi_pb_q: f64,
    ofi_pa_p: f64,
    ofi_pa_q: f64,
    ofi_smooth: f64,
    ice_pb_p: f64,
    ice_pb_q: f64,
    ice_b_refill: f64,
    ice_pa_p: f64,
    ice_pa_q: f64,
    ice_a_refill: f64,
    sw_pb: f64,
    sw_pb_v: f64,
    sw_pa: f64,
    sw_pa_v: f64,
    sp_pb_v: f64,
    sp_pa_v: f64,
    sp_b_churn: f64,
    sp_a_churn: f64,
    e13_dev_bps: f64,
    e14_pressure: f64,
    e15_basis: f64,
}

impl Engines01To15 {
    pub const fn new() -> Self {
        Self {
            last_price: 0.0,
            vpin_buy: 0.0,
            vpin_sell: 0.0,
            hawkes_lam: 1.0,
            kyle_last_mid: 0.0,
            kyle_lam: 0.0,
            roll_prev_dp: 0.0,
            roll_cov: 0.0,
            ofi_pb_p: 0.0,
            ofi_pb_q: 0.0,
            ofi_pa_p: 0.0,
            ofi_pa_q: 0.0,
            ofi_smooth: 0.0,
            ice_pb_p: 0.0,
            ice_pb_q: 0.0,
            ice_b_refill: 0.0,
            ice_pa_p: 0.0,
            ice_pa_q: 0.0,
            ice_a_refill: 0.0,
            sw_pb: 0.0,
            sw_pb_v: 0.0,
            sw_pa: 0.0,
            sw_pa_v: 0.0,
            sp_pb_v: 0.0,
            sp_pa_v: 0.0,
            sp_b_churn: 0.0,
            sp_a_churn: 0.0,
            e13_dev_bps: 0.0,
            e14_pressure: 0.0,
            e15_basis: 0.0,
        }
    }

    pub fn eval(&mut self, inp: &EngineTickInput, out: &mut [EngineSignal; NUM_ALPHA_ENGINES]) {
        let price = inp.price;
        if !price.is_finite() || price <= 0.0 {
            return;
        }
        let dt = inp.dt_sec.clamp(0.001, 60.0);
        let prev_p = self.last_price;
        let has_prev = prev_p > 0.0;
        let ret_bps = if has_prev { ((price - prev_p) / prev_p) * 10_000.0 } else { 0.0 };
        let total_l1 = (inp.bid_sz + inp.ask_sz).max(1e-9);
        let imb_l1 = ((inp.bid_sz - inp.ask_sz) / total_l1).clamp(-1.0, 1.0);
        let mid = if inp.bid > 0.0 && inp.ask > 0.0 { (inp.bid + inp.ask) * 0.5 } else { price };
        let micro = if inp.bid > 0.0 && inp.ask > 0.0 {
            ((inp.bid * inp.ask_sz) + (inp.ask * inp.bid_sz)) / total_l1
        } else {
            mid
        };

        // E01: Microprice Baseline Anchor
        let base_dev = if mid > 0.0 { ((micro - mid) / mid) * 10_000.0 } else { 0.0 };
        out[0] = EngineSignal::sanitized(1, Timeframe::Sec1, tanh_s(base_dev, 2.0), if inp.has_book { 0.80 } else { 0.0 });

        // E02: Scalp Burst (Stoikov Micro-Price & L1 Imbalance)
        out[1] = EngineSignal::sanitized(2, Timeframe::Sec1, imb_l1, if inp.has_book { 0.85 } else { 0.0 });

        // E03: Burst Velocity
        out[2] = EngineSignal::sanitized(3, Timeframe::Sec1, tanh_s(ret_bps / dt, 8.0), if has_prev { 0.75 } else { 0.0 });

        // E04: L2 Orderbook Wall Scanner (Top-5)
        let tot5 = inp.top5_bid_vol + inp.top5_ask_vol;
        let depth_ratio = if tot5 > 0.0 { (inp.top5_bid_vol - inp.top5_ask_vol) / tot5 } else { 0.0 };
        out[3] = EngineSignal::sanitized(4, Timeframe::Sec1, depth_ratio, if inp.has_book { 0.80 } else { 0.0 });

        // E05: VPIN Toxic Flow
        if has_prev {
            self.vpin_buy *= 0.85;
            self.vpin_sell *= 0.85;
            if price > prev_p {
                self.vpin_buy += inp.bid_sz + (inp.ask_sz * 0.5);
            } else if price < prev_p {
                self.vpin_sell += inp.ask_sz + (inp.bid_sz * 0.5);
            } else {
                self.vpin_buy += inp.bid_sz * 0.5;
                self.vpin_sell += inp.ask_sz * 0.5;
            }
            let tv = self.vpin_buy + self.vpin_sell;
            let signed_vpin = if tv > 0.0 { (self.vpin_buy - self.vpin_sell) / tv } else { 0.0 };
            out[4] = EngineSignal::sanitized(5, Timeframe::Sec1, signed_vpin, 0.80);
        }

        // E06: Hawkes Self-Exciting Intensity
        self.hawkes_lam = 1.0 + (self.hawkes_lam - 1.0) * (-2.5 * dt).exp() + 1.8;
        let hawkes_norm = ((self.hawkes_lam - 1.0) / 8.5).clamp(0.0, 1.0);
        out[5] = EngineSignal::sanitized(6, Timeframe::Sec1, ret_bps.signum() * hawkes_norm, if has_prev { 0.65 } else { 0.0 });

        // E07: Kyle's Lambda (Market Impact)
        if self.kyle_last_mid > 0.0 {
            let impact_bps = ((mid - self.kyle_last_mid).abs() / self.kyle_last_mid) * 10_000.0;
            let net_of = (inp.bid_sz - inp.ask_sz).abs().max(1.0);
            self.kyle_lam = (self.kyle_lam * 0.80) + (((impact_bps / net_of) * 100.0) * 0.20);
            out[6] = EngineSignal::sanitized(7, Timeframe::Sec1, (mid - self.kyle_last_mid).signum() * tanh_s(self.kyle_lam, 0.50), 0.70);
        }
        self.kyle_last_mid = mid;

        // E08: Roll's Model Effective Spread
        if has_prev {
            let dp = price - prev_p;
            self.roll_cov = (self.roll_cov * 0.85) + ((dp * self.roll_prev_dp) * 0.15);
            self.roll_prev_dp = dp;
            let eff_bps = if self.roll_cov < 0.0 { (2.0 * (-self.roll_cov).sqrt() / price) * 10_000.0 } else { 0.0 };
            let roll_score = if self.roll_cov > 0.0 {
                tanh_s(ret_bps, 2.0)
            } else {
                -tanh_s(ret_bps, 2.0) * (1.0 - (eff_bps / 5.0).clamp(0.0, 0.8))
            };
            out[7] = EngineSignal::sanitized(8, Timeframe::Sec1, roll_score, 0.65);
        }

        // E09: Cont-Kukanov-Stoikov OFI (Top-3 Queue)
        if inp.has_book && self.ofi_pb_p > 0.0 && self.ofi_pa_p > 0.0 {
            let e_bid = if inp.bid > self.ofi_pb_p { inp.top3_bid_vol } else if inp.bid < self.ofi_pb_p { -self.ofi_pb_q } else { inp.top3_bid_vol - self.ofi_pb_q };
            let e_ask = if inp.ask < self.ofi_pa_p { inp.top3_ask_vol } else if inp.ask > self.ofi_pa_p { -self.ofi_pa_q } else { inp.top3_ask_vol - self.ofi_pa_q };
            self.ofi_smooth = (self.ofi_smooth * 0.80) + ((e_bid - e_ask) * 0.20);
            out[8] = EngineSignal::sanitized(9, Timeframe::Sec1, tanh_s(self.ofi_smooth, 150.0), 0.85);
        }
        if inp.has_book {
            self.ofi_pb_p = inp.bid; self.ofi_pb_q = inp.top3_bid_vol;
            self.ofi_pa_p = inp.ask; self.ofi_pa_q = inp.top3_ask_vol;
        }

        // E10: Iceberg Order Sniffer
        if inp.has_book {
            if (inp.bid - self.ice_pb_p).abs() < 1e-5 {
                if (inp.bid_sz - self.ice_pb_q).abs() > 1e-9 { self.ice_b_refill = (self.ice_b_refill * 0.85) + 1.0; }
            } else { self.ice_b_refill = 0.0; }
            if (inp.ask - self.ice_pa_p).abs() < 1e-5 {
                if (inp.ask_sz - self.ice_pa_q).abs() > 1e-9 { self.ice_a_refill = (self.ice_a_refill * 0.85) + 1.0; }
            } else { self.ice_a_refill = 0.0; }
            self.ice_pb_p = inp.bid; self.ice_pb_q = inp.bid_sz;
            self.ice_pa_p = inp.ask; self.ice_pa_q = inp.ask_sz;
            out[9] = EngineSignal::sanitized(10, Timeframe::Sec1, tanh_s(self.ice_b_refill - self.ice_a_refill, 3.2), 0.75);
        }

        // E11: Whale Book Sweep Detector
        if inp.has_book && self.sw_pb > 0.0 && self.sw_pa > 0.0 {
            let ask_jump = ((inp.ask - self.sw_pa) / self.sw_pa) * 10_000.0;
            let bid_drop = ((self.sw_pb - inp.bid) / self.sw_pb) * 10_000.0;
            let ask_swept = if inp.ask > self.sw_pa { self.sw_pa_v } else { (self.sw_pa_v - inp.top3_ask_vol).max(0.0) };
            let bid_swept = if inp.bid < self.sw_pb { self.sw_pb_v } else { (self.sw_pb_v - inp.top3_bid_vol).max(0.0) };
            let sweep = tanh_s(ask_jump.max(0.0) * (ask_swept / 50.0) - bid_drop.max(0.0) * (bid_swept / 50.0), 1.5);
            out[10] = EngineSignal::sanitized(11, Timeframe::Sec1, sweep, 0.80);
        }
        if inp.has_book {
            self.sw_pb = inp.bid; self.sw_pb_v = inp.top3_bid_vol;
            self.sw_pa = inp.ask; self.sw_pa_v = inp.top3_ask_vol;
        }

        // E12: Spoofing & Cancellation Filter
        if inp.has_book && self.sp_pb_v > 0.0 && self.sp_pa_v > 0.0 {
            let bid_pulled = (self.sp_pb_v - inp.top5_bid_vol).max(0.0) / self.sp_pb_v;
            let ask_pulled = (self.sp_pa_v - inp.top5_ask_vol).max(0.0) / self.sp_pa_v;
            self.sp_b_churn = (self.sp_b_churn * 0.75) + (bid_pulled * 0.25);
            self.sp_a_churn = (self.sp_a_churn * 0.75) + (ask_pulled * 0.25);
            out[11] = EngineSignal::sanitized(12, Timeframe::Sec1, tanh_s(self.sp_a_churn - self.sp_b_churn, 0.35), 0.70);
        }
        if inp.has_book { self.sp_pb_v = inp.top5_bid_vol; self.sp_pa_v = inp.top5_ask_vol; }

        // E13: Micro-Price Deviation & Fair-Value Skew
        let spread = (inp.ask - inp.bid).max(1e-8);
        let raw_dev = micro - mid;
        let dev_bps = if mid > 0.0 { (raw_dev / mid) * 10_000.0 } else { 0.0 };
        self.e13_dev_bps = (self.e13_dev_bps * 0.75) + (dev_bps * 0.25);
        out[12] = EngineSignal::sanitized(13, Timeframe::Sec1, (raw_dev / (spread * 0.5)).clamp(-1.0, 1.0), if inp.has_book { 0.80 } else { 0.0 });

        // E14: Time-Weighted Pressure Decay
        self.e14_pressure = (self.e14_pressure * 0.80) + (imb_l1 * 0.20);
        out[13] = EngineSignal::sanitized(14, Timeframe::Sec1, tanh_s(self.e14_pressure, 0.35), if inp.has_book { 0.75 } else { 0.0 });

        // E15: Funding Rate & Mark-to-Spot Basis
        if inp.spot_price > 0.0 {
            let raw_basis = ((price - inp.spot_price) / inp.spot_price) * 10_000.0;
            self.e15_basis = (self.e15_basis * 0.80) + (raw_basis * 0.20);
            out[14] = EngineSignal::sanitized(15, Timeframe::Min1, -tanh_s(self.e15_basis + (inp.funding_rate * 50_000.0), 15.0), 0.70);
        }

        self.last_price = price;
    }
}
/// Stack-only state for custom Stochastic, Cross-Asset & Volatility Engines 16..30
#[derive(Debug, Clone, Copy)]
pub struct Engines16To30 {
    last_price: f64,
    k_pos: f64,
    k_vel: f64,
    k_p: [f64; 4],
    k_init: bool,
    ito_mu: f64,
    ito_var: f64,
    ou18_mu: f64,
    ou18_var: f64,
    jmp_var: f64,
    jmp_lam: f64,
    h_cum: f64,
    h_max: f64,
    h_min: f64,
    h_std: f64,
    h_val: f64,
    hmm_calm: f64,
    hmm_burst: f64,
    ent_bins: [f64; 4],
    wav_buf: [f64; 4],
    mah_mean: [f64; 2],
    mah_cov: [f64; 4],
    eig_cov: [f64; 4],
    ll_prev_self: f64,
    ll_beta: f64,
    coint_mean: f64,
    coint_var: f64,
    garch_eps2: f64,
    garch_sig2: f64,
    yz_open: f64,
    yz_high: f64,
    yz_low: f64,
    yz_prev_c: f64,
    yz_cnt: u32,
    yz_v: [f64; 3],
    bpv_prev_abs: f64,
    bpv_rv: f64,
    bpv_bv: f64,
}

impl Engines16To30 {
    pub const fn new() -> Self {
        Self {
            last_price: 0.0,
            k_pos: 0.0,
            k_vel: 0.0,
            k_p: [1.0, 0.0, 0.0, 1.0],
            k_init: false,
            ito_mu: 0.0,
            ito_var: 0.0,
            ou18_mu: 0.0,
            ou18_var: 1.0,
            jmp_var: 1.0,
            jmp_lam: 0.0,
            h_cum: 0.0,
            h_max: 0.0,
            h_min: 0.0,
            h_std: 0.5,
            h_val: 0.5,
            hmm_calm: 0.85,
            hmm_burst: 0.15,
            ent_bins: [1.0; 4],
            wav_buf: [0.0; 4],
            mah_mean: [0.0; 2],
            mah_cov: [1.0, 0.0, 0.0, 0.1],
            eig_cov: [1.0, 0.0, 0.0, 1.0],
            ll_prev_self: 0.0,
            ll_beta: 0.0,
            coint_mean: 0.0,
            coint_var: 1.0,
            garch_eps2: 0.5,
            garch_sig2: 1.0,
            yz_open: 0.0,
            yz_high: 0.0,
            yz_low: 0.0,
            yz_prev_c: 0.0,
            yz_cnt: 0,
            yz_v: [0.2; 3],
            bpv_prev_abs: 0.0,
            bpv_rv: 0.5,
            bpv_bv: 0.5,
        }
    }

    pub fn eval(&mut self, inp: &EngineTickInput, out: &mut [EngineSignal; NUM_ALPHA_ENGINES]) {
        let price = inp.price;
        if !price.is_finite() || price <= 0.0 {
            return;
        }
        let dt = inp.dt_sec.clamp(0.001, 60.0);
        let prev_p = self.last_price;
        let has_prev = prev_p > 0.0;
        let ret_bps = if has_prev { ((price - prev_p) / prev_p) * 10_000.0 } else { 0.0 };
        let total_l1 = (inp.bid_sz + inp.ask_sz).max(1e-9);
        let imb_l1 = ((inp.bid_sz - inp.ask_sz) / total_l1).clamp(-1.0, 1.0);

        // E16: 2x2 Matrix State-Space Kalman Filter
        if !self.k_init {
            self.k_pos = price;
            self.k_vel = 0.0;
            self.k_p = [1.0, 0.0, 0.0, 1.0];
            self.k_init = true;
        } else {
            let x0 = self.k_pos + self.k_vel;
            let [p00, p01, p10, p11] = self.k_p;
            let pp00 = p00 + p01 + p10 + p11 + 0.01;
            let pp01 = p01 + p11;
            let pp10 = p10 + p11;
            let pp11 = p11 + 0.005;
            let innov = price - x0;
            let s_cov = (pp00 + 0.25).max(1e-9);
            let k0 = pp00 / s_cov;
            let k1 = pp10 / s_cov;
            self.k_pos = x0 + k0 * innov;
            self.k_vel += k1 * innov;
            self.k_p = [
                (1.0 - k0) * pp00,
                (1.0 - k0) * pp01,
                (-k1 * pp00) + pp10,
                (-k1 * pp01) + pp11,
            ];
            out[15] = EngineSignal::sanitized(16, Timeframe::Sec1, tanh_s((innov / price) * 10_000.0, 2.0), 0.80);
        }

        // E17: Ito's Lemma Stochastic Drift & Diffusion
        if has_prev {
            self.ito_mu = (self.ito_mu * 0.85) + ((ret_bps / dt) * 0.15);
            self.ito_var = (self.ito_var * 0.85) + (((ret_bps * ret_bps) / dt) * 0.15);
            out[16] = EngineSignal::sanitized(17, Timeframe::Sec1, tanh_s(self.ito_mu - (0.5 * self.ito_var * 0.01), 2.5), 0.70);
        }

        // E18: Ornstein-Uhlenbeck (O-U) Mean Reversion
        if self.ou18_mu <= 0.0 {
            self.ou18_mu = price;
            self.ou18_var = 1.0;
        } else {
            self.ou18_mu += 0.18 * (price - self.ou18_mu);
            let d = price - self.ou18_mu;
            self.ou18_var = ((self.ou18_var * 0.85) + (d * d * 0.15)).max(1e-6);
            out[17] = EngineSignal::sanitized(18, Timeframe::Sec15, -tanh_s(d / self.ou18_var.sqrt(), 2.0), 0.75);
        }

        // E19: Merton Jump-Diffusion Burst Detector
        if has_prev {
            let jr = ret_bps.abs() / self.jmp_var.sqrt().max(0.2);
            let is_jump = if jr > 2.8 { 1.0 } else { 0.0 };
            if is_jump == 0.0 {
                self.jmp_var = ((self.jmp_var * 0.88) + (ret_bps * ret_bps * 0.12)).max(0.05);
            }
            self.jmp_lam = (self.jmp_lam * 0.85) + (is_jump * 0.15);
            out[18] = EngineSignal::sanitized(19, Timeframe::Sec1, ret_bps.signum() * tanh_s(jr * self.jmp_lam, 0.8), 0.70);
        }

        // E20: Hurst Exponent (R/S Fractal Regime)
        if has_prev {
            self.h_cum = (self.h_cum * 0.85) + ret_bps;
            self.h_max = (self.h_max * 0.90).max(self.h_cum);
            self.h_min = (self.h_min * 0.90).min(self.h_cum);
            let range_r = (self.h_max - self.h_min).max(0.05);
            self.h_std = ((self.h_std * 0.88) + (ret_bps.abs() * 0.12)).max(0.05);
            let raw_h = ((range_r / self.h_std).max(1.05).ln() / (20.0_f64).ln()).clamp(0.10, 0.90);
            self.h_val = (self.h_val * 0.80) + (raw_h * 0.20);
            out[19] = EngineSignal::sanitized(20, Timeframe::Sec15, (self.h_val - 0.5) * 2.5 * tanh_s(ret_bps, 2.0), 0.75);
        }

        // E21: Hidden Markov Model (HMM) Bayesian Regime Switcher
        if has_prev {
            let r_sq = ret_bps * ret_bps;
            let p0 = 0.92 * self.hmm_calm + 0.20 * self.hmm_burst;
            let p1 = 0.08 * self.hmm_calm + 0.80 * self.hmm_burst;
            let u0 = p0 * ((-r_sq / 0.72).exp() / 0.6);
            let u1 = p1 * ((-r_sq / 24.5).exp() / 3.5);
            let ns = (u0 + u1).max(1e-12);
            self.hmm_calm = u0 / ns;
            self.hmm_burst = u1 / ns;
            out[20] = EngineSignal::sanitized(21, Timeframe::Sec1, tanh_s(ret_bps, 2.5) * self.hmm_burst, 0.70);
        }

        // E22: Shannon Entropy Information Filter
        if has_prev {
            for b in self.ent_bins.iter_mut() { *b *= 0.90; }
            if ret_bps < -1.5 { self.ent_bins[0] += 1.0; }
            else if ret_bps < 0.0 { self.ent_bins[1] += 1.0; }
            else if ret_bps <= 1.5 { self.ent_bins[2] += 1.0; }
            else { self.ent_bins[3] += 1.0; }
            let tot = (self.ent_bins[0] + self.ent_bins[1] + self.ent_bins[2] + self.ent_bins[3]).max(1.0);
            let mut h = 0.0;
            for b in self.ent_bins {
                let p = b / tot;
                if p > 0.001 { h -= p * p.log2(); }
            }
            let norm_h = (h / 2.0).clamp(0.0, 1.0);
            let dir = ((self.ent_bins[2] + self.ent_bins[3]) - (self.ent_bins[0] + self.ent_bins[1])) / tot;
            out[21] = EngineSignal::sanitized(22, Timeframe::Sec15, dir * (1.0 - norm_h), 0.70);
        }

        // E23: 4-Point Orthogonal Haar Wavelet Denoiser
        if self.wav_buf[3] <= 0.0 {
            self.wav_buf = [price; 4];
        } else {
            self.wav_buf = [self.wav_buf[1], self.wav_buf[2], self.wav_buf[3], price];
            let s = std::f64::consts::FRAC_1_SQRT_2;
            let [w0, w1, w2, w3] = self.wav_buf;
            let c0 = 0.5 * (w0 + w1 + w2 + w3);
            let c1 = 0.5 * (w0 + w1 - w2 - w3);
            let mut c3 = s * (w2 - w3);
            let thr = price * 0.00015;
            c3 = if c3.abs() <= thr { 0.0 } else { c3.signum() * (c3.abs() - thr) };
            let clean_p = (0.5 * c0) - (0.5 * c1) - (s * c3);
            let edge = if clean_p > 0.0 { ((price - clean_p) / clean_p) * 10_000.0 } else { 0.0 };
            out[22] = EngineSignal::sanitized(23, Timeframe::Sec1, tanh_s(edge, 1.5), 0.75);
        }

        // E24: Mahalanobis 2x2 Multivariate Anomaly
        if has_prev {
            let dx = ret_bps - self.mah_mean[0];
            let dy = imb_l1 - self.mah_mean[1];
            let a = self.mah_cov[0] + 0.05;
            let b = self.mah_cov[1];
            let c = self.mah_cov[2];
            let d = self.mah_cov[3] + 0.01;
            let det = (a * d) - (b * c);
            let d_sq = if det.abs() > 1e-9 {
                let ix = (d * dx - b * dy) / det;
                let iy = (-c * dx + a * dy) / det;
                (dx * ix + dy * iy).max(0.0)
            } else {
                dx * dx + dy * dy
            };
            self.mah_mean[0] = (self.mah_mean[0] * 0.90) + (ret_bps * 0.10);
            self.mah_mean[1] = (self.mah_mean[1] * 0.90) + (imb_l1 * 0.10);
            self.mah_cov[0] = (self.mah_cov[0] * 0.90) + (dx * dx * 0.10);
            self.mah_cov[1] = (self.mah_cov[1] * 0.90) + (dx * dy * 0.10);
            self.mah_cov[2] = self.mah_cov[1];
            self.mah_cov[3] = (self.mah_cov[3] * 0.90) + (dy * dy * 0.10);
            out[23] = EngineSignal::sanitized(24, Timeframe::Sec1, ret_bps.signum() * tanh_s(d_sq.sqrt() - 1.5, 2.0), 0.70);
        }

        // E25: Eigenvalue Cross-Asset Covariance Absorption Ratio (BTC Anchor)
        if has_prev {
            let r0 = ret_bps;
            let r1 = inp.btc_anchor_ret_bps;
            self.eig_cov[0] = (self.eig_cov[0] * 0.90) + (r0 * r0 * 0.10);
            self.eig_cov[1] = (self.eig_cov[1] * 0.90) + (r0 * r1 * 0.10);
            self.eig_cov[2] = self.eig_cov[1];
            self.eig_cov[3] = (self.eig_cov[3] * 0.90) + (r1 * r1 * 0.10);
            let tr = self.eig_cov[0] + self.eig_cov[3];
            let disc = ((self.eig_cov[0] - self.eig_cov[3]).powi(2) + 4.0 * self.eig_cov[1].powi(2)).sqrt();
            let abs_ratio = ((0.5 * (tr + disc)) / tr.abs().max(1e-6)).clamp(0.0, 1.0);
            out[24] = EngineSignal::sanitized(25, Timeframe::Sec15, tanh_s(r0 + r1, 3.0) * abs_ratio, 0.65);
        }

        // E26: Cross-Asset Lead-Lag Matrix (BTC -> ETH/SOL)
        if has_prev {
            self.ll_beta = (self.ll_beta * 0.88) + (ret_bps * self.ll_prev_self * 0.12);
            self.ll_prev_self = ret_bps;
            let lag_gap = inp.btc_anchor_ret_bps - ret_bps;
            let conf = if inp.btc_anchor_ret_bps.abs() > 1e-6 { 0.70 } else { 0.25 };
            out[25] = EngineSignal::sanitized(26, Timeframe::Sec1, tanh_s(lag_gap, 2.0), conf);
        }

        // E27: Cointegration Spread Z-Score (BTC -> ETH/SOL)
        if has_prev {
            let spread_diff = ret_bps - inp.btc_anchor_ret_bps;
            let diff = spread_diff - self.coint_mean;
            self.coint_mean = (self.coint_mean * 0.90) + (spread_diff * 0.10);
            self.coint_var = ((self.coint_var * 0.90) + (diff * diff * 0.10)).max(1e-4);
            let conf = if inp.btc_anchor_ret_bps.abs() > 1e-6 { 0.65 } else { 0.25 };
            out[26] = EngineSignal::sanitized(27, Timeframe::Sec15, -tanh_s(diff / self.coint_var.sqrt(), 2.1), conf);
        }

        // E28: GARCH(1,1) Conditional Volatility Clustering
        if has_prev {
            let cond_var = (0.05 + (0.18 * self.garch_eps2) + (0.78 * self.garch_sig2)).max(0.01);
            let shock = ret_bps / cond_var.sqrt();
            self.garch_eps2 = ret_bps * ret_bps;
            self.garch_sig2 = cond_var;
            out[27] = EngineSignal::sanitized(28, Timeframe::Sec15, tanh_s(shock, 2.2), 0.70);
        }

        // E29: Yang-Zhang Minimum-Variance Volatility
        if self.yz_prev_c <= 0.0 {
            self.yz_open = price;
            self.yz_high = price;
            self.yz_low = price;
            self.yz_prev_c = price;
            self.yz_cnt = 1;
        } else {
            self.yz_high = self.yz_high.max(price);
            self.yz_low = self.yz_low.min(price);
            self.yz_cnt += 1;
            if self.yz_cnt >= 4 {
                let o_j = (self.yz_open / self.yz_prev_c.max(1e-12)).ln() * 10_000.0;
                let c_d = (price / self.yz_open.max(1e-12)).ln() * 10_000.0;
                let u_h = (self.yz_high / self.yz_open.max(1e-12)).ln() * 10_000.0;
                let d_l = (self.yz_low / self.yz_open.max(1e-12)).ln() * 10_000.0;
                let rs = (u_h * (u_h - c_d) + d_l * (d_l - c_d)).max(0.0);
                self.yz_v[0] = (self.yz_v[0] * 0.85) + (o_j * o_j * 0.15);
                self.yz_v[1] = (self.yz_v[1] * 0.85) + (c_d * c_d * 0.15);
                self.yz_v[2] = (self.yz_v[2] * 0.85) + (rs * 0.15);
                self.yz_prev_c = price;
                self.yz_open = price;
                self.yz_high = price;
                self.yz_low = price;
                self.yz_cnt = 0;
            }
            let yz_sig = (self.yz_v[0] + 0.25 * self.yz_v[1] + 0.75 * self.yz_v[2]).max(0.01).sqrt();
            out[28] = EngineSignal::sanitized(29, Timeframe::Sec15, tanh_s(ret_bps / yz_sig, 2.0), 0.65);
        }

        // E30: Realized Bipower Variation (Continuous vs Jump Separator)
        if has_prev {
            let abs_r = ret_bps.abs();
            self.bpv_rv = (self.bpv_rv * 0.85) + (abs_r * abs_r * 0.15);
            self.bpv_bv = (self.bpv_bv * 0.85) + (std::f64::consts::FRAC_PI_2 * abs_r * self.bpv_prev_abs * 0.15);
            self.bpv_prev_abs = abs_r;
            let jump_share = ((self.bpv_rv - self.bpv_bv).max(0.0) / self.bpv_rv.max(1e-4)).clamp(0.0, 1.0);
            out[29] = EngineSignal::sanitized(30, Timeframe::Sec1, ret_bps.signum() * jump_share, 0.70);
        }

        self.last_price = price;
    }
}
