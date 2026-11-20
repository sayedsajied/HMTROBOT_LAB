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
/// Stack-only state for custom Convexity, Greeks, Execution & Tail-Risk Engines 31..45
#[derive(Debug, Clone, Copy)]
pub struct Engines31To45 {
    last_price: f64,
    // E31 Taylor Convexity
    t_p2: f64,
    t_p1: f64,
    t_vec: [f64; 3],
    // E32 Lyapunov Chaos
    ly_phase: [f64; 3],
    ly_prev_d: f64,
    ly_lam: f64,
    // E33 Kolmogorov-Smirnov Regime Shift
    ks_fast: [f64; 4],
    ks_slow: [f64; 4],
    // E34 Wasserstein L2 Transport
    w_cdf: [f64; 4],
    // E35 Hilbert Phase
    hil_buf: [f64; 4],
    // E36 BSM Greeks
    bsm_k: f64,
    bsm_sig: f64,
    bsm_delta: f64,
    // E37 Vanna-Volga
    vv_k: f64,
    vv_sig: f64,
    vv_flow: f64,
    // E38 Charm-Vera
    cv_k: f64,
    cv_sig: f64,
    cv_tau: f64,
    cv_bleed: f64,
    // E39 Speed-Zomma
    sz_k: f64,
    sz_sig: f64,
    sz_squeeze: f64,
    // E40 Almgren-Chriss
    ac_var: f64,
    ac_kappa: f64,
    ac_rate: f64,
    // E41 Avellaneda-Stoikov
    as_var: f64,
    as_skew: f64,
    // E42 Nash Game Theory
    nash_p: f64,
    nash_q: f64,
    // E43 Lotka-Volterra
    lv_x: f64,
    lv_y: f64,
    // E44 Merton-Kelly
    kel_mu: f64,
    kel_var: f64,
    // E45 EVT Tail Shield
    evt_u: f64,
    evt_ex: f64,
}

impl Engines31To45 {
    pub const fn new() -> Self {
        Self {
            last_price: 0.0,
            t_p2: 0.0,
            t_p1: 0.0,
            t_vec: [0.0; 3],
            ly_phase: [0.0; 3],
            ly_prev_d: 0.1,
            ly_lam: -0.05,
            ks_fast: [0.25, 0.50, 0.75, 1.0],
            ks_slow: [0.25, 0.50, 0.75, 1.0],
            w_cdf: [0.25, 0.50, 0.75, 1.0],
            hil_buf: [0.0; 4],
            bsm_k: 0.0,
            bsm_sig: 0.015,
            bsm_delta: 0.5,
            vv_k: 0.0,
            vv_sig: 0.015,
            vv_flow: 0.0,
            cv_k: 0.0,
            cv_sig: 0.015,
            cv_tau: 1.0,
            cv_bleed: 0.0,
            sz_k: 0.0,
            sz_sig: 0.015,
            sz_squeeze: 0.0,
            ac_var: 1.0,
            ac_kappa: 1.0,
            ac_rate: 0.5,
            as_var: 1.0,
            as_skew: 0.0,
            nash_p: 0.5,
            nash_q: 0.5,
            lv_x: 1.0,
            lv_y: 0.5,
            kel_mu: 0.0,
            kel_var: 1.0,
            evt_u: 1.2,
            evt_ex: 0.5,
        }
    }

    pub fn eval(&mut self, inp: &EngineTickInput, out: &mut [EngineSignal; NUM_ALPHA_ENGINES]) {
        let price = inp.price;
        if !price.is_finite() || price <= 0.0 {
            return;
        }
        let prev_p = self.last_price;
        let has_prev = prev_p > 0.0;
        let ret_bps = if has_prev { ((price - prev_p) / prev_p) * 10_000.0 } else { 0.0 };
        let spread_bps = if inp.bid > 0.0 && inp.ask > inp.bid {
            (((inp.ask - inp.bid) / price) * 10_000.0).max(0.1)
        } else {
            0.2
        };
        let top_depth = (inp.bid_sz + inp.ask_sz).max(0.1);
        let imb_l1 = ((inp.bid_sz - inp.ask_sz) / top_depth).clamp(-1.0, 1.0);

        // E31: Taylor Series 2nd-Order Convexity & Gamma Accelerator
        if self.t_p1 > 0.0 && self.t_p2 > 0.0 {
            let vel = ((price - self.t_p1) / self.t_p1) * 10_000.0;
            let prev_vel = ((self.t_p1 - self.t_p2) / self.t_p2) * 10_000.0;
            let gamma = vel - prev_vel;
            let proj = vel + (0.5 * gamma);
            self.t_vec[0] = (self.t_vec[0] * 0.75) + (vel * 0.25);
            self.t_vec[1] = (self.t_vec[1] * 0.75) + (gamma * 0.25);
            self.t_vec[2] = (self.t_vec[2] * 0.75) + (proj * 0.25);
            out[30] = EngineSignal::sanitized(31, Timeframe::Sec1, tanh_s(self.t_vec[2], 2.5), 0.75);
        }
        self.t_p2 = self.t_p1;
        self.t_p1 = price;

        // E32: Largest Lyapunov Exponent (3D Phase-Space Chaos)
        if has_prev {
            let curr = [self.ly_phase[1], self.ly_phase[2], ret_bps];
            let d0 = curr[0] - self.ly_phase[0];
            let d1 = curr[1] - self.ly_phase[1];
            let d2 = curr[2] - self.ly_phase[2];
            let curr_d = (d0 * d0 + d1 * d1 + d2 * d2).sqrt().max(0.02);
            let inst_lam = (curr_d / self.ly_prev_d.max(0.02)).ln().clamp(-2.0, 2.0);
            self.ly_lam = (self.ly_lam * 0.85) + (inst_lam * 0.15);
            self.ly_phase = curr;
            self.ly_prev_d = curr_d;
            let chaos_damp = (1.0 - self.ly_lam.max(0.0)).clamp(0.1, 1.0);
            out[31] = EngineSignal::sanitized(32, Timeframe::Sec15, tanh_s(ret_bps, 2.0) * chaos_damp, 0.65);
        }

        // E33: Kolmogorov-Smirnov (K-S) Empirical Distribution Shift
        if has_prev {
            let step = [
                if ret_bps <= -1.2 { 1.0 } else { 0.0 },
                if ret_bps <= 0.0 { 1.0 } else { 0.0 },
                if ret_bps <= 1.2 { 1.0 } else { 0.0 },
                1.0,
            ];
            let mut ks_d = 0.0f64;
            for i in 0..4 {
                self.ks_fast[i] = (self.ks_fast[i] * 0.78) + (step[i] * 0.22);
                self.ks_slow[i] = (self.ks_slow[i] * 0.97) + (step[i] * 0.03);
                ks_d = ks_d.max((self.ks_fast[i] - self.ks_slow[i]).abs());
            }
            let dir_bias = self.ks_slow[1] - self.ks_fast[1];
            out[32] = EngineSignal::sanitized(33, Timeframe::Sec15, tanh_s(dir_bias * (1.0 + ks_d * 2.0), 0.35), 0.75);
        }

        // E34: Wasserstein Earth-Mover's Distance (L2 Optimal Transport)
        let tot_mass = inp.b2_sz + inp.bid_sz + inp.ask_sz + inp.a2_sz;
        if inp.has_book && tot_mass > 0.001 {
            let curr = [
                inp.b2_sz / tot_mass,
                (inp.b2_sz + inp.bid_sz) / tot_mass,
                (inp.b2_sz + inp.bid_sz + inp.ask_sz) / tot_mass,
                1.0,
            ];
            let mut w1 = 0.0;
            let mut shift = 0.0;
            for i in 0..3 {
                let d = curr[i] - self.w_cdf[i];
                w1 += d.abs();
                shift += d;
            }
            for i in 0..4 {
                self.w_cdf[i] = (self.w_cdf[i] * 0.70) + (curr[i] * 0.30);
            }
            out[33] = EngineSignal::sanitized(34, Timeframe::Sec1, tanh_s(shift * (1.0 + w1), 0.30), 0.75);
        }

        // E35: Discrete Hilbert Transform (Instantaneous Phase & Cycle Turn)
        if has_prev {
            self.hil_buf = [self.hil_buf[1], self.hil_buf[2], self.hil_buf[3], ret_bps];
            let in_p = 0.5 * self.hil_buf[1] + 0.5 * self.hil_buf[2];
            let quad = (-0.6366 * self.hil_buf[0])
                + (-0.2122 * self.hil_buf[1])
                + (0.2122 * self.hil_buf[2])
                + (0.6366 * self.hil_buf[3]);
            let amp = (in_p * in_p + quad * quad).sqrt();
            let phi_deg = quad.atan2(in_p).to_degrees();
            let cyc_dir = if phi_deg > -135.0 && phi_deg < -45.0 {
                1.0
            } else if phi_deg > 45.0 && phi_deg < 135.0 {
                -1.0
            } else {
                0.0
            };
            out[34] = EngineSignal::sanitized(35, Timeframe::Sec15, cyc_dir * tanh_s(amp, 1.5), 0.65);
        }

        // E36: BSM Synthetic Delta & Vega Dealer Hedge Impulse
        if self.bsm_k <= 0.0 {
            self.bsm_k = price;
        } else {
            let ret_abs = ((price - self.bsm_k) / self.bsm_k).abs();
            self.bsm_sig = ((self.bsm_sig * 0.90) + (ret_abs * 0.10)).max(0.0005);
            self.bsm_k = (self.bsm_k * 0.92) + (price * 0.08);
            let d1 = ((price / self.bsm_k).ln() + (0.5 * self.bsm_sig * self.bsm_sig)) / self.bsm_sig;
            let syn_delta = 1.0 / (1.0 + (-1.702 * d1).exp());
            let pdf_d1 = (-0.5 * d1 * d1).exp() * 0.39894228;
            let syn_vega = pdf_d1 * self.bsm_sig * 10_000.0;
            let impulse = ((syn_delta - self.bsm_delta) * 100.0) + (syn_vega * d1.signum() * 0.1);
            self.bsm_delta = syn_delta;
            out[35] = EngineSignal::sanitized(36, Timeframe::Sec15, tanh_s(impulse, 3.0), 0.70);
        }

        // E37: Vanna & Volga Second-Order Cross-Greek Flow
        if self.vv_k <= 0.0 {
            self.vv_k = price;
        } else {
            let ret_s = (price - self.vv_k) / self.vv_k;
            let next_sig = ((self.vv_sig * 0.88) + (ret_s.abs() * 0.12)).max(0.0005);
            let d_sig = next_sig - self.vv_sig;
            self.vv_sig = next_sig;
            self.vv_k = (self.vv_k * 0.90) + (price * 0.10);
            let d1 = ((price / self.vv_k).ln() + (0.5 * next_sig * next_sig)) / next_sig;
            let d2 = d1 - next_sig;
            let pdf_d1 = (-0.5 * d1 * d1).exp() * 0.39894228;
            let vanna = (-pdf_d1 * (d2 / next_sig)).clamp(-50.0, 50.0);
            let flow = (vanna * d_sig * 10_000.0) + (ret_s * 1_000.0);
            self.vv_flow = (self.vv_flow * 0.75) + (flow * 0.25);
            out[36] = EngineSignal::sanitized(37, Timeframe::Sec15, tanh_s(self.vv_flow, 2.0), 0.70);
        }

        // E38: Charm (Delta Bleed) & Vera (Vega Decay)
        if self.cv_k <= 0.0 {
            self.cv_k = price;
        } else {
            let ret_abs = ((price - self.cv_k) / self.cv_k).abs();
            self.cv_sig = ((self.cv_sig * 0.90) + (ret_abs * 0.10)).max(0.0005);
            self.cv_k = (self.cv_k * 0.93) + (price * 0.07);
            self.cv_tau = if self.cv_tau <= 0.18 { 1.0 } else { self.cv_tau - 0.04 };
            let st = self.cv_tau.sqrt();
            let d1 = ((price / self.cv_k).ln() + (0.5 * self.cv_sig * self.cv_sig * self.cv_tau)) / (self.cv_sig * st);
            let d2 = d1 - (self.cv_sig * st);
            let pdf = (-0.5 * d1 * d1).exp() * 0.39894228;
            let charm = (pdf * (d2 / (2.0 * self.cv_tau))).clamp(-25.0, 25.0);
            let vera = (-pdf * st * (0.5 + (d1 * d2) / (2.0 * self.cv_tau))).clamp(-25.0, 25.0);
            self.cv_bleed = (self.cv_bleed * 0.75) + (((charm * 1.5) + (vera * 0.5)) * 0.25);
            out[37] = EngineSignal::sanitized(38, Timeframe::Min1, tanh_s(self.cv_bleed, 1.5), 0.65);
        }

        // E39: Speed & Zomma Third-Order Gamma Squeeze
        if self.sz_k <= 0.0 {
            self.sz_k = price;
        } else {
            let ret_s = (price - self.sz_k) / self.sz_k;
            self.sz_sig = ((self.sz_sig * 0.88) + (ret_s.abs() * 0.12)).max(0.0005);
            self.sz_k = (self.sz_k * 0.92) + (price * 0.08);
            let d1 = ((price / self.sz_k).ln() + (0.5 * self.sz_sig * self.sz_sig)) / self.sz_sig;
            let d2 = d1 - self.sz_sig;
            let pdf = (-0.5 * d1 * d1).exp() * 0.39894228;
            let gamma = pdf / (price * self.sz_sig);
            let speed = (-gamma / price * (d1 / self.sz_sig + 1.0) * 100_000.0).clamp(-100.0, 100.0);
            let zomma = (gamma * (d1 * d2 - 1.0) / self.sz_sig).clamp(-50.0, 50.0);
            let sqz = (speed * 0.6) + (zomma * 0.4) + (ret_s * 500.0);
            self.sz_squeeze = (self.sz_squeeze * 0.70) + (sqz * 0.30);
            out[38] = EngineSignal::sanitized(39, Timeframe::Sec15, tanh_s(self.sz_squeeze, 3.0), 0.70);
        }

        // E40: Almgren-Chriss Optimal Execution Trajectory
        if has_prev {
            self.ac_var = ((self.ac_var * 0.88) + (ret_bps * ret_bps * 0.12)).max(0.05);
            let eta = (spread_bps / top_depth.sqrt()).clamp(0.05, 50.0);
            let kappa = ((0.35 * self.ac_var) / eta).sqrt().clamp(0.1, 5.0);
            let rate = (1.0 - ((kappa * 0.5).sinh() / kappa.sinh().max(1e-6))).clamp(0.0, 1.0);
            self.ac_kappa = (self.ac_kappa * 0.75) + (kappa * 0.25);
            self.ac_rate = (self.ac_rate * 0.75) + (rate * 0.25);
            out[39] = EngineSignal::sanitized(40, Timeframe::Sec1, tanh_s(ret_bps, 2.0) * self.ac_rate, 0.70);
        }

        // E41: Avellaneda-Stoikov Reservation Price Skew
        if has_prev {
            self.as_var = ((self.as_var * 0.88) + (ret_bps * ret_bps * 0.12)).max(0.05);
            let inv_q = (inp.ask_sz - inp.bid_sz) / top_depth;
            let res_skew = -inv_q * 0.25 * self.as_var;
            self.as_skew = (self.as_skew * 0.75) + (res_skew * 0.25);
            out[40] = EngineSignal::sanitized(41, Timeframe::Sec1, tanh_s(self.as_skew, 0.90), 0.80);
        }

        // E42: Nash Equilibrium Bimatrix Game Theory (Predator vs Maker)
        if has_prev {
            let abs_r = ret_bps.abs();
            let m11 = abs_r - (0.5 * spread_bps);
            let m12 = -spread_bps;
            let m21 = -0.3 * abs_r;
            let m22 = imb_l1.abs() * 1.5;
            let denom = (m11 - m12 - m21 + m22).abs().max(0.1);
            let p_sw = ((m22 - m21).abs() / denom).clamp(0.05, 0.95);
            let q_rt = ((m22 - m12).abs() / denom).clamp(0.05, 0.95);
            self.nash_p = (self.nash_p * 0.75) + (p_sw * 0.25);
            self.nash_q = (self.nash_q * 0.75) + (q_rt * 0.25);
            let edge = (self.nash_p - 0.5) * 2.0;
            out[41] = EngineSignal::sanitized(42, Timeframe::Sec1, ret_bps.signum() * edge.max(0.0), 0.70);
        }

        // E43: Lotka-Volterra Predator-Prey Liquidity Dynamics
        if has_prev {
            let mass = top_depth.ln().max(0.2);
            let x = (self.lv_x * 0.80) + (mass * 0.20);
            let y = (self.lv_y * 0.75) + (ret_bps.abs().max(0.1) * 0.25);
            let dx = (0.40 * x) - (0.30 * x * y);
            let dy = (0.25 * x * y) - (0.35 * y);
            self.lv_x = (x + dx * 0.15).clamp(0.05, 15.0);
            self.lv_y = (y + dy * 0.15).clamp(0.05, 15.0);
            let ratio = self.lv_y / self.lv_x;
            out[42] = EngineSignal::sanitized(43, Timeframe::Sec1, ret_bps.signum() * tanh_s(ratio - 1.0, 1.2), 0.65);
        }

        // E44: Continuous-Time Merton-Kelly Optimal Leverage
        if has_prev {
            self.kel_mu = (self.kel_mu * 0.85) + (ret_bps * 0.15);
            let d = ret_bps - self.kel_mu;
            self.kel_var = ((self.kel_var * 0.85) + (d * d * 0.15)).max(0.10);
            let net_edge = if self.kel_mu.abs() > 0.15 {
                self.kel_mu.signum() * (self.kel_mu.abs() - 0.15)
            } else {
                0.0
            };
            let half_kelly = (0.5 * (net_edge / self.kel_var)).clamp(-5.0, 5.0);
            out[43] = EngineSignal::sanitized(44, Timeframe::Sec15, tanh_s(half_kelly, 1.5), 0.75);
        }

        // E45: Extreme Value Theory (EVT) Generalized Pareto Tail Shield
        if has_prev {
            let abs_r = ret_bps.abs();
            self.evt_u = ((self.evt_u * 0.92) + (abs_r * 0.12)).clamp(0.4, 8.0);
            let excess = (abs_r - self.evt_u).max(0.0);
            self.evt_ex = if excess > 0.0 {
                (self.evt_ex * 0.80) + (excess * 0.20)
            } else {
                (self.evt_ex * 0.96).max(0.15)
            };
            let xi = ((self.evt_ex / (self.evt_u + self.evt_ex)) * 0.65).clamp(0.05, 0.45);
            let beta = (self.evt_ex * (1.0 - xi)).max(0.1);
            let var_99 = self.evt_u + (beta / xi) * ((10.0_f64).powf(xi) - 1.0);
            let tail_safe = if abs_r > var_99 || xi > 0.32 { 0.15 } else { 1.0 };
            out[44] = EngineSignal::sanitized(45, Timeframe::Sec1, tanh_s(ret_bps, 2.5) * tail_safe, 0.70);
        }

        self.last_price = price;
    }
}
/// Stack-only state for custom Copula, PDE, Stoch-Vol, Levy & Econophysics Engines 46..60
#[derive(Debug, Clone, Copy)]
pub struct Engines46To60 {
    last_price: f64,
    // E46 Clayton & Gumbel Copula
    cop_tau: f64,
    // E47 Fokker-Planck PDE
    fp_mu: f64,
    fp_var: f64,
    fp_grid: [f64; 4],
    // E48 Feynman-Kac Path-Integral
    fk_mu: f64,
    fk_sig: f64,
    fk_w: [f64; 4],
    // E49 HJB Optimal Control
    hjb_mu: f64,
    hjb_var: f64,
    hjb_u: f64,
    // E50 Heston-CIR
    hes_v: f64,
    hes_theta: f64,
    hes_rho: f64,
    // E51 SABR Smile
    sabr_f: f64,
    sabr_alpha: f64,
    sabr_rho: f64,
    sabr_nu: f64,
    // E52 Merton Jump-Diffusion
    mjd_sig: f64,
    mjd_lam: f64,
    mjd_k: f64,
    mjd_drift: f64,
    // E53 Bates SVJ Hybrid
    bat_v: f64,
    bat_jv: f64,
    bat_imp: f64,
    // E54 Variance Gamma Levy
    vg_theta: f64,
    vg_sig2: f64,
    vg_nu: f64,
    // E55 NIG Hyperbolic Levy
    nig_alpha: f64,
    nig_beta: f64,
    nig_delta: f64,
    // E56 Ising Spin Lattice
    is_s1: f64,
    is_mag: f64,
    is_temp: f64,
    // E57 Kuramoto Phase Sync (Self + BTC Anchor)
    kur_th_s: f64,
    kur_th_b: f64,
    kur_om_s: f64,
    kur_om_b: f64,
    // E58 Boltzmann Free Energy
    bol_t: f64,
    bol_e: [f64; 4],
    // E59 Tsallis q-Entropy
    tsa_var: f64,
    tsa_q: f64,
    tsa_p: [f64; 4],
    // E60 Renyi Multifractal
    ren_p: [f64; 4],
}

impl Engines46To60 {
    pub const fn new() -> Self {
        Self {
            last_price: 0.0,
            cop_tau: 0.4,
            fp_mu: 0.0,
            fp_var: 1.0,
            fp_grid: [0.25; 4],
            fk_mu: 0.0,
            fk_sig: 1.0,
            fk_w: [0.25; 4],
            hjb_mu: 0.0,
            hjb_var: 1.0,
            hjb_u: 0.0,
            hes_v: 1.0,
            hes_theta: 1.0,
            hes_rho: 0.0,
            sabr_f: 0.0,
            sabr_alpha: 1.0,
            sabr_rho: 0.0,
            sabr_nu: 0.5,
            mjd_sig: 0.8,
            mjd_lam: 0.05,
            mjd_k: 0.0,
            mjd_drift: 0.0,
            bat_v: 0.8,
            bat_jv: 0.2,
            bat_imp: 0.0,
            vg_theta: 0.0,
            vg_sig2: 1.0,
            vg_nu: 0.25,
            nig_alpha: 1.5,
            nig_beta: 0.0,
            nig_delta: 0.8,
            is_s1: 0.0,
            is_mag: 0.0,
            is_temp: 1.0,
            kur_th_s: 0.0,
            kur_th_b: 1.0,
            kur_om_s: 0.0,
            kur_om_b: 0.0,
            bol_t: 1.0,
            bol_e: [1.0; 4],
            tsa_var: 1.0,
            tsa_q: 1.35,
            tsa_p: [0.25; 4],
            ren_p: [0.25; 4],
        }
    }

    pub fn eval(&mut self, inp: &EngineTickInput, out: &mut [EngineSignal; NUM_ALPHA_ENGINES]) {
        let price = inp.price;
        if !price.is_finite() || price <= 0.0 {
            return;
        }
        let prev_p = self.last_price;
        let has_prev = prev_p > 0.0;
        let ret_bps = if has_prev { ((price - prev_p) / prev_p) * 10_000.0 } else { 0.0 };
        let spread_bps = if inp.bid > 0.0 && inp.ask > inp.bid {
            (((inp.ask - inp.bid) / price) * 10_000.0).max(0.1)
        } else {
            0.2
        };
        let top_depth = (inp.bid_sz + inp.ask_sz).max(0.1);
        let imb_l1 = ((inp.bid_sz - inp.ask_sz) / top_depth).clamp(-1.0, 1.0);

        // E46: Clayton & Gumbel Archimedean Copula Asymmetric Tail-Dependence
        if has_prev {
            let u_curr = (1.0 / (1.0 + (-1.15 * ret_bps).exp())).clamp(0.01, 0.99);
            let u_btc = (1.0 / (1.0 + (-1.15 * inp.btc_anchor_ret_bps).exp())).clamp(0.01, 0.99);
            let concord = ((u_curr - 0.5) * (u_btc - 0.5) * 4.0).clamp(-0.9, 0.95);
            self.cop_tau = ((self.cop_tau * 0.88) + (concord * 0.12)).clamp(0.05, 0.90);
            let theta_c = ((2.0 * self.cop_tau) / (1.0 - self.cop_tau)).max(0.1);
            let lam_low = (2.0_f64).powf(-1.0 / theta_c);
            let theta_g = (1.0 / (1.0 - self.cop_tau)).max(1.02);
            let lam_up = (2.0 - (2.0_f64).powf(1.0 / theta_g)).clamp(0.0, 0.99);
            let tail_sig = if u_curr > 0.5 {
                (u_curr - 0.5) * 2.0 * lam_up
            } else {
                (u_curr - 0.5) * 2.0 * lam_low
            };
            out[45] = EngineSignal::sanitized(46, Timeframe::Sec15, tail_sig, 0.70);
        }

        // E47: Fokker-Planck Probability Density Evolution (4-Bin Forward PDE)
        if has_prev {
            self.fp_mu = (self.fp_mu * 0.85) + (ret_bps * 0.15);
            let d = ret_bps - self.fp_mu;
            self.fp_var = ((self.fp_var * 0.85) + (d * d * 0.15)).clamp(0.05, 10.0);
            let adv = (self.fp_mu * 0.08).clamp(-0.20, 0.20);
            let dr = (self.fp_var * 0.02).clamp(0.01, 0.15);
            let g = self.fp_grid;
            let dp = [
                (-adv * g[0]) + dr * (g[1] - g[0]),
                (adv * (g[0] - g[1])) + dr * (g[0] - 2.0 * g[1] + g[2]),
                (adv * (g[1] - g[2])) + dr * (g[1] - 2.0 * g[2] + g[3]),
                (adv * g[2]) + dr * (g[2] - g[3]),
            ];
            let mut sum = 0.0;
            for i in 0..4 {
                self.fp_grid[i] = (g[i] + dp[i]).max(0.01);
                sum += self.fp_grid[i];
            }
            for i in 0..4 { self.fp_grid[i] /= sum.max(1e-6); }
            let flux = (self.fp_grid[2] + self.fp_grid[3]) - (self.fp_grid[0] + self.fp_grid[1]);
            out[46] = EngineSignal::sanitized(47, Timeframe::Sec15, tanh_s(flux, 0.40), 0.75);
        }

        // E48: Feynman-Kac Stochastic PDE Path-Integral Expectancy Pricer
        if has_prev {
            self.fk_mu = (self.fk_mu * 0.85) + (ret_bps * 0.15);
            self.fk_sig = ((self.fk_sig * 0.85) + ((ret_bps - self.fk_mu).abs() * 0.15)).max(0.20);
            let paths = [-2.0 * self.fk_sig, -0.5 * self.fk_sig, 0.5 * self.fk_sig, 2.0 * self.fk_sig];
            let two_var = 2.0 * self.fk_sig * self.fk_sig;
            let pot_v = (ret_bps.abs() * 0.02).clamp(0.0, 0.5);
            let mut raw = [0.0; 4];
            let mut rsum = 0.0;
            for i in 0..4 {
                let act = ((paths[i] - self.fk_mu) * (paths[i] - self.fk_mu)) / two_var;
                raw[i] = (-act - pot_v).exp();
                rsum += raw[i];
            }
            let mut exp_bps = 0.0;
            for i in 0..4 {
                self.fk_w[i] = (self.fk_w[i] * 0.70) + ((raw[i] / rsum.max(1e-6)) * 0.30);
                exp_bps += self.fk_w[i] * paths[i];
            }
            out[47] = EngineSignal::sanitized(48, Timeframe::Sec15, tanh_s(exp_bps, 0.90), 0.75);
        }

        // E49: Hamilton-Jacobi-Bellman (HJB) Stochastic Optimal Control
        if has_prev {
            self.hjb_mu = (self.hjb_mu * 0.84) + (ret_bps * 0.16);
            let d = ret_bps - self.hjb_mu;
            self.hjb_var = ((self.hjb_var * 0.86) + (d * d * 0.14)).max(0.08);
            let eta = spread_bps * 0.35;
            let denom = (0.40 * self.hjb_var) + (2.0 * eta);
            let u_raw = (self.hjb_mu / denom).clamp(-3.0, 3.0);
            self.hjb_u = (self.hjb_u * 0.70) + (u_raw * 0.30);
            out[48] = EngineSignal::sanitized(49, Timeframe::Sec1, tanh_s(self.hjb_u, 1.0), 0.80);
        }

        // E50: Heston-CIR Stochastic Volatility & Spot-Vol Leverage
        if has_prev {
            let inst_v = (ret_bps * ret_bps).max(0.02);
            let vp = self.hes_v.max(0.02);
            self.hes_theta = ((self.hes_theta * 0.96) + (inst_v * 0.04)).max(0.10);
            let dv = inst_v - vp;
            let xi = (dv.abs() / vp.sqrt()).clamp(0.10, 5.0);
            self.hes_v = (vp + 0.25 * (self.hes_theta - vp) + (0.20 * dv)).max(0.02);
            let cov_s = (ret_bps * dv).clamp(-10.0, 10.0) / 10.0;
            self.hes_rho = ((self.hes_rho * 0.85) + (cov_s * 0.15)).clamp(-0.98, 0.98);
            let feller = (0.50 * self.hes_theta) / (xi * xi);
            let vol_ratio = (self.hes_v / self.hes_theta).clamp(0.2, 4.0);
            let h_sig = self.hes_rho * tanh_s(vol_ratio - 1.0, 1.0) * feller.clamp(0.25, 1.0);
            out[49] = EngineSignal::sanitized(50, Timeframe::Sec15, h_sig, 0.70);
        }

        // E51: SABR Volatility Smile & Skew Curvature
        if self.sabr_f <= 0.0 {
            self.sabr_f = price;
        } else {
            let rb = ((price - self.sabr_f) / self.sabr_f) * 10_000.0;
            let ar = rb.abs().max(0.05);
            self.sabr_f = (self.sabr_f * 0.90) + (price * 0.10);
            let da = ar - self.sabr_alpha;
            self.sabr_alpha = ((self.sabr_alpha * 0.85) + (ar * 0.15)).clamp(0.10, 25.0);
            self.sabr_rho = ((self.sabr_rho * 0.88) + ((rb.signum() * da.signum() * 0.85) * 0.12)).clamp(-0.95, 0.95);
            self.sabr_nu = ((self.sabr_nu * 0.85) + ((da.abs() / self.sabr_alpha) * 0.15)).clamp(0.10, 3.5);
            let z = (self.sabr_nu / self.sabr_alpha) * (price / self.sabr_f).ln() * 100.0;
            let skew_shift = (0.5 * self.sabr_rho * self.sabr_nu * z)
                + ((2.0 - 3.0 * self.sabr_rho * self.sabr_rho) / 24.0) * (self.sabr_nu * self.sabr_nu * z * z);
            out[50] = EngineSignal::sanitized(51, Timeframe::Sec15, self.sabr_rho * tanh_s(skew_shift.abs() + 0.1, 0.35), 0.70);
        }

        // E52: Merton Jump-Diffusion (Poisson Jump Intensity & Compensated Drift)
        if has_prev {
            let ar = ret_bps.abs();
            let is_j = if ar > (2.2 * self.mjd_sig) && ar > 0.6 { 1.0 } else { 0.0 };
            self.mjd_sig = if is_j > 0.5 {
                (self.mjd_sig * 0.96) + (ar * 0.04)
            } else {
                ((self.mjd_sig * 0.88) + (ar * 0.12)).clamp(0.15, 10.0)
            };
            self.mjd_lam = ((self.mjd_lam * 0.90) + (is_j * 0.10)).clamp(0.01, 0.99);
            self.mjd_k = if is_j > 0.5 { (self.mjd_k * 0.65) + (ret_bps * 0.35) } else { self.mjd_k * 0.92 };
            self.mjd_drift = (self.mjd_drift * 0.80) + ((ret_bps - (self.mjd_lam * self.mjd_k)) * 0.20);
            out[51] = EngineSignal::sanitized(52, Timeframe::Sec1, tanh_s(self.mjd_k * self.mjd_lam + self.mjd_drift * 0.5, 1.0), 0.75);
        }

        // E53: Bates SVJ (Stochastic Vol + Coupled Price-Vol Co-Jump)
        if has_prev {
            let iv = (ret_bps * ret_bps).max(0.02);
            let vp = self.bat_v.max(0.05);
            let cj = if iv > (3.5 * vp) && iv > 1.2 { iv - vp } else { 0.0 };
            self.bat_v = ((vp * 0.88) + (iv.min(3.0 * vp) * 0.12)).clamp(0.05, 25.0);
            self.bat_jv = ((self.bat_jv * 0.82) + (cj * 0.18)).clamp(0.01, 50.0);
            let j_dom = self.bat_jv / (self.bat_v + self.bat_jv).max(0.05);
            self.bat_imp = (self.bat_imp * 0.70) + ((ret_bps * j_dom) * 0.30);
            out[52] = EngineSignal::sanitized(53, Timeframe::Sec1, tanh_s(self.bat_imp, 1.2), 0.75);
        }

        // E54: Variance Gamma (Madan-Carr-Chang Pure Jump Levy Process)
        if has_prev {
            self.vg_theta = (self.vg_theta * 0.85) + ((ret_bps.max(0.0) - (-ret_bps).max(0.0)) * 0.15);
            let csq = (ret_bps - self.vg_theta) * (ret_bps - self.vg_theta);
            self.vg_sig2 = ((self.vg_sig2 * 0.86) + (csq * 0.14)).clamp(0.05, 30.0);
            let n4 = (csq * csq) / (self.vg_sig2 * self.vg_sig2).max(0.01);
            let inu = ((n4 - 3.0).max(0.0) / 3.0).clamp(0.05, 2.5);
            self.vg_nu = (self.vg_nu * 0.88) + (inu * 0.12);
            out[53] = EngineSignal::sanitized(54, Timeframe::Sec15, tanh_s(self.vg_theta * (1.0 + self.vg_nu), 1.2), 0.70);
        }

        // E55: Normal Inverse Gaussian (NIG) Hyperbolic Heavy-Tail Levy
        if has_prev {
            let ar = ret_bps.abs();
            self.nig_delta = ((self.nig_delta * 0.86) + (ar * 0.14)).clamp(0.15, 12.0);
            let iasym = (ret_bps / self.nig_delta).clamp(-2.5, 2.5);
            self.nig_beta = (self.nig_beta * 0.84) + (iasym * 0.16);
            let tsh = (ar / self.nig_delta).clamp(0.2, 4.0);
            let ra = (self.nig_alpha * 0.85) + ((1.8 / tsh.sqrt()) * 0.15);
            self.nig_alpha = ra.max(self.nig_beta.abs() + 0.15).clamp(0.25, 5.0);
            let g_sub = ((self.nig_alpha * self.nig_alpha) - (self.nig_beta * self.nig_beta)).max(0.01).sqrt();
            let nig_drift = self.nig_delta * (self.nig_beta / g_sub);
            out[54] = EngineSignal::sanitized(55, Timeframe::Sec15, tanh_s(nig_drift, 1.1), 0.70);
        }

        // E56: Ising Ferromagnetic Spin Lattice & Phase-Transition Herd Detector
        if has_prev {
            let s1 = (ret_bps * 1.5).tanh();
            let s2 = (imb_l1 * 2.5).tanh();
            let s3 = ((self.is_s1 + s1) * 0.8).tanh();
            let svar = ((s1 - s2).powi(2) + (s1 - s3).powi(2)) * 0.5;
            self.is_temp = ((self.is_temp * 0.85) + (svar * 0.15)).clamp(0.15, 3.0);
            let h_ext = (s1 + s2 + s3) / 3.0;
            let imag = (((1.25 * self.is_mag) + h_ext) / self.is_temp).tanh();
            self.is_mag = (self.is_mag * 0.70) + (imag * 0.30);
            self.is_s1 = s1;
            let cool_gate = (1.6 / (self.is_temp + 0.4)).clamp(0.2, 1.0);
            out[55] = EngineSignal::sanitized(56, Timeframe::Sec1, self.is_mag * cool_gate, 0.80);
        }

        // E57: Kuramoto Coupled Oscillator Cross-Asset Phase-Sync (Self <-> BTC)
        if has_prev {
            self.kur_om_s = (self.kur_om_s * 0.80) + (ret_bps * 0.20);
            self.kur_om_b = (self.kur_om_b * 0.80) + (inp.btc_anchor_ret_bps * 0.20);
            let d_s = (self.kur_om_s * 0.15) + (0.225 * (self.kur_th_b - self.kur_th_s).sin());
            let d_b = (self.kur_om_b * 0.15) + (0.225 * (self.kur_th_s - self.kur_th_b).sin());
            self.kur_th_s = (self.kur_th_s + d_s) % std::f64::consts::TAU;
            self.kur_th_b = (self.kur_th_b + d_b) % std::f64::consts::TAU;
            let mc = (self.kur_th_s.cos() + self.kur_th_b.cos()) * 0.5;
            let ms = (self.kur_th_s.sin() + self.kur_th_b.sin()) * 0.5;
            let sync_r = (mc * mc + ms * ms).sqrt().clamp(0.0, 1.0);
            let col_vel = (self.kur_om_s + self.kur_om_b) * 0.5;
            out[56] = EngineSignal::sanitized(57, Timeframe::Sec15, tanh_s(col_vel, 1.0) * sync_r, 0.70);
        }

        // E58: Boltzmann-Gibbs Helmholtz Free Energy & Thermodynamic Entropy
        if has_prev {
            self.bol_t = ((self.bol_t * 0.86) + (ret_bps.abs().max(0.1) * 0.14)).clamp(0.20, 8.0);
            let grid = [-2.0, -0.7, 0.7, 2.0];
            let mut bw = [0.0; 4];
            let mut z = 0.0;
            for i in 0..4 {
                let ie = ((grid[i] - ret_bps).powi(2) * 0.25).clamp(0.05, 15.0);
                self.bol_e[i] = (self.bol_e[i] * 0.75) + (ie * 0.25);
                bw[i] = (-self.bol_e[i] / self.bol_t).exp();
                z += bw[i];
            }
            let mut p = [0.0; 4];
            let mut ent = 0.0;
            for i in 0..4 {
                p[i] = bw[i] / z.max(1e-6);
                if p[i] > 1e-6 { ent -= p[i] * p[i].ln(); }
            }
            let work_bias = (p[2] + p[3]) - (p[0] + p[1]);
            let ord = (1.0 - (ent / 1.3863)).clamp(0.1, 1.0);
            out[57] = EngineSignal::sanitized(58, Timeframe::Sec15, work_bias * ord, 0.70);
        }

        // E59: Tsallis Non-Extensive q-Entropy & Power-Law q-Gaussian Cascade
        if has_prev {
            let sq = ret_bps * ret_bps;
            self.tsa_var = ((self.tsa_var * 0.88) + (sq * 0.12)).max(0.05);
            let tr = (sq / self.tsa_var).clamp(0.1, 12.0);
            let iq = (1.10 + (0.12 * tr)).clamp(1.05, 2.40);
            self.tsa_q = (self.tsa_q * 0.85) + (iq * 0.15);
            let states = [-2.0, -0.6, 0.6, 2.0];
            let beta = 1.0 / (2.0 * self.tsa_var);
            let mut qw = [0.0; 4];
            let mut qs = 0.0;
            for i in 0..4 {
                let u = beta * (states[i] - ret_bps).powi(2);
                let base = (1.0 - (1.0 - self.tsa_q) * u).max(1e-4);
                qw[i] = base.powf(1.0 / (1.0 - self.tsa_q));
                qs += qw[i];
            }
            for i in 0..4 {
                self.tsa_p[i] = (self.tsa_p[i] * 0.72) + ((qw[i] / qs.max(1e-6)) * 0.28);
            }
            let q_skew = (self.tsa_p[2] + self.tsa_p[3]) - (self.tsa_p[0] + self.tsa_p[1]);
            out[58] = EngineSignal::sanitized(59, Timeframe::Sec15, tanh_s(q_skew * self.tsa_q, 0.85), 0.70);
        }

        // E60: Renyi Generalized Multi-Fractal Entropy Spectrum
        if has_prev {
            let centers = [-2.0, -0.5, 0.5, 2.0];
            let mut rw = [0.0; 4];
            let mut rs = 0.0;
            for i in 0..4 {
                rw[i] = (-0.45 * (ret_bps - centers[i]).powi(2)).exp();
                rs += rw[i];
            }
            let mut psum = 0.0;
            for i in 0..4 {
                self.ren_p[i] = (self.ren_p[i] * 0.78) + ((rw[i] / rs.max(1e-6)) * 0.22);
                psum += self.ren_p[i];
            }
            let mut sa05 = 0.0;
            let mut sa4 = 0.0;
            for i in 0..4 {
                self.ren_p[i] /= psum.max(1e-6);
                sa05 += self.ren_p[i].max(1e-6).sqrt();
                sa4 += self.ren_p[i].powi(4);
            }
            let h05 = 2.0 * sa05.ln();
            let h4 = (-1.0 / 3.0) * sa4.max(1e-9).ln();
            let spec_w = (h05 - h4).max(0.0);
            let dir_b = (self.ren_p[2] + self.ren_p[3]) - (self.ren_p[0] + self.ren_p[1]);
            out[59] = EngineSignal::sanitized(60, Timeframe::Sec15, tanh_s(dir_b * (0.5 + spec_w), 0.70), 0.70);
        }

        self.last_price = price;
    }
}
/// Stack-only state for custom Info-Geometry, Quantum & Field-Theory Engines 61..75
#[derive(Debug, Clone, Copy)]
pub struct Engines61To75 {
    last_price: f64,
    // E61 KL Divergence Shift
    kl_p: [f64; 4],
    kl_q: [f64; 4],
    // E62 Wasserstein EMD
    w62_p: [f64; 4],
    w62_q: [f64; 4],
    // E63 Fisher Information Manifold
    fish_mu: f64,
    fish_var: f64,
    // E64 Schrodinger WKB Tunneling
    sch_e: f64,
    sch_v: f64,
    sch_tp: f64,
    // E65 Heisenberg Uncertainty
    heis_dx: f64,
    heis_dp: f64,
    heis_prod: f64,
    // E66 Quantum Harmonic Oscillator
    qho_anchor: f64,
    qho_x: f64,
    qho_n: f64,
    // E67 Langevin Dynamics
    lang_v: f64,
    lang_g: f64,
    lang_kbt: f64,
    // E68 Ornstein-Uhlenbeck Half-Life
    ou68_mu: f64,
    ou68_x: f64,
    ou68_th: f64,
    ou68_sig: f64,
    // E69 Kramers Escape
    kra_anchor: f64,
    kra_x: f64,
    kra_d: f64,
    kra_k: f64,
    // E70 Landau-Ginzburg
    lg_phi: f64,
    lg_r: f64,
    // E71 Renormalization Group
    rg_g: f64,
    rg_beta: f64,
    // E72 Topological Hall
    topo_d: [f64; 3],
    // E73 Bose-Fermi Condensate
    bf_t: f64,
    bf_mu: f64,
    bf_bec: f64,
    // E74 Casimir Vacuum Force
    cas_d: f64,
    cas_f: f64,
    cas_drv: f64,
    // E75 Feynman Path Integral
    fey_s: [f64; 3],
    fey_coh: f64,
}

impl Engines61To75 {
    pub const fn new() -> Self {
        Self {
            last_price: 0.0,
            kl_p: [0.25; 4],
            kl_q: [0.25; 4],
            w62_p: [0.25; 4],
            w62_q: [0.25; 4],
            fish_mu: 0.0,
            fish_var: 1.0,
            sch_e: 0.5,
            sch_v: 1.5,
            sch_tp: 0.1,
            heis_dx: 1.0,
            heis_dp: 1.0,
            heis_prod: 1.0,
            qho_anchor: 0.0,
            qho_x: 0.0,
            qho_n: 0.0,
            lang_v: 0.0,
            lang_g: 0.5,
            lang_kbt: 1.0,
            ou68_mu: 0.0,
            ou68_x: 0.0,
            ou68_th: 0.25,
            ou68_sig: 1.0,
            kra_anchor: 0.0,
            kra_x: 0.0,
            kra_d: 1.0,
            kra_k: 0.05,
            lg_phi: 0.0,
            lg_r: 0.5,
            rg_g: 0.5,
            rg_beta: 0.0,
            topo_d: [0.1, 0.1, 1.0],
            bf_t: 1.0,
            bf_mu: -0.5,
            bf_bec: 0.0,
            cas_d: 1.0,
            cas_f: 0.5,
            cas_drv: 0.0,
            fey_s: [1.0; 3],
            fey_coh: 0.5,
        }
    }

    pub fn eval(&mut self, inp: &EngineTickInput, out: &mut [EngineSignal; NUM_ALPHA_ENGINES]) {
        let price = inp.price;
        if !price.is_finite() || price <= 0.0 {
            return;
        }
        let prev_p = self.last_price;
        let has_prev = prev_p > 0.0;
        let ret_bps = if has_prev { ((price - prev_p) / prev_p) * 10_000.0 } else { 0.0 };
        let spread_bps = if inp.bid > 0.0 && inp.ask > inp.bid {
            (((inp.ask - inp.bid) / price) * 10_000.0).clamp(0.10, 10.0)
        } else {
            0.25
        };
        let top_depth = (inp.bid_sz + inp.ask_sz).max(0.1);
        let imb_l1 = ((inp.bid_sz - inp.ask_sz) / top_depth).clamp(-1.0, 1.0);

        // E61: Kullback-Leibler (KL) Divergence & Jeffreys Relative Entropy Shift
        if has_prev {
            let centers = [-2.0, -0.5, 0.5, 2.0];
            let mut rw = [0.0; 4];
            let mut rs = 0.0;
            for i in 0..4 {
                rw[i] = (-0.50 * (ret_bps - centers[i]).powi(2)).exp();
                rs += rw[i];
            }
            let (mut ps, mut qs) = (0.0, 0.0);
            for i in 0..4 {
                let d = rw[i] / rs.max(1e-6);
                self.kl_p[i] = (self.kl_p[i] * 0.60) + (d * 0.40);
                self.kl_q[i] = (self.kl_q[i] * 0.94) + (d * 0.06);
                ps += self.kl_p[i];
                qs += self.kl_q[i];
            }
            let (mut d_pq, mut d_qp) = (0.0, 0.0);
            for i in 0..4 {
                self.kl_p[i] /= ps.max(1e-6);
                self.kl_q[i] /= qs.max(1e-6);
                let pi = self.kl_p[i].max(1e-6);
                let qi = self.kl_q[i].max(1e-6);
                d_pq += pi * (pi / qi).ln();
                d_qp += qi * (qi / pi).ln();
            }
            let j_div = (d_pq + d_qp).max(0.0);
            let info_dir = ((self.kl_p[2] + self.kl_p[3]) - (self.kl_q[2] + self.kl_q[3])) * 2.0;
            out[60] = EngineSignal::sanitized(61, Timeframe::Sec15, tanh_s(info_dir * (1.0 + j_div * 4.0), 0.35), 0.75);
        }

        // E62: Wasserstein Optimal Transport & Earth Mover's Distance (EMD)
        if has_prev {
            let comb = ret_bps + (imb_l1 * 0.8);
            let grid = [-2.0, -0.6, 0.6, 2.0];
            let mut rw = [0.0; 4];
            let mut rs = 0.0;
            for i in 0..4 {
                rw[i] = (-0.45 * (comb - grid[i]).powi(2)).exp();
                rs += rw[i];
            }
            let (mut ps, mut qs) = (0.0, 0.0);
            for i in 0..4 {
                let m = rw[i] / rs.max(1e-6);
                self.w62_p[i] = (self.w62_p[i] * 0.58) + (m * 0.42);
                self.w62_q[i] = (self.w62_q[i] * 0.93) + (m * 0.07);
                ps += self.w62_p[i];
                qs += self.w62_q[i];
            }
            for i in 0..4 {
                self.w62_p[i] /= ps.max(1e-6);
                self.w62_q[i] /= qs.max(1e-6);
            }
            let (mut cp, mut cq, mut w1, mut flux) = (0.0, 0.0, 0.0, 0.0);
            for i in 0..3 {
                cp += self.w62_p[i];
                cq += self.w62_q[i];
                let dx = grid[i + 1] - grid[i];
                let dc = cq - cp;
                w1 += dc.abs() * dx;
                flux += dc * dx;
            }
            out[61] = EngineSignal::sanitized(62, Timeframe::Sec1, tanh_s(flux * (1.0 + w1), 0.35), 0.75);
        }

        // E63: Fisher Information Metric & Cramer-Rao Riemannian Manifold
        if has_prev {
            let next_mu = (self.fish_mu * 0.84) + (ret_bps * 0.16);
            let diff = ret_bps - next_mu;
            let next_var = ((self.fish_var * 0.85) + (diff * diff * 0.15)).clamp(0.05, 25.0);
            let d_mu = next_mu - self.fish_mu;
            let d_var = next_var - self.fish_var;
            let g11 = 1.0 / next_var;
            let g22 = 1.0 / (2.0 * next_var * next_var);
            let ds = ((d_mu * d_mu * g11) + (d_var * d_var * g22)).max(0.0).sqrt();
            self.fish_mu = next_mu;
            self.fish_var = next_var;
            out[62] = EngineSignal::sanitized(63, Timeframe::Sec15, tanh_s(next_mu * (0.5 + ds), 0.85), 0.70);
        }

        // E64: Schrodinger Wavefunction & WKB Quantum Barrier Tunneling
        if has_prev {
            let inst_e = (0.5 * ret_bps * ret_bps).clamp(0.02, 25.0);
            let opp_wall = if ret_bps >= 0.0 { inp.ask_sz } else { inp.bid_sz };
            let inst_v = (opp_wall.max(1.0).ln() * 0.85).clamp(0.20, 15.0);
            self.sch_e = (self.sch_e * 0.78) + (inst_e * 0.22);
            self.sch_v = (self.sch_v * 0.85) + (inst_v * 0.15);
            let def = (self.sch_v - self.sch_e).max(0.0);
            let inst_t = if self.sch_e >= self.sch_v {
                0.99
            } else {
                (-0.45 * spread_bps.sqrt() * def.sqrt()).clamp(-10.0, 0.0).exp()
            };
            self.sch_tp = (self.sch_tp * 0.72) + (inst_t * 0.28);
            out[63] = EngineSignal::sanitized(64, Timeframe::Sec1, tanh_s(ret_bps, 1.5) * self.sch_tp, 0.75);
        }

        // E65: Heisenberg Uncertainty Principle & Minimum Wavepacket Bound
        if has_prev {
            let ar = ret_bps.abs();
            self.heis_dx = ((self.heis_dx * 0.84) + (ar * 0.16)).clamp(0.05, 50.0);
            self.heis_dp = ((self.heis_dp * 0.84) + (ar * 0.16)).clamp(0.05, 50.0);
            self.heis_prod = (self.heis_prod * 0.75) + ((self.heis_dx * self.heis_dp) * 0.25);
            let bound_ratio = self.heis_prod / 0.25;
            let coh = (1.5 / (bound_ratio + 0.5)).clamp(0.15, 1.0);
            out[64] = EngineSignal::sanitized(65, Timeframe::Sec15, tanh_s(ret_bps, 1.5) * coh, 0.65);
        }

        // E66: Quantum Harmonic Oscillator & Dirac Ladder Operators
        if self.qho_anchor <= 0.0 {
            self.qho_anchor = price;
        } else {
            let disp_x = ((price - self.qho_anchor) / self.qho_anchor) * 10_000.0;
            let mom_p = disp_x - self.qho_x;
            self.qho_anchor = (self.qho_anchor * 0.92) + (price * 0.08);
            let omega = (top_depth.max(1.0).ln() * 0.35).clamp(0.40, 3.0);
            let ham = (omega * disp_x * disp_x) + ((mom_p * mom_p) / omega);
            let inst_n = (ham - 0.5).clamp(0.0, 15.0);
            self.qho_n = (self.qho_n * 0.72) + (inst_n * 0.28);
            self.qho_x = disp_x;
            out[65] = EngineSignal::sanitized(66, Timeframe::Sec15, tanh_s(disp_x * (0.3 + self.qho_n * 0.25), 2.0), 0.70);
        }

        // E67: Langevin SDE & Fluctuation-Dissipation Theorem
        if has_prev {
            let ig = (top_depth.max(1.0).ln() * 0.22).clamp(0.10, 2.50);
            self.lang_g = (self.lang_g * 0.82) + (ig * 0.18);
            let dv = (-self.lang_g * self.lang_v * 0.25) + (ret_bps * 0.65);
            self.lang_v = (self.lang_v * 0.60) + (dv * 0.40);
            self.lang_kbt = ((self.lang_kbt * 0.85) + ((ret_bps * ret_bps).max(0.05) * 0.15)).clamp(0.05, 30.0);
            let einstein_d = self.lang_kbt / self.lang_g;
            out[66] = EngineSignal::sanitized(67, Timeframe::Sec1, tanh_s(self.lang_v * (einstein_d / 2.0).clamp(0.3, 1.5), 1.0), 0.75);
        }

        // E68: Ornstein-Uhlenbeck (OU) Mean-Reverting Velocity & Half-Life
        if self.ou68_mu <= 0.0 {
            self.ou68_mu = price;
        } else {
            let xc = ((price - self.ou68_mu) / self.ou68_mu) * 10_000.0;
            let dx = xc - self.ou68_x;
            self.ou68_mu = (self.ou68_mu * 0.94) + (price * 0.06);
            let ipull = if self.ou68_x.abs() > 0.15 { (-dx / self.ou68_x).clamp(0.02, 1.50) } else { self.ou68_th };
            self.ou68_th = ((self.ou68_th * 0.86) + (ipull * 0.14)).clamp(0.05, 1.20);
            let resid = (dx + (self.ou68_th * self.ou68_x)).abs();
            self.ou68_sig = ((self.ou68_sig * 0.85) + (resid * 0.15)).clamp(0.10, 20.0);
            let sig_eq = (self.ou68_sig / (2.0 * self.ou68_th).sqrt()).max(0.10);
            let ou_z = (xc / sig_eq).clamp(-5.0, 5.0);
            self.ou68_x = xc;
            let ou_sig = if self.ou68_th > 0.12 { -tanh_s(ou_z, 2.0) } else { tanh_s(ou_z, 2.5) };
            out[67] = EngineSignal::sanitized(68, Timeframe::Sec15, ou_sig, 0.75);
        }

        // E69: Kramers Escape Rate & Bistable Double-Well Barrier
        if self.kra_anchor <= 0.0 {
            self.kra_anchor = price;
        } else {
            let rb = ((price - self.kra_anchor) / self.kra_anchor) * 10_000.0;
            self.kra_anchor = (self.kra_anchor * 0.93) + (price * 0.07);
            self.kra_x = ((self.kra_x * 0.75) + (rb * 0.25)).clamp(-3.5, 3.5);
            self.kra_d = ((self.kra_d * 0.85) + ((rb * rb) * 0.15)).clamp(0.10, 25.0);
            let dv = (1.20 * (top_depth.max(1.0).ln() * 0.30).clamp(0.30, 3.0)).max(0.20);
            let ik = 0.65 * (-dv / self.kra_d).exp();
            self.kra_k = (self.kra_k * 0.72) + (ik * 0.28);
            out[68] = EngineSignal::sanitized(69, Timeframe::Sec15, tanh_s(self.kra_x, 1.2) * (self.kra_k * 2.2).clamp(0.1, 1.0), 0.70);
        }

        // E70: Landau-Ginzburg Spontaneous Symmetry Breaking
        if has_prev {
            let iphi = ((ret_bps * 0.8) + (imb_l1 * 1.2)).tanh();
            self.lg_phi = (self.lg_phi * 0.72) + (iphi * 0.28);
            let ir = 0.65 - (1.80 * self.lg_phi.abs());
            self.lg_r = ((self.lg_r * 0.82) + (ir * 0.18)).clamp(-1.50, 1.50);
            let boost = (1.0 - self.lg_r).clamp(0.2, 2.0);
            out[69] = EngineSignal::sanitized(70, Timeframe::Sec1, (self.lg_phi * boost).clamp(-1.0, 1.0), 0.80);
        }

        // E71: Renormalization Group (RG) Flow & Scale-Invariant Criticality
        if has_prev {
            self.rg_g = ((self.rg_g * 0.82) + (ret_bps.abs() * 0.18)).clamp(0.05, 5.0);
            let b = (-1.20 * self.rg_g * self.rg_g) + (0.45 * self.rg_g.powi(3));
            self.rg_beta = (self.rg_beta * 0.78) + (b * 0.22);
            let xi = (1.0 / (self.rg_g - 0.65).abs().max(0.02)).clamp(0.5, 50.0);
            out[70] = EngineSignal::sanitized(71, Timeframe::Sec15, tanh_s(ret_bps, 2.0) * (xi / 15.0).clamp(0.2, 1.0), 0.65);
        }

        // E72: Topological Insulator, Berry Curvature & Quantum Hall Edge-State
        if has_prev {
            let [px, py, pz] = self.topo_d;
            let dx = (px * 0.75) + (ret_bps * 0.25);
            let dy = (py * 0.75) + ((imb_l1 * 2.0) * 0.25);
            let igap = 0.85 - (ret_bps.abs() * 0.55) - (imb_l1.abs() * 0.65);
            let dz = ((pz * 0.82) + (igap * 0.18)).clamp(-2.5, 2.5);
            let norm = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-4);
            let cx = py * dz - pz * dy;
            let cy = pz * dx - px * dz;
            let cz = px * dy - py * dx;
            let berry = (dx * cx + dy * cy + dz * cz) / (norm * norm * norm);
            let chern = if dz < 0.0 { dx.signum() } else { 0.0 };
            let hall = (chern * 0.70) + (berry.clamp(-1.5, 1.5) * 0.30);
            self.topo_d = [dx, dy, dz];
            out[71] = EngineSignal::sanitized(72, Timeframe::Sec1, hall.clamp(-1.0, 1.0), 0.75);
        }

        // E73: Fermi-Dirac Pauli Exclusion & Bose-Einstein Liquidity Condensate
        if has_prev {
            let it = (ret_bps.abs() * 0.6 + (1.0 - imb_l1.abs()) * 0.8).clamp(0.10, 4.0);
            self.bf_t = (self.bf_t * 0.82) + (it * 0.18);
            let crowd = (ret_bps.abs() * 0.45) + (imb_l1.abs() * 0.85);
            let imu = (-1.20 / (1.0 + crowd * 2.5)).clamp(-1.50, -0.02);
            self.bf_mu = (self.bf_mu * 0.78) + (imu * 0.22);
            let xa = (-self.bf_mu / self.bf_t).clamp(0.02, 10.0);
            let n0 = 1.0 / (xa.exp() - 1.0).max(0.02);
            let nf = 1.0 / (xa.exp() + 1.0);
            let bec = (n0 / (n0 + nf + 2.0)).clamp(0.0, 0.99);
            self.bf_bec = (self.bf_bec * 0.72) + (bec * 0.28);
            out[72] = EngineSignal::sanitized(73, Timeframe::Sec1, tanh_s(ret_bps + imb_l1, 1.2) * self.bf_bec, 0.75);
        }

        // E74: Casimir Quantum Vacuum Fluctuation & Zero-Point Spread Force
        if has_prev {
            let raw_sp = spread_bps.max(0.25);
            self.cas_d = ((self.cas_d * 0.78) + (raw_sp * 0.22)).clamp(0.35, 10.0);
            let pm = ((inp.bid_sz * inp.ask_sz).max(1.0).ln() * 0.45).clamp(0.30, 4.0);
            let icas = ((1.20 * pm) / self.cas_d.powi(4).max(0.05)).clamp(0.01, 25.0);
            self.cas_f = (self.cas_f * 0.72) + (icas * 0.28);
            self.cas_drv = (self.cas_drv * 0.75) + (((imb_l1 * 0.6) + (ret_bps * 0.4)) * 0.25);
            out[73] = EngineSignal::sanitized(74, Timeframe::Sec1, tanh_s(self.cas_drv * (self.cas_f / 2.0).clamp(0.25, 1.8), 0.75), 0.70);
        }

        // E75: Feynman Path Integral & Least-Action Quantum Trajectory
        if has_prev {
            let vp = [-1.5, 0.0, 1.5];
            let mut pw = [0.0; 3];
            let mut pk = 0.0;
            for i in 0..3 {
                let kin = 0.5 * (ret_bps - vp[i]).powi(2);
                let pot = (1.0 - (vp[i].signum() * imb_l1)).max(0.1) * 0.8;
                let ia = (kin + pot).clamp(0.05, 20.0);
                self.fey_s[i] = (self.fey_s[i] * 0.75) + (ia * 0.25);
                pw[i] = (-self.fey_s[i] / 0.85).exp();
                pk += pw[i];
            }
            let p0 = pw[0] / pk.max(1e-6);
            let p1 = pw[1] / pk.max(1e-6);
            let p2 = pw[2] / pk.max(1e-6);
            let opt_dir = p2 - p0;
            let max_p = p0.max(p1).max(p2);
            self.fey_coh = (self.fey_coh * 0.72) + (max_p * 0.28);
            out[74] = EngineSignal::sanitized(75, Timeframe::Sec1, (opt_dir * (0.5 + self.fey_coh)).clamp(-1.0, 1.0), 0.75);
        }

        self.last_price = price;
    }
}
