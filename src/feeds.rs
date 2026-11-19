#![allow(dead_code)]

use crate::types::{
    FundingBasisSnapshot, L2OrderBook, OhlcvBar, Side, SymbolId, Timeframe, ToxicFlowMetrics,
    TradeTick,
};

pub const TIMEFRAMES: [Timeframe; 4] = [
    Timeframe::Sec1,
    Timeframe::Sec15,
    Timeframe::Min1,
    Timeframe::Min5,
];

const TOXIC_RING_CAPACITY: usize = 64;

#[inline]
pub fn timeframe_duration_ms(tf: Timeframe) -> u64 {
    match tf {
        Timeframe::Sec1 => 1_000,
        Timeframe::Sec15 => 15_000,
        Timeframe::Min1 => 60_000,
        Timeframe::Min5 => 300_000,
    }
}

/// Zero-allocation market event dispatched from WebSocket to Strategy/AI loop
#[derive(Debug, Clone, Copy)]
pub enum FeedEvent {
    OrderBook(L2OrderBook),
    Trade(TradeTick),
    FundingBasis(FundingBasisSnapshot),
    BarClosed(OhlcvBar),
    ToxicUpdate(SymbolId, ToxicFlowMetrics),
    Connected,
    Disconnected,
}

/// Tracks 24/7 WebSocket liveness, stale-feed timeouts, and parse health
#[derive(Debug, Clone, Copy, Default)]
pub struct FeedHealthMonitor {
    pub is_connected: bool,
    pub last_msg_ts_ms: u64,
    pub last_book_ts_ms: u64,
    pub reconnect_count: u64,
    pub parse_error_count: u64,
    pub dropped_event_count: u64,
}

impl FeedHealthMonitor {
    #[inline]
    pub fn on_connected(&mut self, now_ms: u64) {
        self.is_connected = true;
        self.last_msg_ts_ms = now_ms;
    }

    #[inline]
    pub fn on_disconnected(&mut self) {
        self.is_connected = false;
        self.reconnect_count = self.reconnect_count.saturating_add(1);
    }

    #[inline]
    pub fn on_message(&mut self, now_ms: u64) {
        self.last_msg_ts_ms = now_ms;
    }

    #[inline]
    pub fn on_book(&mut self, now_ms: u64) {
        self.last_msg_ts_ms = now_ms;
        self.last_book_ts_ms = now_ms;
    }

    #[inline]
    pub fn on_parse_error(&mut self) {
        self.parse_error_count = self.parse_error_count.saturating_add(1);
    }

    #[inline]
    pub fn is_stale(&self, now_ms: u64, max_stale_ms: u64) -> bool {
        if !self.is_connected || self.last_msg_ts_ms == 0 {
            return true;
        }
        now_ms.saturating_sub(self.last_msg_ts_ms) > max_stale_ms
    }
}

/// Exponential backoff with deterministic jitter (prevents exchange IP bans during outages)
#[derive(Debug, Clone, Copy)]
pub struct ReconnectBackoff {
    base_ms: u64,
    max_ms: u64,
    attempts: u32,
}

impl ReconnectBackoff {
    pub const fn new(base_ms: u64, max_ms: u64) -> Self {
        Self {
            base_ms,
            max_ms,
            attempts: 0,
        }
    }

    pub fn next_delay_ms(&mut self, seed_ts_ms: u64) -> u64 {
        let shift = self.attempts.min(10);
        let raw = self.base_ms.saturating_mul(1u64 << shift).min(self.max_ms);
        self.attempts = self.attempts.saturating_add(1);
        let jitter = seed_ts_ms % (self.base_ms.max(1) / 2).max(1);
        raw.saturating_add(jitter).min(self.max_ms)
    }

    #[inline]
    pub fn reset(&mut self) {
        self.attempts = 0;
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct TradeSample {
    ts_ms: u64,
    qty: f64,
    is_buy: bool,
}

/// Jump Trading Pillar #3: Stack-allocated rolling adverse-selection / toxic-flow detector
#[derive(Debug, Clone, Copy)]
pub struct ToxicFlowTracker {
    ring: [TradeSample; TOXIC_RING_CAPACITY],
    cursor: usize,
    count: usize,
    window_ms: u64,
    prev_top_depth: f64,
    last_depletion_ratio: f64,
}

impl ToxicFlowTracker {
    pub const fn new(window_ms: u64) -> Self {
        Self {
            ring: [TradeSample {
                ts_ms: 0,
                qty: 0.0,
                is_buy: false,
            }; TOXIC_RING_CAPACITY],
            cursor: 0,
            count: 0,
            window_ms: if window_ms == 0 { 5_000 } else { window_ms },
            prev_top_depth: 0.0,
            last_depletion_ratio: 0.0,
        }
    }

    /// Update book-depletion component when top-3 L2 liquidity suddenly vanishes
    pub fn on_book(&mut self, book: &L2OrderBook) {
        if !book.is_valid() {
            return;
        }
        let depth_levels = 3usize;
        let mut curr_depth = 0.0;
        for i in 0..depth_levels.min(book.bid_count) {
            if book.bids[i].is_valid() {
                curr_depth += book.bids[i].qty;
            }
        }
        for i in 0..depth_levels.min(book.ask_count) {
            if book.asks[i].is_valid() {
                curr_depth += book.asks[i].qty;
            }
        }

        if self.prev_top_depth > 0.0 && curr_depth < self.prev_top_depth {
            let drop_ratio = (self.prev_top_depth - curr_depth) / self.prev_top_depth;
            self.last_depletion_ratio = drop_ratio.clamp(0.0, 1.0);
        } else {
            self.last_depletion_ratio *= 0.7;
        }
        if curr_depth.is_finite() && curr_depth > 0.0 {
            self.prev_top_depth = curr_depth;
        }
    }

    /// Record live trade tape tick and compute updated toxicity metrics
    pub fn on_trade(&mut self, tick: &TradeTick) -> ToxicFlowMetrics {
        if tick.is_valid() {
            self.ring[self.cursor] = TradeSample {
                ts_ms: tick.local_ts_ms,
                qty: tick.qty,
                is_buy: tick.aggressor == Side::Buy,
            };
            self.cursor = (self.cursor + 1) % TOXIC_RING_CAPACITY;
            if self.count < TOXIC_RING_CAPACITY {
                self.count += 1;
            }
        }
        self.compute_metrics(tick.local_ts_ms)
    }

    pub fn compute_metrics(&self, now_ms: u64) -> ToxicFlowMetrics {
        let cutoff = now_ms.saturating_sub(self.window_ms);
        let mut buy_vol = 0.0;
        let mut sell_vol = 0.0;
        let mut active_trades = 0u32;

        for i in 0..self.count {
            let s = self.ring[i];
            if s.ts_ms >= cutoff && s.qty.is_finite() && s.qty > 0.0 {
                active_trades = active_trades.saturating_add(1);
                if s.is_buy {
                    buy_vol += s.qty;
                } else {
                    sell_vol += s.qty;
                }
            }
        }

        let total_vol = buy_vol + sell_vol;
        let buy_ratio = if total_vol > 0.0 {
            (buy_vol / total_vol).clamp(0.0, 1.0)
        } else {
            0.5
        };

        let window_sec = (self.window_ms as f64 / 1000.0).max(0.1);
        let intensity = (active_trades as f64) / window_sec;

        // One-sided flow imbalance [0, 1] + normalized burst intensity + book depletion
        let imbalance = ((buy_ratio - 0.5).abs() * 2.0).clamp(0.0, 1.0);
        let norm_intensity = (intensity / 20.0).clamp(0.0, 1.0);
        let depletion = self.last_depletion_ratio.clamp(0.0, 1.0);

        let toxicity = (0.50 * imbalance) + (0.25 * norm_intensity) + (0.25 * depletion);

        ToxicFlowMetrics {
            buy_aggressor_ratio: buy_ratio,
            trade_intensity_per_sec: intensity,
            book_depletion_ratio: depletion,
            toxicity_score: toxicity.clamp(0.0, 1.0),
        }
    }
}

/// Zero-allocation Multi-Timeframe (1s, 15s, 1m, 5m) live OHLCV bar builder
#[derive(Debug, Clone, Copy)]
pub struct MtfBarAggregator {
    symbol_id: SymbolId,
    active_bars: [Option<OhlcvBar>; 4],
}

impl MtfBarAggregator {
    pub const fn new(symbol_id: SymbolId) -> Self {
        Self {
            symbol_id,
            active_bars: [None; 4],
        }
    }

    /// Ingests one TradeTick and returns any newly completed bars across [1s, 15s, 1m, 5m]
    pub fn on_trade(&mut self, tick: &TradeTick) -> [Option<OhlcvBar>; 4] {
        let mut closed = [None; 4];
        if !tick.is_valid() || tick.symbol_id != self.symbol_id {
            return closed;
        }

        let ts = if tick.exchange_ts_ms > 0 {
            tick.exchange_ts_ms
        } else {
            tick.local_ts_ms
        };

        for idx in 0..4 {
            let tf = TIMEFRAMES[idx];
            let dur = timeframe_duration_ms(tf);
            let bucket_start = (ts / dur) * dur;
            let bucket_end = bucket_start.saturating_add(dur);

            match self.active_bars[idx] {
                Some(mut bar) if bar.start_ts_ms == bucket_start => {
                    if tick.price > bar.high {
                        bar.high = tick.price;
                    }
                    if tick.price < bar.low {
                        bar.low = tick.price;
                    }
                    bar.close = tick.price;
                    match tick.aggressor {
                        Side::Buy => bar.buy_volume += tick.qty,
                        Side::Sell => bar.sell_volume += tick.qty,
                    }
                    bar.trade_count = bar.trade_count.saturating_add(1);
                    self.active_bars[idx] = Some(bar);
                }
                Some(prev_bar) if bucket_start > prev_bar.start_ts_ms => {
                    closed[idx] = Some(prev_bar);
                    self.active_bars[idx] = Some(Self::new_bar(
                        self.symbol_id,
                        tf,
                        tick,
                        bucket_start,
                        bucket_end,
                    ));
                }
                Some(_) => {
                    // Out-of-order stale trade older than current bucket: safely ignored
                }
                None => {
                    self.active_bars[idx] = Some(Self::new_bar(
                        self.symbol_id,
                        tf,
                        tick,
                        bucket_start,
                        bucket_end,
                    ));
                }
            }
        }

        closed
    }

    #[inline]
    fn new_bar(
        symbol_id: SymbolId,
        timeframe: Timeframe,
        tick: &TradeTick,
        start_ts_ms: u64,
        end_ts_ms: u64,
    ) -> OhlcvBar {
        let (buy_volume, sell_volume) = match tick.aggressor {
            Side::Buy => (tick.qty, 0.0),
            Side::Sell => (0.0, tick.qty),
        };
        OhlcvBar {
            symbol_id,
            timeframe,
            open: tick.price,
            high: tick.price,
            low: tick.price,
            close: tick.price,
            buy_volume,
            sell_volume,
            trade_count: 1,
            start_ts_ms,
            end_ts_ms,
        }
    }
}
use crate::config::ExchangeId;
use crate::types::{PriceLevel, BOOK_DEPTH};
use serde_json::{json, Value};

#[inline]
fn normalize_ts_ms(raw: u64) -> u64 {
    if raw > 1_000_000_000_000_000 {
        raw / 1_000 // microseconds -> milliseconds
    } else if raw > 1_000_000_000_000 {
        raw // already milliseconds
    } else if raw > 1_000_000_000 {
        raw.saturating_mul(1_000) // seconds -> milliseconds
    } else {
        raw
    }
}

#[inline]
fn val_to_f64(v: &Value) -> Option<f64> {
    let parsed = v
        .as_f64()
        .or_else(|| v.as_str().and_then(|s| s.trim().parse::<f64>().ok()))?;
    if parsed.is_finite() {
        Some(parsed)
    } else {
        None
    }
}

#[inline]
fn val_to_u64(v: &Value) -> Option<u64> {
    v.as_u64()
        .or_else(|| v.as_str().and_then(|s| s.trim().parse::<u64>().ok()))
}

#[inline]
fn resolve_symbol_id(symbols: &[String], raw_sym: &str) -> Option<SymbolId> {
    symbols
        .iter()
        .position(|s| s.eq_ignore_ascii_case(raw_sym))
        .map(|idx| SymbolId(idx as u16))
}

/// Builds the real exchange WebSocket subscription JSON for L2 Depth, Trades, and Ticker/Funding
pub fn build_subscribe_payload(exchange: ExchangeId, symbols: &[String]) -> String {
    match exchange {
        ExchangeId::DeltaExchange => json!({
            "type": "subscribe",
            "payload": {
                "channels": [
                    { "name": "l2_orderbook", "symbols": symbols },
                    { "name": "all_trades", "symbols": symbols },
                    { "name": "v2/ticker", "symbols": symbols }
                ]
            }
        })
        .to_string(),
        ExchangeId::Bybit => {
            let mut args = Vec::with_capacity(symbols.len() * 3);
            for s in symbols {
                args.push(format!("orderbook.50.{}", s));
                args.push(format!("publicTrade.{}", s));
                args.push(format!("tickers.{}", s));
            }
            json!({
                "op": "subscribe",
                "args": args
            })
            .to_string()
        }
        ExchangeId::Binance => {
            let mut params = Vec::with_capacity(symbols.len() * 3);
            for s in symbols {
                let lower = s.to_lowercase();
                params.push(format!("{}@depth10@100ms", lower));
                params.push(format!("{}@aggTrade", lower));
                params.push(format!("{}@markPrice@1s", lower));
            }
            json!({
                "method": "SUBSCRIBE",
                "params": params,
                "id": 1
            })
            .to_string()
        }
    }
}

/// Zero-panic unified parser for Delta Exchange, Bybit, and Binance live JSON streams
pub fn parse_ws_message(
    exchange: ExchangeId,
    raw_json: &str,
    symbols: &[String],
    local_ts_ms: u64,
) -> Option<FeedEvent> {
    let v: Value = serde_json::from_str(raw_json).ok()?;
    match exchange {
        ExchangeId::DeltaExchange => parse_delta(&v, symbols, local_ts_ms),
        ExchangeId::Bybit => parse_bybit(&v, symbols, local_ts_ms),
        ExchangeId::Binance => parse_binance(&v, symbols, local_ts_ms),
    }
}

fn fill_tuple_levels(arr: &[Value], out: &mut [PriceLevel; BOOK_DEPTH]) -> usize {
    let mut count = 0usize;
    for entry in arr.iter() {
        if count >= BOOK_DEPTH {
            break;
        }
        if let Some(pair) = entry.as_array() {
            if pair.len() >= 2 {
                if let (Some(price), Some(qty)) = (val_to_f64(&pair[0]), val_to_f64(&pair[1])) {
                    if price > 0.0 && qty > 0.0 {
                        out[count] = PriceLevel { price, qty };
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

fn fill_delta_levels(arr: &[Value], out: &mut [PriceLevel; BOOK_DEPTH]) -> usize {
    let mut count = 0usize;
    for entry in arr.iter() {
        if count >= BOOK_DEPTH {
            break;
        }
        let price = entry.get("limit_price").and_then(val_to_f64);
        let qty = entry.get("size").and_then(val_to_f64);
        if let (Some(p), Some(q)) = (price, qty) {
            if p > 0.0 && q > 0.0 {
                out[count] = PriceLevel { price: p, qty: q };
                count += 1;
            }
        }
    }
    count
}

fn parse_delta(v: &Value, symbols: &[String], local_ts_ms: u64) -> Option<FeedEvent> {
    let msg_type = v.get("type")?.as_str()?;
    let sym_str = v.get("symbol")?.as_str()?;
    let symbol_id = resolve_symbol_id(symbols, sym_str)?;
    let ex_ts = v
        .get("timestamp")
        .and_then(val_to_u64)
        .map(normalize_ts_ms)
        .unwrap_or(local_ts_ms);

    match msg_type {
        "l2_orderbook" => {
            let mut book = L2OrderBook {
                symbol_id,
                exchange_ts_ms: ex_ts,
                local_ts_ms,
                seq_id: v.get("sequence_no").and_then(val_to_u64).unwrap_or(0),
                ..Default::default()
            };
            if let Some(buys) = v.get("buy").and_then(|x| x.as_array()) {
                book.bid_count = fill_delta_levels(buys, &mut book.bids);
            }
            if let Some(sells) = v.get("sell").and_then(|x| x.as_array()) {
                book.ask_count = fill_delta_levels(sells, &mut book.asks);
            }
            if book.is_valid() {
                Some(FeedEvent::OrderBook(book))
            } else {
                None
            }
        }
        "all_trades" => {
            let price = v.get("price").and_then(val_to_f64)?;
            let qty = v.get("size").and_then(val_to_f64)?;
            let aggressor = match v.get("buyer_role").and_then(|x| x.as_str()) {
                Some("taker") => Side::Buy,
                _ => Side::Sell,
            };
            let tick = TradeTick {
                symbol_id,
                price,
                qty,
                aggressor,
                exchange_ts_ms: ex_ts,
                local_ts_ms,
                trade_id: v.get("trade_id").and_then(val_to_u64).unwrap_or(0),
            };
            if tick.is_valid() {
                Some(FeedEvent::Trade(tick))
            } else {
                None
            }
        }
        "v2/ticker" => {
            let mark_price = v.get("mark_price").and_then(val_to_f64)?;
            let index_price = v
                .get("spot_price")
                .and_then(val_to_f64)
                .unwrap_or(mark_price);
            let funding_rate = v.get("funding_rate").and_then(val_to_f64).unwrap_or(0.0);
            Some(FeedEvent::FundingBasis(FundingBasisSnapshot {
                symbol_id,
                mark_price,
                index_price,
                funding_rate,
                next_funding_ts_ms: v
                    .get("next_funding_realization")
                    .and_then(val_to_u64)
                    .map(normalize_ts_ms)
                    .unwrap_or(0),
                local_ts_ms,
            }))
        }
        _ => None,
    }
}

fn parse_bybit(v: &Value, symbols: &[String], local_ts_ms: u64) -> Option<FeedEvent> {
    let topic = v.get("topic")?.as_str()?;
    let data = v.get("data")?;
    let ex_ts = v
        .get("ts")
        .and_then(val_to_u64)
        .map(normalize_ts_ms)
        .unwrap_or(local_ts_ms);

    if topic.starts_with("orderbook.") {
        let sym_str = data.get("s")?.as_str()?;
        let symbol_id = resolve_symbol_id(symbols, sym_str)?;
        let mut book = L2OrderBook {
            symbol_id,
            exchange_ts_ms: ex_ts,
            local_ts_ms,
            seq_id: data.get("seq").and_then(val_to_u64).unwrap_or(0),
            ..Default::default()
        };
        if let Some(bids) = data.get("b").and_then(|x| x.as_array()) {
            book.bid_count = fill_tuple_levels(bids, &mut book.bids);
        }
        if let Some(asks) = data.get("a").and_then(|x| x.as_array()) {
            book.ask_count = fill_tuple_levels(asks, &mut book.asks);
        }
        return if book.is_valid() {
            Some(FeedEvent::OrderBook(book))
        } else {
            None
        };
    }

    if topic.starts_with("publicTrade.") {
        let first = data.as_array()?.first()?;
        let sym_str = first.get("s")?.as_str()?;
        let symbol_id = resolve_symbol_id(symbols, sym_str)?;
        let price = first.get("p").and_then(val_to_f64)?;
        let qty = first.get("v").and_then(val_to_f64)?;
        let aggressor = match first.get("S").and_then(|x| x.as_str()) {
            Some("Buy") => Side::Buy,
            _ => Side::Sell,
        };
        let tick = TradeTick {
            symbol_id,
            price,
            qty,
            aggressor,
            exchange_ts_ms: first
                .get("T")
                .and_then(val_to_u64)
                .map(normalize_ts_ms)
                .unwrap_or(ex_ts),
            local_ts_ms,
            trade_id: first.get("i").and_then(val_to_u64).unwrap_or(0),
        };
        return if tick.is_valid() {
            Some(FeedEvent::Trade(tick))
        } else {
            None
        };
    }

    if topic.starts_with("tickers.") {
        let sym_str = data.get("symbol")?.as_str()?;
        let symbol_id = resolve_symbol_id(symbols, sym_str)?;
        let mark_price = data.get("markPrice").and_then(val_to_f64)?;
        let index_price = data
            .get("indexPrice")
            .and_then(val_to_f64)
            .unwrap_or(mark_price);
        let funding_rate = data.get("fundingRate").and_then(val_to_f64).unwrap_or(0.0);
        return Some(FeedEvent::FundingBasis(FundingBasisSnapshot {
            symbol_id,
            mark_price,
            index_price,
            funding_rate,
            next_funding_ts_ms: data
                .get("nextFundingTime")
                .and_then(val_to_u64)
                .map(normalize_ts_ms)
                .unwrap_or(0),
            local_ts_ms,
        }));
    }

    None
}

fn parse_binance(v: &Value, symbols: &[String], local_ts_ms: u64) -> Option<FeedEvent> {
    let event_type = v.get("e")?.as_str()?;
    let sym_str = v.get("s")?.as_str()?;
    let symbol_id = resolve_symbol_id(symbols, sym_str)?;
    let ex_ts = v
        .get("E")
        .and_then(val_to_u64)
        .map(normalize_ts_ms)
        .unwrap_or(local_ts_ms);

    match event_type {
        "depthUpdate" => {
            let mut book = L2OrderBook {
                symbol_id,
                exchange_ts_ms: ex_ts,
                local_ts_ms,
                seq_id: v.get("u").and_then(val_to_u64).unwrap_or(0),
                ..Default::default()
            };
            let bids_val = v.get("b").or_else(|| v.get("bids"));
            let asks_val = v.get("a").or_else(|| v.get("asks"));
            if let Some(bids) = bids_val.and_then(|x| x.as_array()) {
                book.bid_count = fill_tuple_levels(bids, &mut book.bids);
            }
            if let Some(asks) = asks_val.and_then(|x| x.as_array()) {
                book.ask_count = fill_tuple_levels(asks, &mut book.asks);
            }
            if book.is_valid() {
                Some(FeedEvent::OrderBook(book))
            } else {
                None
            }
        }
        "aggTrade" => {
            let price = v.get("p").and_then(val_to_f64)?;
            let qty = v.get("q").and_then(val_to_f64)?;
            // On Binance, m == true means buyer is maker -> aggressor is Sell
            let is_buyer_maker = v.get("m").and_then(|x| x.as_bool()).unwrap_or(false);
            let aggressor = if is_buyer_maker {
                Side::Sell
            } else {
                Side::Buy
            };
            let tick = TradeTick {
                symbol_id,
                price,
                qty,
                aggressor,
                exchange_ts_ms: v
                    .get("T")
                    .and_then(val_to_u64)
                    .map(normalize_ts_ms)
                    .unwrap_or(ex_ts),
                local_ts_ms,
                trade_id: v.get("a").and_then(val_to_u64).unwrap_or(0),
            };
            if tick.is_valid() {
                Some(FeedEvent::Trade(tick))
            } else {
                None
            }
        }
        "markPriceUpdate" => {
            let mark_price = v.get("p").and_then(val_to_f64)?;
            let index_price = v.get("i").and_then(val_to_f64).unwrap_or(mark_price);
            let funding_rate = v.get("r").and_then(val_to_f64).unwrap_or(0.0);
            Some(FeedEvent::FundingBasis(FundingBasisSnapshot {
                symbol_id,
                mark_price,
                index_price,
                funding_rate,
                next_funding_ts_ms: v
                    .get("T")
                    .and_then(val_to_u64)
                    .map(normalize_ts_ms)
                    .unwrap_or(0),
                local_ts_ms,
            }))
        }
        _ => None,
    }
}
