#![allow(dead_code)]

/// Fixed L2 orderbook depth per side (stack-allocated, zero heap allocation)
pub const BOOK_DEPTH: usize = 10;

/// Compact numeric symbol handle for zero-allocation hot paths (e.g., 0 = BTCUSD, 1 = PAXGUSD)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SymbolId(pub u16);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    #[inline]
    pub fn sign(self) -> f64 {
        match self {
            Self::Buy => 1.0,
            Self::Sell => -1.0,
        }
    }

    #[inline]
    pub fn opposite(self) -> Self {
        match self {
            Self::Buy => Self::Sell,
            Self::Sell => Self::Buy,
        }
    }
}

/// Multi-timeframe buckets aggregated live from WebSocket ticks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Timeframe {
    Sec1,
    Sec15,
    Min1,
    Min5,
}

/// Market regimes classified by Layer-1 HMM AI
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    BullTrend,
    BearTrend,
    SidewaysChop,
    HighVolCascade,
}

/// Single price level in an L2 orderbook
#[derive(Debug, Clone, Copy, Default)]
pub struct PriceLevel {
    pub price: f64,
    pub qty: f64,
}

impl PriceLevel {
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.price.is_finite() && self.price > 0.0 && self.qty.is_finite() && self.qty >= 0.0
    }
}

/// Zero-allocation L2 Orderbook snapshot (Jump Trading microstructure core)
#[derive(Debug, Clone, Copy)]
pub struct L2OrderBook {
    pub symbol_id: SymbolId,
    pub bids: [PriceLevel; BOOK_DEPTH],
    pub asks: [PriceLevel; BOOK_DEPTH],
    pub bid_count: usize,
    pub ask_count: usize,
    pub exchange_ts_ms: u64,
    pub local_ts_ms: u64,
    pub seq_id: u64,
}

impl Default for L2OrderBook {
    fn default() -> Self {
        Self {
            symbol_id: SymbolId(0),
            bids: [PriceLevel::default(); BOOK_DEPTH],
            asks: [PriceLevel::default(); BOOK_DEPTH],
            bid_count: 0,
            ask_count: 0,
            exchange_ts_ms: 0,
            local_ts_ms: 0,
            seq_id: 0,
        }
    }
}

impl L2OrderBook {
    #[inline]
    pub fn best_bid(&self) -> Option<PriceLevel> {
        if self.bid_count == 0 {
            return None;
        }
        let lvl = self.bids[0];
        if lvl.is_valid() { Some(lvl) } else { None }
    }

    #[inline]
    pub fn best_ask(&self) -> Option<PriceLevel> {
        if self.ask_count == 0 {
            return None;
        }
        let lvl = self.asks[0];
        if lvl.is_valid() { Some(lvl) } else { None }
    }

    /// Ensures book is non-empty, finite, and uncrossed (best_bid < best_ask)
    #[inline]
    pub fn is_valid(&self) -> bool {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => b.price < a.price,
            _ => false,
        }
    }

    #[inline]
    pub fn mid_price(&self) -> Option<f64> {
        let b = self.best_bid()?;
        let a = self.best_ask()?;
        if b.price >= a.price {
            return None;
        }
        Some((b.price + a.price) * 0.5)
    }

    /// Top-of-book spread in basis points (bps)
    #[inline]
    pub fn spread_bps(&self) -> Option<f64> {
        let b = self.best_bid()?;
        let a = self.best_ask()?;
        let mid = self.mid_price()?;
        if mid <= 0.0 {
            return None;
        }
        Some(((a.price - b.price) / mid) * 10_000.0)
    }

    /// Volume-weighted L1 microprice (institutional short-horizon fair value)
    #[inline]
    pub fn microprice(&self) -> Option<f64> {
        let b = self.best_bid()?;
        let a = self.best_ask()?;
        if b.price >= a.price {
            return None;
        }
        let total_qty = b.qty + a.qty;
        if !total_qty.is_finite() || total_qty <= 0.0 {
            return Some((b.price + a.price) * 0.5);
        }
        Some(((b.price * a.qty) + (a.price * b.qty)) / total_qty)
    }

    /// Multi-level orderbook imbalance in [-1.0, 1.0] across top `depth` levels
    #[inline]
    pub fn imbalance(&self, depth: usize) -> f64 {
        let max_b = depth.min(self.bid_count).min(BOOK_DEPTH);
        let max_a = depth.min(self.ask_count).min(BOOK_DEPTH);

        let mut bid_vol = 0.0;
        for i in 0..max_b {
            if self.bids[i].is_valid() {
                bid_vol += self.bids[i].qty;
            }
        }

        let mut ask_vol = 0.0;
        for i in 0..max_a {
            if self.asks[i].is_valid() {
                ask_vol += self.asks[i].qty;
            }
        }

        let total = bid_vol + ask_vol;
        if !total.is_finite() || total <= 0.0 {
            return 0.0;
        }
        ((bid_vol - ask_vol) / total).clamp(-1.0, 1.0)
    }

    /// Network + exchange processing latency in milliseconds
    #[inline]
    pub fn feed_latency_ms(&self) -> u64 {
        self.local_ts_ms.saturating_sub(self.exchange_ts_ms)
    }

    /// True if no fresh update has arrived within `max_stale_ms`
    #[inline]
    pub fn is_stale(&self, now_ms: u64, max_stale_ms: u64) -> bool {
        now_ms.saturating_sub(self.local_ts_ms) > max_stale_ms
    }
}
/// Live trade tape event (Jump Trading aggressor flow stream)
#[derive(Debug, Clone, Copy)]
pub struct TradeTick {
    pub symbol_id: SymbolId,
    pub price: f64,
    pub qty: f64,
    pub aggressor: Side,
    pub exchange_ts_ms: u64,
    pub local_ts_ms: u64,
    pub trade_id: u64,
}

impl TradeTick {
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.price.is_finite() && self.price > 0.0 && self.qty.is_finite() && self.qty > 0.0
    }

    #[inline]
    pub fn notional_usd(&self) -> f64 {
        if self.is_valid() {
            self.price * self.qty
        } else {
            0.0
        }
    }
}

/// Perpetual vs Spot index basis and funding snapshot
#[derive(Debug, Clone, Copy, Default)]
pub struct FundingBasisSnapshot {
    pub symbol_id: SymbolId,
    pub mark_price: f64,
    pub index_price: f64,
    pub funding_rate: f64,
    pub next_funding_ts_ms: u64,
    pub local_ts_ms: u64,
}

impl FundingBasisSnapshot {
    /// Basis in basis points: positive means Perp trades at a premium over Spot
    #[inline]
    pub fn basis_bps(&self) -> Option<f64> {
        if !self.mark_price.is_finite()
            || !self.index_price.is_finite()
            || self.mark_price <= 0.0
            || self.index_price <= 0.0
        {
            return None;
        }
        Some(((self.mark_price - self.index_price) / self.index_price) * 10_000.0)
    }
}

/// Multi-timeframe aggregated micro-bar for the 86 alpha engines
#[derive(Debug, Clone, Copy)]
pub struct OhlcvBar {
    pub symbol_id: SymbolId,
    pub timeframe: Timeframe,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub trade_count: u32,
    pub start_ts_ms: u64,
    pub end_ts_ms: u64,
}

impl OhlcvBar {
    /// High-to-low volatility range in bps (used by Vol Spike Veto)
    #[inline]
    pub fn range_bps(&self) -> f64 {
        if !self.open.is_finite()
            || !self.high.is_finite()
            || !self.low.is_finite()
            || self.open <= 0.0
            || self.high < self.low
        {
            return 0.0;
        }
        ((self.high - self.low) / self.open) * 10_000.0
    }

    /// Signed volume delta ratio in [-1.0, 1.0]
    #[inline]
    pub fn volume_delta_ratio(&self) -> f64 {
        let total = self.buy_volume + self.sell_volume;
        if !total.is_finite() || total <= 0.0 {
            return 0.0;
        }
        ((self.buy_volume - self.sell_volume) / total).clamp(-1.0, 1.0)
    }
}

/// Standardized microsecond output from any of the 86 alpha engines
#[derive(Debug, Clone, Copy)]
pub struct EngineSignal {
    pub engine_id: u16,
    pub timeframe: Timeframe,
    pub score: f64,      // [-1.0, 1.0]
    pub confidence: f64, // [0.0, 1.0]
}

impl EngineSignal {
    /// Self-healing sanitizer: neutralizes NaN/Inf outputs from faulty engines
    #[inline]
    pub fn sanitized(engine_id: u16, timeframe: Timeframe, score: f64, confidence: f64) -> Self {
        if !score.is_finite() || !confidence.is_finite() {
            return Self {
                engine_id,
                timeframe,
                score: 0.0,
                confidence: 0.0,
            };
        }
        Self {
            engine_id,
            timeframe,
            score: score.clamp(-1.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    #[inline]
    pub fn weighted_score(&self) -> f64 {
        if !self.score.is_finite() || !self.confidence.is_finite() {
            return 0.0;
        }
        self.score.clamp(-1.0, 1.0) * self.confidence.clamp(0.0, 1.0)
    }
}

/// Adverse selection / toxic flow detector state (Jump Trading Pillar #3)
#[derive(Debug, Clone, Copy, Default)]
pub struct ToxicFlowMetrics {
    pub buy_aggressor_ratio: f64, // [0.0, 1.0]
    pub trade_intensity_per_sec: f64,
    pub book_depletion_ratio: f64, // [0.0, 1.0]
    pub toxicity_score: f64,       // [0.0, 1.0]
}

impl ToxicFlowMetrics {
    #[inline]
    pub fn is_toxic(&self, threshold: f64) -> bool {
        self.toxicity_score.is_finite() && self.toxicity_score >= threshold
    }
}

/// Institutional order execution types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    PostOnlyLimit,  // Maker-first priority
    IocLimit,       // Immediate-or-Cancel when edge >= 3x taker cost
    Market,         // Emergency exit only
    ReduceOnlyStop, // Exchange-side hard stop-loss
}

/// Deterministic Order Lifecycle (Wintermute / Optiver anti-duplicate state machine)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderStatus {
    Idle,
    Submitting,
    Open,
    PartiallyFilled,
    Cancelling,
    Filled,
    Cancelled,
    Rejected,
    Failed,
}

impl OrderStatus {
    #[inline]
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Filled | Self::Cancelled | Self::Rejected | Self::Failed
        )
    }

    #[inline]
    pub fn is_active(self) -> bool {
        matches!(
            self,
            Self::Submitting | Self::Open | Self::PartiallyFilled | Self::Cancelling
        )
    }

    /// Strict deterministic state transition validator
    #[inline]
    pub fn can_transition_to(self, next: Self) -> bool {
        match (self, next) {
            (Self::Idle, Self::Submitting) => true,
            (
                Self::Submitting,
                Self::Open
                | Self::PartiallyFilled
                | Self::Filled
                | Self::Rejected
                | Self::Failed,
            ) => true,
            (
                Self::Open,
                Self::PartiallyFilled | Self::Filled | Self::Cancelling | Self::Cancelled,
            ) => true,
            (
                Self::PartiallyFilled,
                Self::PartiallyFilled | Self::Filled | Self::Cancelling | Self::Cancelled,
            ) => true,
            (Self::Cancelling, Self::Cancelled | Self::Filled | Self::Failed) => true,
            _ => false,
        }
    }
}

/// Zero-allocation deterministic order record
#[derive(Debug, Clone, Copy)]
pub struct OrderRecord {
    pub client_order_id: u64,
    pub exchange_order_id: u64,
    pub symbol_id: SymbolId,
    pub side: Side,
    pub order_type: OrderType,
    pub price: f64,
    pub qty: f64,
    pub filled_qty: f64,
    pub status: OrderStatus,
    pub created_ts_ms: u64,
    pub updated_ts_ms: u64,
}

impl OrderRecord {
    pub fn new(
        client_order_id: u64,
        symbol_id: SymbolId,
        side: Side,
        order_type: OrderType,
        price: f64,
        qty: f64,
        now_ms: u64,
    ) -> Self {
        Self {
            client_order_id,
            exchange_order_id: 0,
            symbol_id,
            side,
            order_type,
            price,
            qty,
            filled_qty: 0.0,
            status: OrderStatus::Idle,
            created_ts_ms: now_ms,
            updated_ts_ms: now_ms,
        }
    }

    #[inline]
    pub fn remaining_qty(&self) -> f64 {
        (self.qty - self.filled_qty).max(0.0)
    }

    pub fn transition_to(&mut self, next: OrderStatus, now_ms: u64) -> Result<(), &'static str> {
        if !self.status.can_transition_to(next) {
            return Err("Invalid deterministic order state transition");
        }
        self.status = next;
        self.updated_ts_ms = now_ms;
        Ok(())
    }
}

/// Live inventory & PnL tracker (Jump inventory-skew & exchange reconciliation core)
#[derive(Debug, Clone, Copy, Default)]
pub struct PositionState {
    pub symbol_id: SymbolId,
    pub net_qty: f64, // +ve = Long, -ve = Short, 0.0 = Flat
    pub avg_entry_price: f64,
    pub unrealized_pnl_usd: f64,
    pub realized_pnl_usd: f64,
    pub hard_sl_price: f64,
    pub last_update_ms: u64,
}

impl PositionState {
    #[inline]
    pub fn is_flat(&self) -> bool {
        self.net_qty.abs() < 1e-9
    }

    #[inline]
    pub fn notional_usd(&self, mark_price: f64) -> f64 {
        if !mark_price.is_finite() || mark_price <= 0.0 {
            return 0.0;
        }
        self.net_qty.abs() * mark_price
    }

    #[inline]
    pub fn update_unrealized_pnl(&mut self, mark_price: f64, now_ms: u64) {
        if self.is_flat()
            || !mark_price.is_finite()
            || !self.avg_entry_price.is_finite()
            || mark_price <= 0.0
        {
            self.unrealized_pnl_usd = 0.0;
        } else {
            self.unrealized_pnl_usd = (mark_price - self.avg_entry_price) * self.net_qty;
        }
        self.last_update_ms = now_ms;
    }
}
