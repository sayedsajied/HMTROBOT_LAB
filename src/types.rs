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
#[cfg(test)]
mod tests {
    use super::*;

    fn assert_is_copy<T: Copy>() {}

    #[test]
    fn test_hot_path_structs_are_zero_allocation_copy() {
        assert_is_copy::<L2OrderBook>();
        assert_is_copy::<TradeTick>();
        assert_is_copy::<FundingBasisSnapshot>();
        assert_is_copy::<OhlcvBar>();
        assert_is_copy::<EngineSignal>();
        assert_is_copy::<ToxicFlowMetrics>();
        assert_is_copy::<OrderRecord>();
        assert_is_copy::<PositionState>();
    }

    #[test]
    fn test_l2_book_microprice_spread_and_imbalance() {
        let mut book = L2OrderBook::default();
        book.symbol_id = SymbolId(0);
        book.bids[0] = PriceLevel { price: 100.0, qty: 3.0 };
        book.bids[1] = PriceLevel { price: 99.5, qty: 5.0 };
        book.bid_count = 2;

        book.asks[0] = PriceLevel { price: 102.0, qty: 1.0 };
        book.asks[1] = PriceLevel { price: 102.5, qty: 1.0 };
        book.ask_count = 2;

        book.exchange_ts_ms = 10_000;
        book.local_ts_ms = 10_045;

        assert!(book.is_valid());
        assert_eq!(book.feed_latency_ms(), 45);
        assert!(!book.is_stale(12_000, 3_000));
        assert!(book.is_stale(13_500, 3_000));

        let mid = book.mid_price().unwrap();
        assert!((mid - 101.0).abs() < 1e-9);

        // Microprice = (100*1 + 102*3) / (3 + 1) = 101.5 (heavy bid pushes fair value up)
        let micro = book.microprice().unwrap();
        assert!((micro - 101.5).abs() < 1e-9);

        // Top-2 Imbalance = (8 - 2) / (8 + 2) = +0.6
        assert!((book.imbalance(2) - 0.6).abs() < 1e-9);

        let spread = book.spread_bps().unwrap();
        assert!((spread - (20_000.0 / 101.0)).abs() < 1e-6);
    }

    #[test]
    fn test_crossed_or_nan_orderbook_is_rejected() {
        let mut crossed = L2OrderBook::default();
        crossed.bids[0] = PriceLevel { price: 105.0, qty: 2.0 };
        crossed.asks[0] = PriceLevel { price: 104.0, qty: 2.0 };
        crossed.bid_count = 1;
        crossed.ask_count = 1;

        assert!(!crossed.is_valid());
        assert!(crossed.mid_price().is_none());
        assert!(crossed.microprice().is_none());

        let mut nan_book = L2OrderBook::default();
        nan_book.bids[0] = PriceLevel { price: f64::NAN, qty: 1.0 };
        nan_book.asks[0] = PriceLevel { price: 100.0, qty: 1.0 };
        nan_book.bid_count = 1;
        nan_book.ask_count = 1;

        assert!(!nan_book.is_valid());
    }

    #[test]
    fn test_trade_tick_and_funding_basis() {
        let tick = TradeTick {
            symbol_id: SymbolId(0),
            price: 90_000.0,
            qty: 0.25,
            aggressor: Side::Buy,
            exchange_ts_ms: 1_000,
            local_ts_ms: 1_020,
            trade_id: 42,
        };
        assert!(tick.is_valid());
        assert!((tick.notional_usd() - 22_500.0).abs() < 1e-9);

        let basis = FundingBasisSnapshot {
            symbol_id: SymbolId(0),
            mark_price: 100_050.0,
            index_price: 100_000.0,
            funding_rate: 0.0001,
            next_funding_ts_ms: 9_999,
            local_ts_ms: 1_020,
        };
        assert!((basis.basis_bps().unwrap() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_ohlcv_bar_volatility_and_delta() {
        let bar = OhlcvBar {
            symbol_id: SymbolId(1),
            timeframe: Timeframe::Sec15,
            open: 100.0,
            high: 101.0,
            low: 99.0,
            close: 100.5,
            buy_volume: 7.0,
            sell_volume: 3.0,
            trade_count: 18,
            start_ts_ms: 0,
            end_ts_ms: 15_000,
        };
        assert!((bar.range_bps() - 200.0).abs() < 1e-9);
        assert!((bar.volume_delta_ratio() - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_engine_signal_self_healing_sanitizer() {
        let bad = EngineSignal::sanitized(7, Timeframe::Sec1, f64::NAN, 0.9);
        assert_eq!(bad.score, 0.0);
        assert_eq!(bad.confidence, 0.0);
        assert_eq!(bad.weighted_score(), 0.0);

        let clamped = EngineSignal::sanitized(8, Timeframe::Min1, 5.0, 2.0);
        assert_eq!(clamped.score, 1.0);
        assert_eq!(clamped.confidence, 1.0);
        assert_eq!(clamped.weighted_score(), 1.0);
    }

    #[test]
    fn test_toxic_flow_threshold_guard() {
        let toxic = ToxicFlowMetrics {
            buy_aggressor_ratio: 0.92,
            trade_intensity_per_sec: 40.0,
            book_depletion_ratio: 0.85,
            toxicity_score: 0.82,
        };
        assert!(toxic.is_toxic(0.75));
        assert!(!toxic.is_toxic(0.90));

        let nan_toxic = ToxicFlowMetrics {
            toxicity_score: f64::NAN,
            ..Default::default()
        };
        assert!(!nan_toxic.is_toxic(0.5));
    }

    #[test]
    fn test_deterministic_order_state_machine() {
        let mut order = OrderRecord::new(
            1001,
            SymbolId(0),
            Side::Buy,
            OrderType::PostOnlyLimit,
            95_000.0,
            0.1,
            1_000,
        );
        assert_eq!(order.status, OrderStatus::Idle);
        assert!((order.remaining_qty() - 0.1).abs() < 1e-9);

        // Illegal jump Idle -> Filled must fail
        assert!(order.transition_to(OrderStatus::Filled, 1_005).is_err());

        // Valid institutional lifecycle
        assert!(order.transition_to(OrderStatus::Submitting, 1_010).is_ok());
        assert!(order.status.is_active());
        assert!(order.transition_to(OrderStatus::Open, 1_020).is_ok());
        assert!(order.transition_to(OrderStatus::PartiallyFilled, 1_050).is_ok());
        order.filled_qty = 0.04;
        assert!((order.remaining_qty() - 0.06).abs() < 1e-9);
        assert!(order.transition_to(OrderStatus::Filled, 1_090).is_ok());
        assert!(order.status.is_terminal());

        // Terminal state cannot reopen
        assert!(order.transition_to(OrderStatus::Open, 1_100).is_err());
    }

    #[test]
    fn test_position_state_long_and_short_pnl() {
        let mut long_pos = PositionState {
            symbol_id: SymbolId(0),
            net_qty: 0.5,
            avg_entry_price: 100_000.0,
            ..Default::default()
        };
        assert!(!long_pos.is_flat());
        long_pos.update_unrealized_pnl(101_000.0, 5_000);
        assert!((long_pos.unrealized_pnl_usd - 500.0).abs() < 1e-9);
        assert!((long_pos.notional_usd(101_000.0) - 50_500.0).abs() < 1e-9);

        let mut short_pos = PositionState {
            symbol_id: SymbolId(0),
            net_qty: -0.5,
            avg_entry_price: 100_000.0,
            ..Default::default()
        };
        short_pos.update_unrealized_pnl(99_000.0, 5_000);
        assert!((short_pos.unrealized_pnl_usd - 500.0).abs() < 1e-9);
    }
}
