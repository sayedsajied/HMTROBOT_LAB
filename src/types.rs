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
