#![allow(dead_code)]

use crate::types::SymbolId;

/// Directional side for sniper trade intent
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeDirection {
    Flat,
    Long,
    Short,
}

/// Jump Pillar #5: Maker-First Post-Only priority; Taker only on >=3x fee edge or emergency exit
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStyle {
    None,
    MakerPostOnly,
    TakerAggressive,
    EmergencyClose,
}

/// Deterministic reason when strategy blocks or skips an entry
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockReason {
    None,
    SafeModeActive,
    ToxicFlowShield,
    InsufficientAlpha,
    FailedFeeGate,
    SpreadTooWide,
    InventoryCapReached,
    InvalidMarketData,
}

/// Static fee, edge, and inventory limits (BTC / ETH / SOL)
#[derive(Debug, Clone, Copy)]
pub struct StrategyConfig {
    pub maker_fee_bps: f64,
    pub taker_fee_bps: f64,
    pub est_slippage_bps: f64,
    pub taker_edge_mult: f64,   // Mandatory 3.0x round-trip fee + slippage gate
    pub maker_edge_mult: f64,   // Minimum edge multiple for Post-Only entries
    pub min_alpha_score: f64,   // Minimum |alpha_score| to consider entry
    pub min_confidence: f64,    // Minimum AI confidence
    pub max_spread_bps: f64,    // Hard spread veto
    pub max_inventory_usd: f64, // Per-symbol inventory cap
    pub base_edge_scale_bps: f64,
}

impl StrategyConfig {
    pub const fn default_crypto() -> Self {
        Self {
            maker_fee_bps: 2.0,
            taker_fee_bps: 5.0,
            est_slippage_bps: 1.5,
            taker_edge_mult: 3.0,
            maker_edge_mult: 1.5,
            min_alpha_score: 0.28,
            min_confidence: 0.55,
            max_spread_bps: 8.0,
            max_inventory_usd: 1_000.0,
            base_edge_scale_bps: 28.0,
        }
    }
}

/// Current live position & unrealized state for a single symbol
#[derive(Debug, Clone, Copy)]
pub struct InventoryState {
    pub net_qty: f64,        // Positive = Long, Negative = Short, 0.0 = Flat
    pub avg_entry_price: f64,
    pub notional_usd: f64,   // Signed USD exposure
    pub unrealized_bps: f64,
    pub has_open_order: bool,
}

impl InventoryState {
    pub const fn flat() -> Self {
        Self {
            net_qty: 0.0,
            avg_entry_price: 0.0,
            notional_usd: 0.0,
            unrealized_bps: 0.0,
            has_open_order: false,
        }
    }

    #[inline]
    pub fn utilization(&self, max_inventory_usd: f64) -> f64 {
        if !self.notional_usd.is_finite() || max_inventory_usd <= 1e-6 {
            return 0.0;
        }
        (self.notional_usd / max_inventory_usd).clamp(-1.0, 1.0)
    }
}

/// Zero-allocation output passed from SniperStrategy to Risk & Execution
#[derive(Debug, Clone, Copy)]
pub struct StrategyIntent {
    pub symbol_id: SymbolId,
    pub direction: TradeDirection,
    pub style: ExecutionStyle,
    pub block_reason: BlockReason,
    pub limit_price: f64,
    pub stop_loss_price: f64,
    pub take_profit_price: f64,
    pub size_fraction: f64,     // [0.0, 1.0] fraction of max_inventory_usd
    pub expected_edge_bps: f64,
    pub fee_hurdle_bps: f64,
    pub toxic_score: f64,       // [0.0, 1.0]
    pub cancel_open_orders: bool,
}

impl StrategyIntent {
    pub const fn blocked(symbol_id: SymbolId, reason: BlockReason, cancel_open_orders: bool, toxic_score: f64) -> Self {
        Self {
            symbol_id,
            direction: TradeDirection::Flat,
            style: ExecutionStyle::None,
            block_reason: reason,
            limit_price: 0.0,
            stop_loss_price: 0.0,
            take_profit_price: 0.0,
            size_fraction: 0.0,
            expected_edge_bps: 0.0,
            fee_hurdle_bps: 0.0,
            toxic_score,
            cancel_open_orders,
        }
    }
}

/// Jump Pillar #5: Fee-First, Maker-First Gatekeeper
pub struct FeeGate;

impl FeeGate {
    #[inline]
    pub fn maker_hurdle_bps(cfg: &StrategyConfig) -> f64 {
        let rt_maker = (2.0 * cfg.maker_fee_bps.max(0.0)) + (0.25 * cfg.est_slippage_bps.max(0.0));
        (rt_maker * cfg.maker_edge_mult.max(1.0)).max(0.5)
    }

    #[inline]
    pub fn taker_hurdle_bps(cfg: &StrategyConfig) -> f64 {
        let rt_taker = (2.0 * cfg.taker_fee_bps.max(0.0)) + cfg.est_slippage_bps.max(0.0);
        (rt_taker * cfg.taker_edge_mult.max(3.0)).max(1.0)
    }

    /// Selects MakerPostOnly by default; upgrades to TakerAggressive only when edge >= 3x RT taker fees + slippage
    #[inline]
    pub fn select_style(
        cfg: &StrategyConfig,
        expected_edge_bps: f64,
        is_urgent_breakout: bool,
        is_emergency_sl: bool,
    ) -> (ExecutionStyle, f64) {
        if is_emergency_sl {
            return (ExecutionStyle::EmergencyClose, 0.0);
        }
        if !expected_edge_bps.is_finite() || expected_edge_bps <= 0.0 {
            return (ExecutionStyle::None, Self::maker_hurdle_bps(cfg));
        }
        let taker_hurdle = Self::taker_hurdle_bps(cfg);
        if is_urgent_breakout && expected_edge_bps >= taker_hurdle {
            return (ExecutionStyle::TakerAggressive, taker_hurdle);
        }
        let maker_hurdle = Self::maker_hurdle_bps(cfg);
        if expected_edge_bps >= maker_hurdle {
            return (ExecutionStyle::MakerPostOnly, maker_hurdle);
        }
        (ExecutionStyle::None, maker_hurdle)
    }
}
