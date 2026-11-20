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
use crate::ai_router::AiRegime;
use crate::math_kernel::AlphaBatch;

/// Snapshot of real-time adverse selection & whale-sweep danger
#[derive(Debug, Clone, Copy)]
pub struct ToxicFlowAssessment {
    pub toxic_score: f64,      // [0.0, 1.0]
    pub whale_sweep_dir: f64,  // [-1.0, 1.0] (+ buy sweep, - sell sweep)
    pub pull_resting_orders: bool,
    pub block_new_entries: bool,
}

/// Jump Pillar #3: Adverse Selection & Whale-Sweep Toxic Flow Shield
#[derive(Debug, Clone, Copy)]
pub struct ToxicFlowShield {
    pub ewma_toxicity: f64,
    pub pull_threshold: f64,
    pub block_threshold: f64,
}

impl ToxicFlowShield {
    pub const fn new() -> Self {
        Self {
            ewma_toxicity: 0.0,
            pull_threshold: 0.62,
            block_threshold: 0.72,
        }
    }

    /// Evaluates toxicity from E05 (VPIN), E07 (Kyle), E11 (Whale Sweep), E12 (Spoof), Spread & HMM
    pub fn evaluate(
        &mut self,
        batch: &AlphaBatch,
        regime: AiRegime,
        spread_bps: f64,
        max_spread_bps: f64,
    ) -> ToxicFlowAssessment {
        let vpin = batch.signals[4].score.abs().clamp(0.0, 1.0);
        let kyle = batch.signals[6].score.abs().clamp(0.0, 1.0);
        let sweep_signed = batch.signals[10].score.clamp(-1.0, 1.0);
        let sweep_abs = sweep_signed.abs();
        let spoof_abs = batch.signals[11].score.abs().clamp(0.0, 1.0);

        let sp_ratio = if spread_bps.is_finite() && max_spread_bps > 1e-6 {
            (spread_bps / max_spread_bps).clamp(0.0, 1.5)
        } else {
            1.0
        };
        let regime_pen = match regime {
            AiRegime::ToxicCrisis => 0.35,
            AiRegime::VolatileBurst => 0.10,
            _ => 0.0,
        };

        let raw_tox = ((vpin * 0.30)
            + (sweep_abs * 0.32)
            + (kyle * 0.14)
            + (spoof_abs * 0.10)
            + (sp_ratio.min(1.0) * 0.14)
            + regime_pen)
            .clamp(0.0, 1.0);

        self.ewma_toxicity = ((self.ewma_toxicity * 0.65) + (raw_tox * 0.35)).clamp(0.0, 1.0);
        let peak_tox = self.ewma_toxicity.max(raw_tox);

        let pull_resting = peak_tox >= self.pull_threshold
            || sweep_abs >= 0.75
            || regime == AiRegime::ToxicCrisis;
        let block_new = peak_tox >= self.block_threshold
            || regime == AiRegime::ToxicCrisis
            || sp_ratio >= 1.0;

        ToxicFlowAssessment {
            toxic_score: peak_tox,
            whale_sweep_dir: sweep_signed,
            pull_resting_orders: pull_resting,
            block_new_entries: block_new,
        }
    }

    /// True if taking/resting in `direction` directly stands in front of an opposing whale sweep
    #[inline]
    pub fn is_adverse_for(&self, direction: TradeDirection, assess: &ToxicFlowAssessment) -> bool {
        match direction {
            TradeDirection::Long => assess.whale_sweep_dir < -0.55,
            TradeDirection::Short => assess.whale_sweep_dir > 0.55,
            TradeDirection::Flat => false,
        }
    }
}

/// Jump Pillar #4: Avellaneda-Stoikov Dynamic Inventory & Quote Skew Controller
#[derive(Debug, Clone, Copy)]
pub struct InventorySkewModel {
    pub gamma_risk: f64,
    pub max_skew_bps: f64,
}

impl InventorySkewModel {
    pub const fn new(gamma_risk: f64, max_skew_bps: f64) -> Self {
        Self {
            gamma_risk,
            max_skew_bps,
        }
    }

    /// Reservation price skew in bps: positive long inventory -> negative skew (encourages selling)
    #[inline]
    pub fn reservation_skew_bps(
        &self,
        inv: &InventoryState,
        max_inv_usd: f64,
        vol_bps: f64,
    ) -> f64 {
        let q = inv.utilization(max_inv_usd);
        let v = if vol_bps.is_finite() { vol_bps.clamp(0.2, 25.0) } else { 1.0 };
        let raw = -q * self.gamma_risk * v;
        raw.clamp(-self.max_skew_bps, self.max_skew_bps)
    }

    /// Dampens size when adding to existing exposure; preserves size when reducing/flattening risk
    #[inline]
    pub fn adjust_size_fraction(
        &self,
        base_frac: f64,
        dir: TradeDirection,
        inv: &InventoryState,
        max_inv_usd: f64,
    ) -> f64 {
        if !base_frac.is_finite() || base_frac <= 0.0 {
            return 0.0;
        }
        let util = inv.utilization(max_inv_usd);
        let adding_same_side = match dir {
            TradeDirection::Long => util > 0.0,
            TradeDirection::Short => util < 0.0,
            TradeDirection::Flat => return 0.0,
        };

        if adding_same_side {
            let headroom = (1.0 - util.abs()).clamp(0.0, 1.0);
            (base_frac * headroom).clamp(0.0, headroom)
        } else {
            // Risk-reducing or flat entry
            base_frac.clamp(0.0, 1.0)
        }
    }

    /// Computes passive Post-Only or aggressive limit price incorporating inventory skew
    #[inline]
    pub fn skewed_limit_price(
        &self,
        mid: f64,
        bid: f64,
        ask: f64,
        dir: TradeDirection,
        style: ExecutionStyle,
        skew_bps: f64,
    ) -> f64 {
        if !mid.is_finite() || mid <= 0.0 {
            return 0.0;
        }
        let b = if bid.is_finite() && bid > 0.0 { bid } else { mid };
        let a = if ask.is_finite() && ask > 0.0 { ask } else { mid };
        let skew_mult = 1.0 + (skew_bps / 10_000.0);

        match (dir, style) {
            (TradeDirection::Long, ExecutionStyle::MakerPostOnly) => {
                // Stay passive at or below best bid
                (b * skew_mult).min(b)
            }
            (TradeDirection::Short, ExecutionStyle::MakerPostOnly) => {
                // Stay passive at or above best ask
                (a * skew_mult).max(a)
            }
            (TradeDirection::Long, ExecutionStyle::TakerAggressive | ExecutionStyle::EmergencyClose) => a,
            (TradeDirection::Short, ExecutionStyle::TakerAggressive | ExecutionStyle::EmergencyClose) => b,
            _ => 0.0,
        }
    }
}
