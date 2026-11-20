#![allow(dead_code)]

use crate::types::SymbolId;

/// Hard pre-trade risk limits and circuit-breaker thresholds (BTC / ETH / SOL)
#[derive(Debug, Clone, Copy)]
pub struct RiskConfig {
    pub max_daily_loss_pct: f64,      // e.g. 2.5% -> hard lock for the day
    pub max_drawdown_pct: f64,        // e.g. 5.0% from peak equity -> kill switch
    pub max_symbol_notional_usd: f64, // Hard cap per symbol (BTC/ETH/SOL)
    pub max_gross_notional_usd: f64,  // Hard cap across all open symbols combined
    pub max_order_notional_usd: f64,  // Single-order fat-finger cap
    pub min_order_notional_usd: f64,  // Exchange minimum dust filter
    pub max_leverage: f64,            // Total gross notional / equity cap
    pub dead_man_timeout_ms: u64,     // Heartbeat timeout (e.g. 5_000 ms) -> Cancel-All
    pub max_orders_per_min: usize,    // Runaway loop / spam guard
    pub max_consecutive_rejects: u32, // Exchange reject storm breaker
    pub require_hard_sl: bool,        // Mandatory exchange-side Stop-Loss on entries
}

impl RiskConfig {
    pub const fn default_crypto() -> Self {
        Self {
            max_daily_loss_pct: 2.5,
            max_drawdown_pct: 5.0,
            max_symbol_notional_usd: 1_000.0,
            max_gross_notional_usd: 2_500.0,
            max_order_notional_usd: 500.0,
            min_order_notional_usd: 10.0,
            max_leverage: 3.0,
            dead_man_timeout_ms: 5_000,
            max_orders_per_min: 30,
            max_consecutive_rejects: 5,
            require_hard_sl: true,
        }
    }
}

/// Deterministic reason for blocking an order or tripping a circuit breaker
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskRejectReason {
    None,
    KillSwitchActive,
    DeadManSwitchTripped,
    DailyLossLimitBreached,
    MaxDrawdownBreached,
    ConsecutiveRejectsTripped,
    OrderRateLimitExceeded,
    SymbolPositionCap,
    GrossExposureCap,
    LeverageCapExceeded,
    MissingHardStopLoss,
    InvalidPriceOrSize,
    BelowMinNotional,
    IntentNotActionable,
}

/// Real-time account equity and exposure snapshot fed into Pre-Trade Risk Gate
#[derive(Debug, Clone, Copy)]
pub struct AccountRiskSnapshot {
    pub equity_usd: f64,
    pub day_start_equity_usd: f64,
    pub peak_equity_usd: f64,
    pub total_gross_exposure_usd: f64,
    pub open_orders_count: u32,
}

impl AccountRiskSnapshot {
    pub const fn new(equity_usd: f64) -> Self {
        Self {
            equity_usd,
            day_start_equity_usd: equity_usd,
            peak_equity_usd: equity_usd,
            total_gross_exposure_usd: 0.0,
            open_orders_count: 0,
        }
    }

    #[inline]
    pub fn is_valid(&self) -> bool {
        self.equity_usd.is_finite()
            && self.equity_usd > 0.0
            && self.day_start_equity_usd.is_finite()
            && self.day_start_equity_usd > 0.0
            && self.peak_equity_usd.is_finite()
            && self.peak_equity_usd > 0.0
            && self.total_gross_exposure_usd.is_finite()
            && self.total_gross_exposure_usd >= 0.0
    }

    /// Daily PnL percentage relative to UTC day-start equity (negative = loss)
    #[inline]
    pub fn daily_pnl_pct(&self) -> f64 {
        if !self.is_valid() {
            return -100.0;
        }
        ((self.equity_usd - self.day_start_equity_usd) / self.day_start_equity_usd) * 100.0
    }

    /// Peak-to-current drawdown percentage (positive = drawdown depth)
    #[inline]
    pub fn drawdown_pct(&self) -> f64 {
        if !self.is_valid() {
            return 100.0;
        }
        let peak = self.peak_equity_usd.max(self.equity_usd);
        (((peak - self.equity_usd) / peak) * 100.0).max(0.0)
    }
}

/// Zero-allocation verdict returned by RiskEngine to Execution
#[derive(Debug, Clone, Copy)]
pub struct RiskVerdict {
    pub symbol_id: SymbolId,
    pub approved: bool,
    pub reject_reason: RiskRejectReason,
    pub approved_notional_usd: f64,
    pub approved_qty: f64,
    pub require_cancel_all: bool,
    pub require_emergency_flatten: bool,
}

impl RiskVerdict {
    pub const fn rejected(
        symbol_id: SymbolId,
        reason: RiskRejectReason,
        require_cancel_all: bool,
        require_emergency_flatten: bool,
    ) -> Self {
        Self {
            symbol_id,
            approved: false,
            reject_reason: reason,
            approved_notional_usd: 0.0,
            approved_qty: 0.0,
            require_cancel_all,
            require_emergency_flatten,
        }
    }

    pub const fn approved(
        symbol_id: SymbolId,
        notional_usd: f64,
        qty: f64,
        require_cancel_all: bool,
    ) -> Self {
        Self {
            symbol_id,
            approved: true,
            reject_reason: RiskRejectReason::None,
            approved_notional_usd: notional_usd,
            approved_qty: qty,
            require_cancel_all,
            require_emergency_flatten: false,
        }
    }
}
