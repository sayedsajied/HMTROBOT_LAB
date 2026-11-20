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
pub const MAX_RATE_WINDOW: usize = 64;

/// Dead-Man's Switch: trips and demands Cancel-All if live heartbeat goes stale
#[derive(Debug, Clone, Copy)]
pub struct DeadManSwitch {
    pub timeout_ms: u64,
    pub last_heartbeat_ms: u64,
    pub tripped: bool,
}

impl DeadManSwitch {
    pub const fn new(timeout_ms: u64) -> Self {
        Self {
            timeout_ms,
            last_heartbeat_ms: 0,
            tripped: false,
        }
    }

    #[inline]
    pub fn heartbeat(&mut self, now_ms: u64) {
        if now_ms > 0 {
            self.last_heartbeat_ms = now_ms;
            self.tripped = false;
        }
    }

    /// Returns true if no valid heartbeat has been received within timeout_ms
    #[inline]
    pub fn check(&mut self, now_ms: u64) -> bool {
        if self.last_heartbeat_ms == 0 || now_ms < self.last_heartbeat_ms {
            self.tripped = true;
            return true;
        }
        let elapsed = now_ms.saturating_sub(self.last_heartbeat_ms);
        self.tripped = elapsed > self.timeout_ms;
        self.tripped
    }
}

/// Zero-allocation sliding 60-second order rate limiter (runaway loop guard)
#[derive(Debug, Clone, Copy)]
pub struct OrderRateLimiter {
    timestamps_ms: [u64; MAX_RATE_WINDOW],
    head: usize,
    count: usize,
    pub max_per_min: usize,
}

impl OrderRateLimiter {
    pub const fn new(max_per_min: usize) -> Self {
        Self {
            timestamps_ms: [0; MAX_RATE_WINDOW],
            head: 0,
            count: 0,
            max_per_min,
        }
    }

    /// Counts submitted orders within the last 60,000 ms
    pub fn recent_count(&self, now_ms: u64) -> usize {
        let mut recent = 0usize;
        let len = self.count.min(MAX_RATE_WINDOW);
        for i in 0..len {
            let ts = self.timestamps_ms[i];
            if ts > 0 && now_ms.saturating_sub(ts) < 60_000 {
                recent += 1;
            }
        }
        recent
    }

    #[inline]
    pub fn can_submit(&self, now_ms: u64) -> bool {
        let limit = self.max_per_min.min(MAX_RATE_WINDOW);
        self.recent_count(now_ms) < limit
    }

    #[inline]
    pub fn record_submit(&mut self, now_ms: u64) {
        self.timestamps_ms[self.head] = now_ms;
        self.head = (self.head + 1) % MAX_RATE_WINDOW;
        if self.count < MAX_RATE_WINDOW {
            self.count += 1;
        }
    }
}

/// Hard Account Circuit Breaker: Daily Loss, Max Drawdown, Reject Storm & Kill Switch
#[derive(Debug, Clone, Copy)]
pub struct CircuitBreaker {
    pub kill_switch_active: bool,
    pub daily_loss_tripped: bool,
    pub drawdown_tripped: bool,
    pub consecutive_rejects: u32,
    pub rejects_tripped: bool,
    pub last_trip_reason: RiskRejectReason,
}

impl CircuitBreaker {
    pub const fn new() -> Self {
        Self {
            kill_switch_active: false,
            daily_loss_tripped: false,
            drawdown_tripped: false,
            consecutive_rejects: 0,
            rejects_tripped: false,
            last_trip_reason: RiskRejectReason::None,
        }
    }

    /// Evaluates account equity against hard daily-loss and max-drawdown breakers
    pub fn evaluate_account(
        &mut self,
        cfg: &RiskConfig,
        snap: &AccountRiskSnapshot,
    ) -> RiskRejectReason {
        if self.kill_switch_active {
            return RiskRejectReason::KillSwitchActive;
        }
        if !snap.is_valid() {
            return RiskRejectReason::InvalidPriceOrSize;
        }
        if snap.drawdown_pct() >= cfg.max_drawdown_pct.abs() {
            self.drawdown_tripped = true;
            self.kill_switch_active = true;
            self.last_trip_reason = RiskRejectReason::MaxDrawdownBreached;
            return RiskRejectReason::MaxDrawdownBreached;
        }
        if self.daily_loss_tripped || snap.daily_pnl_pct() <= -cfg.max_daily_loss_pct.abs() {
            self.daily_loss_tripped = true;
            self.last_trip_reason = RiskRejectReason::DailyLossLimitBreached;
            return RiskRejectReason::DailyLossLimitBreached;
        }
        if self.rejects_tripped {
            return RiskRejectReason::ConsecutiveRejectsTripped;
        }
        RiskRejectReason::None
    }

    /// Tracks consecutive exchange order rejections and trips breaker on storm
    #[inline]
    pub fn record_exchange_reject(&mut self, cfg: &RiskConfig) {
        self.consecutive_rejects = self.consecutive_rejects.saturating_add(1);
        if self.consecutive_rejects >= cfg.max_consecutive_rejects.max(1) {
            self.rejects_tripped = true;
            self.last_trip_reason = RiskRejectReason::ConsecutiveRejectsTripped;
        }
    }

    #[inline]
    pub fn record_exchange_accept(&mut self) {
        self.consecutive_rejects = 0;
    }

    #[inline]
    pub fn trigger_manual_kill(&mut self) {
        self.kill_switch_active = true;
        self.last_trip_reason = RiskRejectReason::KillSwitchActive;
    }

    /// Resets daily loss lock on UTC midnight rollover (preserves hard kill switch)
    #[inline]
    pub fn reset_daily_lock(&mut self) {
        self.daily_loss_tripped = false;
        self.rejects_tripped = false;
        self.consecutive_rejects = 0;
        if !self.kill_switch_active {
            self.last_trip_reason = RiskRejectReason::None;
        }
    }

    #[inline]
    pub fn is_halted(&self) -> bool {
        self.kill_switch_active
            || self.daily_loss_tripped
            || self.drawdown_tripped
            || self.rejects_tripped
    }
}
use crate::sniper_strategy::{ExecutionStyle, InventoryState, StrategyIntent, TradeDirection};

/// Master Pre-Trade Risk Gatekeeper (Jump Pillar #6 — Zero-Heap, Silent, Zero-Panic)
#[derive(Debug, Clone, Copy)]
pub struct RiskEngine {
    pub config: RiskConfig,
    pub dead_man: DeadManSwitch,
    pub rate_limiter: OrderRateLimiter,
    pub breaker: CircuitBreaker,
}

impl RiskEngine {
    pub const fn new(config: RiskConfig) -> Self {
        Self {
            config,
            dead_man: DeadManSwitch::new(config.dead_man_timeout_ms),
            rate_limiter: OrderRateLimiter::new(config.max_orders_per_min),
            breaker: CircuitBreaker::new(),
        }
    }

    #[inline]
    pub fn on_heartbeat(&mut self, now_ms: u64) {
        self.dead_man.heartbeat(now_ms);
    }

    #[inline]
    pub fn on_exchange_accept(&mut self) {
        self.breaker.record_exchange_accept();
    }

    #[inline]
    pub fn on_exchange_reject(&mut self) {
        self.breaker.record_exchange_reject(&self.config);
    }

    #[inline]
    pub fn trigger_kill_switch(&mut self) {
        self.breaker.trigger_manual_kill();
    }

    #[inline]
    pub fn on_utc_day_rollover(&mut self) {
        self.breaker.reset_daily_lock();
    }

    /// True if order direction strictly reduces existing open inventory
    #[inline]
    pub fn is_risk_reducing(dir: TradeDirection, inv: &InventoryState) -> bool {
        match dir {
            TradeDirection::Long => inv.net_qty < -1e-12,
            TradeDirection::Short => inv.net_qty > 1e-12,
            TradeDirection::Flat => false,
        }
    }

    /// Validates mandatory exchange-side hard stop-loss orientation
    #[inline]
    pub fn has_valid_hard_sl(dir: TradeDirection, limit_price: f64, sl_price: f64) -> bool {
        if !limit_price.is_finite() || !sl_price.is_finite() || limit_price <= 0.0 || sl_price <= 0.0 {
            return false;
        }
        match dir {
            TradeDirection::Long => sl_price < limit_price,
            TradeDirection::Short => sl_price > limit_price,
            TradeDirection::Flat => false,
        }
    }

    /// Clamps requested entry notional against Symbol, Gross, Leverage, and Single-Order caps
    pub fn clamp_entry_notional(
        &self,
        requested_usd: f64,
        inv: &InventoryState,
        snap: &AccountRiskSnapshot,
    ) -> Result<f64, RiskRejectReason> {
        if !requested_usd.is_finite() || requested_usd <= 0.0 || !snap.is_valid() {
            return Err(RiskRejectReason::InvalidPriceOrSize);
        }

        let sym_used = inv.notional_usd.abs();
        let sym_room = (self.config.max_symbol_notional_usd - sym_used).max(0.0);
        if sym_room < self.config.min_order_notional_usd {
            return Err(RiskRejectReason::SymbolPositionCap);
        }

        let gross_room = (self.config.max_gross_notional_usd - snap.total_gross_exposure_usd).max(0.0);
        if gross_room < self.config.min_order_notional_usd {
            return Err(RiskRejectReason::GrossExposureCap);
        }

        let max_lev_notional = snap.equity_usd * self.config.max_leverage.max(0.1);
        let lev_room = (max_lev_notional - snap.total_gross_exposure_usd).max(0.0);
        if lev_room < self.config.min_order_notional_usd {
            return Err(RiskRejectReason::LeverageCapExceeded);
        }

        let capped = requested_usd
            .min(self.config.max_order_notional_usd)
            .min(sym_room)
            .min(gross_room)
            .min(lev_room);

        if capped < self.config.min_order_notional_usd {
            return Err(RiskRejectReason::BelowMinNotional);
        }
        Ok(capped)
    }

    /// Primary hot-path pre-trade gate: evaluates every StrategyIntent before Execution
    pub fn evaluate(
        &mut self,
        intent: &StrategyIntent,
        inv: &InventoryState,
        snap: &AccountRiskSnapshot,
        now_ms: u64,
    ) -> RiskVerdict {
        let sym = intent.symbol_id;

        // 1. Dead-Man's Switch check
        let dms_tripped = self.dead_man.check(now_ms);

        // 2. Account-level Circuit Breakers (Daily Loss, Max Drawdown, Kill Switch, Reject Storm)
        let breaker_reason = self.breaker.evaluate_account(&self.config, snap);
        let flatten_needed = matches!(
            breaker_reason,
            RiskRejectReason::KillSwitchActive | RiskRejectReason::MaxDrawdownBreached
        );

        let is_emergency = intent.style == ExecutionStyle::EmergencyClose;
        let reducing = Self::is_risk_reducing(intent.direction, inv);

        // Always allow validated EmergencyClose / risk-reducing exits even if breakers tripped
        if is_emergency && reducing {
            if !intent.limit_price.is_finite() || intent.limit_price <= 0.0 || inv.net_qty.abs() <= 1e-12 {
                return RiskVerdict::rejected(sym, RiskRejectReason::InvalidPriceOrSize, true, flatten_needed);
            }
            let close_qty = inv.net_qty.abs();
            let close_notional = close_qty * intent.limit_price;
            self.rate_limiter.record_submit(now_ms);
            return RiskVerdict::approved(sym, close_notional, close_qty, true);
        }

        if dms_tripped {
            return RiskVerdict::rejected(sym, RiskRejectReason::DeadManSwitchTripped, true, false);
        }

        if breaker_reason != RiskRejectReason::None {
            return RiskVerdict::rejected(sym, breaker_reason, true, flatten_needed);
        }

        // 3. Actionable intent & basic price/size sanity
        if intent.direction == TradeDirection::Flat || intent.style == ExecutionStyle::None {
            return RiskVerdict::rejected(sym, RiskRejectReason::IntentNotActionable, intent.cancel_open_orders, false);
        }

        if !intent.limit_price.is_finite()
            || intent.limit_price <= 0.0
            || !intent.size_fraction.is_finite()
            || intent.size_fraction <= 0.0
        {
            return RiskVerdict::rejected(sym, RiskRejectReason::InvalidPriceOrSize, intent.cancel_open_orders, false);
        }

        // 4. Order-rate runaway loop guard
        if !self.rate_limiter.can_submit(now_ms) {
            return RiskVerdict::rejected(sym, RiskRejectReason::OrderRateLimitExceeded, false, false);
        }

        // 5. Risk-reducing regular order bypasses expansion caps & hard-SL requirement
        if reducing {
            let max_close_usd = inv.notional_usd.abs();
            let req_usd = (intent.size_fraction.clamp(0.0, 1.0) * self.config.max_symbol_notional_usd)
                .min(max_close_usd)
                .min(self.config.max_order_notional_usd);
            if req_usd < self.config.min_order_notional_usd {
                return RiskVerdict::rejected(sym, RiskRejectReason::BelowMinNotional, intent.cancel_open_orders, false);
            }
            let qty = (req_usd / intent.limit_price).min(inv.net_qty.abs());
            self.rate_limiter.record_submit(now_ms);
            return RiskVerdict::approved(sym, qty * intent.limit_price, qty, intent.cancel_open_orders);
        }

        // 6. Mandatory Exchange-Side Hard Stop-Loss for new risk-increasing entries
        if self.config.require_hard_sl
            && !Self::has_valid_hard_sl(intent.direction, intent.limit_price, intent.stop_loss_price)
        {
            return RiskVerdict::rejected(sym, RiskRejectReason::MissingHardStopLoss, intent.cancel_open_orders, false);
        }

        // 7. Position, Gross Exposure, Leverage & Fat-Finger Caps
        let requested_usd = intent.size_fraction.clamp(0.0, 1.0) * self.config.max_symbol_notional_usd;
        let approved_usd = match self.clamp_entry_notional(requested_usd, inv, snap) {
            Ok(usd) => usd,
            Err(reason) => {
                return RiskVerdict::rejected(sym, reason, intent.cancel_open_orders, false);
            }
        };

        let qty = approved_usd / intent.limit_price;
        if !qty.is_finite() || qty <= 0.0 {
            return RiskVerdict::rejected(sym, RiskRejectReason::InvalidPriceOrSize, intent.cancel_open_orders, false);
        }

        self.rate_limiter.record_submit(now_ms);
        RiskVerdict::approved(sym, approved_usd, qty, intent.cancel_open_orders)
    }
}

#[cfg(test)]
mod risk_tests {
    use super::*;

    #[test]
    fn test_dead_man_switch_trips_on_stale_heartbeat() {
        let mut dms = DeadManSwitch::new(5_000);
        assert!(dms.check(1_000)); // Never received heartbeat

        dms.heartbeat(10_000);
        assert!(!dms.check(14_999));
        assert!(dms.check(15_001)); // >5000ms elapsed
    }

    #[test]
    fn test_circuit_breaker_daily_loss_and_drawdown() {
        let cfg = RiskConfig::default_crypto();
        let mut cb = CircuitBreaker::new();

        // Healthy account
        let ok_snap = AccountRiskSnapshot::new(1_000.0);
        assert_eq!(cb.evaluate_account(&cfg, &ok_snap), RiskRejectReason::None);

        // Daily loss > 2.5%
        let daily_loss = AccountRiskSnapshot {
            equity_usd: 970.0,
            day_start_equity_usd: 1_000.0,
            peak_equity_usd: 1_000.0,
            total_gross_exposure_usd: 0.0,
            open_orders_count: 0,
        };
        assert_eq!(
            cb.evaluate_account(&cfg, &daily_loss),
            RiskRejectReason::DailyLossLimitBreached
        );
        assert!(cb.daily_loss_tripped);

        // Max drawdown >= 5.0% activates permanent kill switch
        let dd_snap = AccountRiskSnapshot {
            equity_usd: 940.0,
            day_start_equity_usd: 950.0,
            peak_equity_usd: 1_000.0,
            total_gross_exposure_usd: 0.0,
            open_orders_count: 0,
        };
        assert_eq!(
            cb.evaluate_account(&cfg, &dd_snap),
            RiskRejectReason::MaxDrawdownBreached
        );
        assert!(cb.kill_switch_active);
    }

    #[test]
    fn test_hard_sl_and_notional_caps() {
        let cfg = RiskConfig::default_crypto();
        let engine = RiskEngine::new(cfg);

        assert!(RiskEngine::has_valid_hard_sl(TradeDirection::Long, 90_000.0, 89_500.0));
        assert!(!RiskEngine::has_valid_hard_sl(TradeDirection::Long, 90_000.0, 90_500.0));
        assert!(RiskEngine::has_valid_hard_sl(TradeDirection::Short, 90_000.0, 90_500.0));

        let snap = AccountRiskSnapshot {
            equity_usd: 500.0, // max_leverage 3x => 1500 max gross
            day_start_equity_usd: 500.0,
            peak_equity_usd: 500.0,
            total_gross_exposure_usd: 1_400.0, // only $100 leverage room left
            open_orders_count: 1,
        };
        let inv = InventoryState::flat();
        let capped = engine.clamp_entry_notional(400.0, &inv, &snap);
        assert_eq!(capped, Ok(100.0));
    }

    #[test]
    fn test_order_rate_limiter_window() {
        let mut rl = OrderRateLimiter::new(3);
        rl.record_submit(10_000);
        rl.record_submit(20_000);
        rl.record_submit(30_000);
        assert!(!rl.can_submit(40_000)); // 3 orders within 60s
        assert!(rl.can_submit(75_000));  // 10_000 expired (>60s old)
    }
    }
