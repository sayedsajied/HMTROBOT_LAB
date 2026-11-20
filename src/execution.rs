#![allow(dead_code)]

use crate::sniper_strategy::{ExecutionStyle, TradeDirection};
use crate::types::SymbolId;

pub const MAX_TRACKED_ORDERS: usize = 32;

/// Switches only the execution sink; entire upstream pipeline stays 100% identical
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionMode {
    Paper,
    Live,
}

/// Deterministic Order State Machine (Wintermute Rule #9)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecOrderState {
    Idle,
    Submitting,
    Open,
    PartiallyFilled,
    Filled,
    Cancelled,
    Failed,
}

impl ExecOrderState {
    #[inline]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Filled | Self::Cancelled | Self::Failed)
    }

    #[inline]
    pub const fn is_active(self) -> bool {
        matches!(self, Self::Submitting | Self::Open | Self::PartiallyFilled)
    }

    /// Strict finite-state transition validator
    #[inline]
    pub const fn can_transition_to(self, next: Self) -> bool {
        match (self, next) {
            (Self::Idle, Self::Submitting) => true,
            (Self::Submitting, Self::Open | Self::PartiallyFilled | Self::Filled | Self::Cancelled | Self::Failed) => true,
            (Self::Open, Self::PartiallyFilled | Self::Filled | Self::Cancelled | Self::Failed) => true,
            (Self::PartiallyFilled, Self::PartiallyFilled | Self::Filled | Self::Cancelled | Self::Failed) => true,
            _ => false,
        }
    }
}

/// Zero-allocation tracked order record (BTC / ETH / SOL)
#[derive(Debug, Clone, Copy)]
pub struct ExecOrder {
    pub client_order_id: u64,
    pub exchange_order_id: u64,
    pub symbol_id: SymbolId,
    pub direction: TradeDirection,
    pub style: ExecutionStyle,
    pub state: ExecOrderState,
    pub limit_price: f64,
    pub qty: f64,
    pub filled_qty: f64,
    pub avg_fill_price: f64,
    pub fee_usd: f64,
    pub stop_loss_price: f64,
    pub take_profit_price: f64,
    pub is_reduce_only: bool,
    pub created_ms: u64,
    pub updated_ms: u64,
}

impl ExecOrder {
    pub const fn empty() -> Self {
        Self {
            client_order_id: 0,
            exchange_order_id: 0,
            symbol_id: SymbolId(0),
            direction: TradeDirection::Flat,
            style: ExecutionStyle::None,
            state: ExecOrderState::Idle,
            limit_price: 0.0,
            qty: 0.0,
            filled_qty: 0.0,
            avg_fill_price: 0.0,
            fee_usd: 0.0,
            stop_loss_price: 0.0,
            take_profit_price: 0.0,
            is_reduce_only: false,
            created_ms: 0,
            updated_ms: 0,
        }
    }

    #[inline]
    pub fn remaining_qty(&self) -> f64 {
        (self.qty - self.filled_qty).max(0.0)
    }

    /// Transitions state deterministically; returns false on illegal transition
    pub fn transition(&mut self, next: ExecOrderState, now_ms: u64) -> bool {
        if !self.state.can_transition_to(next) {
            return false;
        }
        self.state = next;
        self.updated_ms = now_ms;
        true
    }

    /// Applies an incremental or full fill and updates volume-weighted average fill price
    pub fn apply_fill(&mut self, fill_qty: f64, fill_price: f64, fee_usd: f64, now_ms: u64) -> bool {
        if !fill_qty.is_finite()
            || !fill_price.is_finite()
            || !fee_usd.is_finite()
            || fill_qty <= 0.0
            || fill_price <= 0.0
            || !self.state.is_active()
        {
            return false;
        }
        let applied = fill_qty.min(self.remaining_qty());
        if applied <= 1e-12 {
            return false;
        }

        let prev_notional = self.filled_qty * self.avg_fill_price;
        let new_notional = prev_notional + (applied * fill_price);
        self.filled_qty += applied;
        if self.filled_qty > 1e-12 {
            self.avg_fill_price = new_notional / self.filled_qty;
        }
        self.fee_usd += fee_usd.max(0.0);

        let next_state = if self.remaining_qty() <= 1e-9 {
            ExecOrderState::Filled
        } else {
            ExecOrderState::PartiallyFilled
        };
        self.transition(next_state, now_ms)
    }
}

/// Realized fill event emitted identically by Paper Simulator and Live Exchange API
#[derive(Debug, Clone, Copy)]
pub struct FillEvent {
    pub client_order_id: u64,
    pub symbol_id: SymbolId,
    pub direction: TradeDirection,
    pub fill_qty: f64,
    pub fill_price: f64,
    pub fee_usd: f64,
    pub is_maker: bool,
    pub is_hard_sl: bool,
    pub is_take_profit: bool,
    pub timestamp_ms: u64,
}

/// Fixed-capacity stack registry managing all active and recent order lifecycles
#[derive(Debug, Clone, Copy)]
pub struct OrderStateMachine {
    slots: [ExecOrder; MAX_TRACKED_ORDERS],
    next_client_id: u64,
}

impl OrderStateMachine {
    pub const fn new() -> Self {
        Self {
            slots: [ExecOrder::empty(); MAX_TRACKED_ORDERS],
            next_client_id: 1,
        }
    }

    /// Allocates an Idle/terminal slot and transitions new order to Submitting
    pub fn create_submitting(
        &mut self,
        symbol_id: SymbolId,
        direction: TradeDirection,
        style: ExecutionStyle,
        limit_price: f64,
        qty: f64,
        stop_loss_price: f64,
        take_profit_price: f64,
        is_reduce_only: bool,
        now_ms: u64,
    ) -> Option<u64> {
        if !limit_price.is_finite() || !qty.is_finite() || limit_price <= 0.0 || qty <= 0.0 {
            return None;
        }
        let slot_idx = self.find_free_slot()?;
        let cid = self.next_client_id;
        self.next_client_id = self.next_client_id.saturating_add(1);

        let mut ord = ExecOrder::empty();
        ord.client_order_id = cid;
        ord.symbol_id = symbol_id;
        ord.direction = direction;
        ord.style = style;
        ord.limit_price = limit_price;
        ord.qty = qty;
        ord.stop_loss_price = if stop_loss_price.is_finite() { stop_loss_price.max(0.0) } else { 0.0 };
        ord.take_profit_price = if take_profit_price.is_finite() { take_profit_price.max(0.0) } else { 0.0 };
        ord.is_reduce_only = is_reduce_only;
        ord.created_ms = now_ms;
        ord.updated_ms = now_ms;

        if !ord.transition(ExecOrderState::Submitting, now_ms) {
            return None;
        }
        self.slots[slot_idx] = ord;
        Some(cid)
    }

    fn find_free_slot(&self) -> Option<usize> {
        for i in 0..MAX_TRACKED_ORDERS {
            if self.slots[i].state == ExecOrderState::Idle {
                return Some(i);
            }
        }
        // Recycle oldest terminal slot if all slots have been used
        let mut oldest_idx = None;
        let mut oldest_ts = u64::MAX;
        for i in 0..MAX_TRACKED_ORDERS {
            if self.slots[i].state.is_terminal() && self.slots[i].updated_ms <= oldest_ts {
                oldest_ts = self.slots[i].updated_ms;
                oldest_idx = Some(i);
            }
        }
        oldest_idx
    }

    pub fn get(&self, client_order_id: u64) -> Option<&ExecOrder> {
        self.slots.iter().find(|o| o.client_order_id == client_order_id && o.state != ExecOrderState::Idle)
    }

    pub fn get_mut(&mut self, client_order_id: u64) -> Option<&mut ExecOrder> {
        self.slots.iter_mut().find(|o| o.client_order_id == client_order_id && o.state != ExecOrderState::Idle)
    }

    pub fn active_count_for(&self, symbol_id: SymbolId) -> usize {
        let mut count = 0usize;
        for o in self.slots.iter() {
            if o.symbol_id == symbol_id && o.state.is_active() {
                count += 1;
            }
        }
        count
    }

    pub fn total_active_count(&self) -> usize {
        let mut count = 0usize;
        for o in self.slots.iter() {
            if o.state.is_active() {
                count += 1;
            }
        }
        count
    }

    /// Cancels all active orders for a specific symbol; returns number cancelled
    pub fn cancel_all_for_symbol(&mut self, symbol_id: SymbolId, now_ms: u64) -> usize {
        let mut cancelled = 0usize;
        for o in self.slots.iter_mut() {
            if o.symbol_id == symbol_id && o.state.is_active() {
                if o.transition(ExecOrderState::Cancelled, now_ms) {
                    cancelled += 1;
                }
            }
        }
        cancelled
    }

    /// Cancels every active order across all symbols (Dead-Man's Switch / Kill-Switch)
    pub fn cancel_all(&mut self, now_ms: u64) -> usize {
        let mut cancelled = 0usize;
        for o in self.slots.iter_mut() {
            if o.state.is_active() && o.transition(ExecOrderState::Cancelled, now_ms) {
                cancelled += 1;
            }
        }
        cancelled
    }

    #[inline]
    pub fn slots(&self) -> &[ExecOrder; MAX_TRACKED_ORDERS] {
        &self.slots
    }
}
/// Realistic local fee & slippage parameters for 100% Paper-to-Live parity
#[derive(Debug, Clone, Copy)]
pub struct PaperSimConfig {
    pub maker_fee_bps: f64,
    pub taker_fee_bps: f64,
    pub taker_slippage_bps: f64,
    pub emergency_slippage_bps: f64,
}

impl PaperSimConfig {
    pub const fn default_crypto() -> Self {
        Self {
            maker_fee_bps: 2.0,
            taker_fee_bps: 5.0,
            taker_slippage_bps: 1.5,
            emergency_slippage_bps: 4.0,
        }
    }
}

/// Deterministic Paper Execution Engine (same FillEvent & state transitions as Live)
#[derive(Debug, Clone, Copy)]
pub struct PaperSimulator {
    pub cfg: PaperSimConfig,
    next_sim_exchange_id: u64,
}

impl PaperSimulator {
    pub const fn new(cfg: PaperSimConfig) -> Self {
        Self {
            cfg,
            next_sim_exchange_id: 100_000,
        }
    }

    #[inline]
    fn alloc_exchange_id(&mut self) -> u64 {
        let id = self.next_sim_exchange_id;
        self.next_sim_exchange_id = self.next_sim_exchange_id.saturating_add(1);
        id
    }

    /// Processes a newly created Submitting order against current top-of-book (bid, ask)
    pub fn submit_order(
        &mut self,
        order: &mut ExecOrder,
        bid: f64,
        ask: f64,
        now_ms: u64,
    ) -> Option<FillEvent> {
        if order.state != ExecOrderState::Submitting
            || !bid.is_finite()
            || !ask.is_finite()
            || bid <= 0.0
            || ask <= bid
        {
            order.transition(ExecOrderState::Failed, now_ms);
            return None;
        }

        order.exchange_order_id = self.alloc_exchange_id();

        match order.style {
            ExecutionStyle::MakerPostOnly => {
                // Post-Only rule: reject/cancel immediately if limit would cross the spread
                let would_cross = match order.direction {
                    TradeDirection::Long => order.limit_price >= ask,
                    TradeDirection::Short => order.limit_price <= bid,
                    TradeDirection::Flat => true,
                };
                if would_cross {
                    order.transition(ExecOrderState::Cancelled, now_ms);
                    return None;
                }
                order.transition(ExecOrderState::Open, now_ms);
                None
            }
            ExecutionStyle::TakerAggressive | ExecutionStyle::EmergencyClose => {
                let slip_bps = if order.style == ExecutionStyle::EmergencyClose {
                    self.cfg.emergency_slippage_bps.max(0.0)
                } else {
                    self.cfg.taker_slippage_bps.max(0.0)
                };
                let fill_px = match order.direction {
                    TradeDirection::Long => ask * (1.0 + (slip_bps / 10_000.0)),
                    TradeDirection::Short => bid * (1.0 - (slip_bps / 10_000.0)),
                    TradeDirection::Flat => {
                        order.transition(ExecOrderState::Failed, now_ms);
                        return None;
                    }
                };
                let fill_qty = order.remaining_qty();
                let fee_usd = (fill_qty * fill_px) * (self.cfg.taker_fee_bps.max(0.0) / 10_000.0);
                if order.apply_fill(fill_qty, fill_px, fee_usd, now_ms) {
                    Some(FillEvent {
                        client_order_id: order.client_order_id,
                        symbol_id: order.symbol_id,
                        direction: order.direction,
                        fill_qty,
                        fill_price: fill_px,
                        fee_usd,
                        is_maker: false,
                        is_hard_sl: order.style == ExecutionStyle::EmergencyClose,
                        is_take_profit: false,
                        timestamp_ms: now_ms,
                    })
                } else {
                    None
                }
            }
            ExecutionStyle::None => {
                order.transition(ExecOrderState::Failed, now_ms);
                None
            }
        }
    }

    /// Evaluates resting Post-Only orders against new (bid, ask); returns number of fills written
    pub fn on_market_quotes(
        &self,
        osm: &mut OrderStateMachine,
        symbol_id: SymbolId,
        bid: f64,
        ask: f64,
        now_ms: u64,
        out_fills: &mut [Option<FillEvent>; MAX_TRACKED_ORDERS],
    ) -> usize {
        if !bid.is_finite() || !ask.is_finite() || bid <= 0.0 || ask <= bid {
            return 0;
        }
        let mut fill_count = 0usize;

        for ord in osm.slots.iter_mut() {
            if ord.symbol_id != symbol_id
                || !matches!(ord.state, ExecOrderState::Open | ExecOrderState::PartiallyFilled)
            {
                continue;
            }
            let touched = match ord.direction {
                TradeDirection::Long => ask <= ord.limit_price,
                TradeDirection::Short => bid >= ord.limit_price,
                TradeDirection::Flat => false,
            };
            if !touched {
                continue;
            }

            let fill_qty = ord.remaining_qty();
            let fill_px = ord.limit_price;
            let fee_usd = (fill_qty * fill_px) * (self.cfg.maker_fee_bps.max(0.0) / 10_000.0);
            if ord.apply_fill(fill_qty, fill_px, fee_usd, now_ms) && fill_count < MAX_TRACKED_ORDERS {
                out_fills[fill_count] = Some(FillEvent {
                    client_order_id: ord.client_order_id,
                    symbol_id: ord.symbol_id,
                    direction: ord.direction,
                    fill_qty,
                    fill_price: fill_px,
                    fee_usd,
                    is_maker: true,
                    is_hard_sl: false,
                    is_take_profit: false,
                    timestamp_ms: now_ms,
                });
                fill_count += 1;
            }
        }
        fill_count
    }
}

/// Per-symbol Exchange-Side Hard Stop-Loss & Take-Profit Bracket
#[derive(Debug, Clone, Copy)]
pub struct StopBracket {
    pub symbol_id: SymbolId,
    pub position_dir: TradeDirection,
    pub qty: f64,
    pub sl_price: f64,
    pub tp_price: f64,
    pub active: bool,
}

impl StopBracket {
    pub const fn inactive(symbol_id: SymbolId) -> Self {
        Self {
            symbol_id,
            position_dir: TradeDirection::Flat,
            qty: 0.0,
            sl_price: 0.0,
            tp_price: 0.0,
            active: false,
        }
    }

    pub fn arm(&mut self, dir: TradeDirection, qty: f64, sl_price: f64, tp_price: f64) {
        if dir == TradeDirection::Flat
            || !qty.is_finite()
            || qty <= 0.0
            || !sl_price.is_finite()
            || sl_price <= 0.0
        {
            self.disarm();
            return;
        }
        self.position_dir = dir;
        self.qty = qty;
        self.sl_price = sl_price;
        self.tp_price = if tp_price.is_finite() && tp_price > 0.0 { tp_price } else { 0.0 };
        self.active = true;
    }

    #[inline]
    pub fn disarm(&mut self) {
        self.position_dir = TradeDirection::Flat;
        self.qty = 0.0;
        self.sl_price = 0.0;
        self.tp_price = 0.0;
        self.active = false;
    }

    /// Checks whether (bid, ask) breaches Hard SL or TP; returns Some((is_sl, is_tp, exit_price))
    #[inline]
    pub fn check_trigger(&self, bid: f64, ask: f64) -> Option<(bool, bool, f64)> {
        if !self.active || !bid.is_finite() || !ask.is_finite() || bid <= 0.0 || ask <= 0.0 {
            return None;
        }
        match self.position_dir {
            TradeDirection::Long => {
                if bid <= self.sl_price {
                    Some((true, false, bid))
                } else if self.tp_price > 0.0 && bid >= self.tp_price {
                    Some((false, true, bid))
                } else {
                    None
                }
            }
            TradeDirection::Short => {
                if ask >= self.sl_price {
                    Some((true, false, ask))
                } else if self.tp_price > 0.0 && ask <= self.tp_price {
                    Some((false, true, ask))
                } else {
                    None
                }
            }
            TradeDirection::Flat => None,
        }
    }
}

/// Non-blocking wire command emitted when ExecutionMode::Live is active
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LiveCommandKind {
    PlaceLimitPostOnly,
    PlaceMarketIoc,
    PlaceHardStop,
    CancelOrder,
    CancelAllSymbol,
    CancelAllGlobal,
}

#[derive(Debug, Clone, Copy)]
pub struct LiveDispatchCommand {
    pub kind: LiveCommandKind,
    pub client_order_id: u64,
    pub symbol_id: SymbolId,
    pub direction: TradeDirection,
    pub price: f64,
    pub qty: f64,
    pub stop_loss_price: f64,
    pub reduce_only: bool,
    pub created_ms: u64,
}
