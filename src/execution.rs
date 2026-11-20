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
