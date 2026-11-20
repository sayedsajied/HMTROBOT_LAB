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
use crate::risk::RiskVerdict;
use crate::sniper_strategy::StrategyIntent;

pub const MAX_EXEC_SYMBOLS: usize = 8;
pub const MAX_STEP_EVENTS: usize = 8;

/// Zero-allocation output returned by ExecutionEngine on every order or market tick
#[derive(Debug, Clone, Copy)]
pub struct ExecutionOutput {
    pub fills: [Option<FillEvent>; MAX_STEP_EVENTS],
    pub fill_count: usize,
    pub live_cmds: [Option<LiveDispatchCommand>; MAX_STEP_EVENTS],
    pub cmd_count: usize,
    pub submitted_order_id: Option<u64>,
    pub cancelled_count: usize,
}

impl ExecutionOutput {
    pub const fn empty() -> Self {
        Self {
            fills: [None; MAX_STEP_EVENTS],
            fill_count: 0,
            live_cmds: [None; MAX_STEP_EVENTS],
            cmd_count: 0,
            submitted_order_id: None,
            cancelled_count: 0,
        }
    }

    #[inline]
    pub fn push_fill(&mut self, fill: FillEvent) {
        if self.fill_count < MAX_STEP_EVENTS {
            self.fills[self.fill_count] = Some(fill);
            self.fill_count += 1;
        }
    }

    #[inline]
    pub fn push_cmd(&mut self, cmd: LiveDispatchCommand) {
        if self.cmd_count < MAX_STEP_EVENTS {
            self.live_cmds[self.cmd_count] = Some(cmd);
            self.cmd_count += 1;
        }
    }
}

/// Master Execution Engine: 100% Paper-to-Live Parity, Deterministic OSM & Hard SL Brackets
#[derive(Debug, Clone, Copy)]
pub struct ExecutionEngine {
    pub mode: ExecutionMode,
    pub osm: OrderStateMachine,
    pub paper: PaperSimulator,
    brackets: [StopBracket; MAX_EXEC_SYMBOLS],
}

impl ExecutionEngine {
    pub const fn new(mode: ExecutionMode, paper_cfg: PaperSimConfig) -> Self {
        Self {
            mode,
            osm: OrderStateMachine::new(),
            paper: PaperSimulator::new(paper_cfg),
            brackets: [StopBracket::inactive(SymbolId(0)); MAX_EXEC_SYMBOLS],
        }
    }

    #[inline]
    fn bracket_index(symbol_id: SymbolId) -> Option<usize> {
        let idx = symbol_id.0 as usize;
        if idx < MAX_EXEC_SYMBOLS {
            Some(idx)
        } else {
            None
        }
    }

    #[inline]
    pub fn bracket_for(&self, symbol_id: SymbolId) -> Option<&StopBracket> {
        Self::bracket_index(symbol_id).map(|i| &self.brackets[i])
    }

    fn arm_bracket_from_order(&mut self, ord: &ExecOrder, filled_qty: f64) {
        if ord.is_reduce_only || ord.stop_loss_price <= 0.0 {
            if let Some(i) = Self::bracket_index(ord.symbol_id) {
                if ord.is_reduce_only {
                    self.brackets[i].disarm();
                }
            }
            return;
        }
        if let Some(i) = Self::bracket_index(ord.symbol_id) {
            self.brackets[i].symbol_id = ord.symbol_id;
            self.brackets[i].arm(ord.direction, filled_qty, ord.stop_loss_price, ord.take_profit_price);
        }
    }

    /// Executes a risk-approved StrategyIntent (identical pipeline for Paper & Live)
    pub fn execute_intent(
        &mut self,
        intent: &StrategyIntent,
        verdict: &RiskVerdict,
        bid: f64,
        ask: f64,
        is_reduce_only: bool,
        now_ms: u64,
    ) -> ExecutionOutput {
        let mut out = ExecutionOutput::empty();
        let sym = intent.symbol_id;

        // 1. Honor Cancel-All requests (Toxic-Flow Shield / Safe-Mode / Risk Breakers)
        if verdict.require_cancel_all || intent.cancel_open_orders {
            let cancelled = self.osm.cancel_all_for_symbol(sym, now_ms);
            out.cancelled_count = cancelled;
            if self.mode == ExecutionMode::Live && (cancelled > 0 || verdict.require_cancel_all) {
                out.push_cmd(LiveDispatchCommand {
                    kind: LiveCommandKind::CancelAllSymbol,
                    client_order_id: 0,
                    symbol_id: sym,
                    direction: TradeDirection::Flat,
                    price: 0.0,
                    qty: 0.0,
                    stop_loss_price: 0.0,
                    reduce_only: false,
                    created_ms: now_ms,
                });
            }
        }

        if !verdict.approved || verdict.approved_qty <= 1e-12 {
            return out;
        }

        // 2. Register deterministic Submitting order in OrderStateMachine
        let cid = match self.osm.create_submitting(
            sym,
            intent.direction,
            intent.style,
            intent.limit_price,
            verdict.approved_qty,
            intent.stop_loss_price,
            intent.take_profit_price,
            is_reduce_only,
            now_ms,
        ) {
            Some(id) => id,
            None => return out,
        };
        out.submitted_order_id = Some(cid);

        // 3. Sink split: Paper Simulator vs Live Exchange Command Queue
        match self.mode {
            ExecutionMode::Paper => {
                let mut armed_snap: Option<(ExecOrder, f64)> = None;
                if let Some(ord) = self.osm.get_mut(cid) {
                    if let Some(fill) = self.paper.submit_order(ord, bid, ask, now_ms) {
                        out.push_fill(fill);
                        armed_snap = Some((*ord, fill.fill_qty));
                    }
                }
                if let Some((ord_snap, fqty)) = armed_snap {
                    self.arm_bracket_from_order(&ord_snap, fqty);
                }
            }
            ExecutionMode::Live => {
                let kind = match intent.style {
                    ExecutionStyle::MakerPostOnly => LiveCommandKind::PlaceLimitPostOnly,
                    ExecutionStyle::TakerAggressive | ExecutionStyle::EmergencyClose => {
                        LiveCommandKind::PlaceMarketIoc
                    }
                    ExecutionStyle::None => return out,
                };
                out.push_cmd(LiveDispatchCommand {
                    kind,
                    client_order_id: cid,
                    symbol_id: sym,
                    direction: intent.direction,
                    price: intent.limit_price,
                    qty: verdict.approved_qty,
                    stop_loss_price: intent.stop_loss_price,
                    reduce_only: is_reduce_only,
                    created_ms: now_ms,
                });

                // Attach mandatory exchange-side Hard Stop-Loss on risk-increasing entries
                if !is_reduce_only && intent.stop_loss_price.is_finite() && intent.stop_loss_price > 0.0 {
                    let sl_dir = match intent.direction {
                        TradeDirection::Long => TradeDirection::Short,
                        TradeDirection::Short => TradeDirection::Long,
                        TradeDirection::Flat => TradeDirection::Flat,
                    };
                    out.push_cmd(LiveDispatchCommand {
                        kind: LiveCommandKind::PlaceHardStop,
                        client_order_id: cid,
                        symbol_id: sym,
                        direction: sl_dir,
                        price: intent.stop_loss_price,
                        qty: verdict.approved_qty,
                        stop_loss_price: intent.stop_loss_price,
                        reduce_only: true,
                        created_ms: now_ms,
                    });
                }
            }
        }

        out
    }

    /// Evaluates resting Paper orders and active Hard SL/TP brackets on each top-of-book quote
    pub fn on_market_tick(
        &mut self,
        symbol_id: SymbolId,
        bid: f64,
        ask: f64,
        now_ms: u64,
    ) -> ExecutionOutput {
        let mut out = ExecutionOutput::empty();
        if self.mode != ExecutionMode::Paper || !bid.is_finite() || !ask.is_finite() || bid <= 0.0 || ask <= bid {
            return out;
        }

        // 1. Check active Hard SL / TP bracket first
        if let Some(idx) = Self::bracket_index(symbol_id) {
            let br = self.brackets[idx];
            if let Some((is_sl, is_tp, exit_px)) = br.check_trigger(bid, ask) {
                let close_dir = match br.position_dir {
                    TradeDirection::Long => TradeDirection::Short,
                    TradeDirection::Short => TradeDirection::Long,
                    TradeDirection::Flat => TradeDirection::Flat,
                };
                if close_dir != TradeDirection::Flat && br.qty > 1e-12 {
                    let slip_bps = if is_sl {
                        self.paper.cfg.emergency_slippage_bps.max(0.0)
                    } else {
                        self.paper.cfg.taker_slippage_bps.max(0.0)
                    };
                    let fill_px = match close_dir {
                        TradeDirection::Long => exit_px * (1.0 + (slip_bps / 10_000.0)),
                        TradeDirection::Short => exit_px * (1.0 - (slip_bps / 10_000.0)),
                        TradeDirection::Flat => exit_px,
                    };
                    let fee_usd = (br.qty * fill_px) * (self.paper.cfg.taker_fee_bps.max(0.0) / 10_000.0);
                    out.push_fill(FillEvent {
                        client_order_id: 0,
                        symbol_id,
                        direction: close_dir,
                        fill_qty: br.qty,
                        fill_price: fill_px,
                        fee_usd,
                        is_maker: false,
                        is_hard_sl: is_sl,
                        is_take_profit: is_tp,
                        timestamp_ms: now_ms,
                    });
                    self.brackets[idx].disarm();
                }
            }
        }

        // 2. Check resting Post-Only limit orders for maker fills
        let mut buf: [Option<FillEvent>; MAX_TRACKED_ORDERS] = [None; MAX_TRACKED_ORDERS];
        let n = self.paper.on_market_quotes(&mut self.osm, symbol_id, bid, ask, now_ms, &mut buf);
        for i in 0..n {
            if let Some(fill) = buf[i] {
                out.push_fill(fill);
                if let Some(ord_snap) = self.osm.get(fill.client_order_id).copied() {
                    self.arm_bracket_from_order(&ord_snap, fill.fill_qty);
                }
            }
        }

        out
    }

    /// Live exchange callback: order accepted and resting on book
    pub fn on_exchange_ack(&mut self, client_order_id: u64, exchange_order_id: u64, now_ms: u64) -> bool {
        if let Some(ord) = self.osm.get_mut(client_order_id) {
            ord.exchange_order_id = exchange_order_id;
            return ord.transition(ExecOrderState::Open, now_ms);
        }
        false
    }

    /// Live exchange callback: incremental or full fill received from private WS/REST
    pub fn on_exchange_fill(
        &mut self,
        client_order_id: u64,
        fill_qty: f64,
        fill_price: f64,
        fee_usd: f64,
        is_maker: bool,
        now_ms: u64,
    ) -> Option<FillEvent> {
        let mut snap: Option<(ExecOrder, FillEvent)> = None;
        if let Some(ord) = self.osm.get_mut(client_order_id) {
            if ord.apply_fill(fill_qty, fill_price, fee_usd, now_ms) {
                let ev = FillEvent {
                    client_order_id: ord.client_order_id,
                    symbol_id: ord.symbol_id,
                    direction: ord.direction,
                    fill_qty,
                    fill_price,
                    fee_usd: fee_usd.max(0.0),
                    is_maker,
                    is_hard_sl: false,
                    is_take_profit: false,
                    timestamp_ms: now_ms,
                };
                snap = Some((*ord, ev));
            }
        }
        if let Some((ord_snap, ev)) = snap {
            self.arm_bracket_from_order(&ord_snap, ev.fill_qty);
            return Some(ev);
        }
        None
    }

    /// Live exchange callback: order cancelled or rejected
    pub fn on_exchange_cancel_or_reject(&mut self, client_order_id: u64, failed: bool, now_ms: u64) -> bool {
        if let Some(ord) = self.osm.get_mut(client_order_id) {
            let next = if failed {
                ExecOrderState::Failed
            } else {
                ExecOrderState::Cancelled
            };
            return ord.transition(next, now_ms);
        }
        false
    }

    /// Global emergency Cancel-All across BTC, ETH, SOL (Dead-Man's Switch / Kill-Switch)
    pub fn cancel_all_global(&mut self, now_ms: u64) -> ExecutionOutput {
        let mut out = ExecutionOutput::empty();
        out.cancelled_count = self.osm.cancel_all(now_ms);
        if self.mode == ExecutionMode::Live {
            out.push_cmd(LiveDispatchCommand {
                kind: LiveCommandKind::CancelAllGlobal,
                client_order_id: 0,
                symbol_id: SymbolId(0),
                direction: TradeDirection::Flat,
                price: 0.0,
                qty: 0.0,
                stop_loss_price: 0.0,
                reduce_only: false,
                created_ms: now_ms,
            });
        }
        out
    }
}

#[cfg(test)]
mod execution_tests {
    use super::*;
    use crate::sniper_strategy::BlockReason;

    fn sample_long_intent(limit_px: f64, sl_px: f64, style: ExecutionStyle) -> StrategyIntent {
        StrategyIntent {
            symbol_id: SymbolId(0),
            direction: TradeDirection::Long,
            style,
            block_reason: BlockReason::None,
            limit_price: limit_px,
            stop_loss_price: sl_px,
            take_profit_price: limit_px + 200.0,
            size_fraction: 0.25,
            expected_edge_bps: 15.0,
            fee_hurdle_bps: 6.5,
            toxic_score: 0.1,
            cancel_open_orders: false,
        }
    }

    #[test]
    fn test_order_state_machine_rejects_illegal_transitions() {
        let mut ord = ExecOrder::empty();
        assert!(!ord.transition(ExecOrderState::Filled, 100)); // Idle -> Filled illegal
        assert!(ord.transition(ExecOrderState::Submitting, 100));
        assert!(ord.transition(ExecOrderState::Open, 110));
        assert!(ord.transition(ExecOrderState::Cancelled, 120));
        assert!(!ord.transition(ExecOrderState::Filled, 130)); // Terminal -> Filled illegal
    }

    #[test]
    fn test_paper_post_only_fill_and_hard_sl_trigger() {
        let mut exec = ExecutionEngine::new(ExecutionMode::Paper, PaperSimConfig::default_crypto());
        let intent = sample_long_intent(90_000.0, 89_500.0, ExecutionStyle::MakerPostOnly);
        let verdict = RiskVerdict::approved(SymbolId(0), 900.0, 0.01, false);

        // Resting passive bid below ask -> Open, no immediate fill
        let res = exec.execute_intent(&intent, &verdict, 90_000.0, 90_010.0, false, 1_000);
        assert_eq!(res.fill_count, 0);
        let cid = res.submitted_order_id.unwrap_or(0);
        assert_eq!(exec.osm.get(cid).map(|o| o.state), Some(ExecOrderState::Open));

        // Price drops to touch 90_000 ask -> Maker fill + Hard SL bracket armed
        let tick_fill = exec.on_market_tick(SymbolId(0), 89_995.0, 90_000.0, 2_000);
        assert_eq!(tick_fill.fill_count, 1);
        assert_eq!(tick_fill.fills[0].map(|f| f.is_maker), Some(true));
        assert_eq!(exec.bracket_for(SymbolId(0)).map(|b| b.active), Some(true));

        // Price breaches 89_500 SL -> Hard SL triggers and disarms bracket
        let sl_tick = exec.on_market_tick(SymbolId(0), 89_490.0, 89_500.0, 3_000);
        assert_eq!(sl_tick.fill_count, 1);
        assert_eq!(sl_tick.fills[0].map(|f| f.is_hard_sl), Some(true));
        assert_eq!(exec.bracket_for(SymbolId(0)).map(|b| b.active), Some(false));
    }

    #[test]
    fn test_live_mode_dispatches_post_only_and_hard_sl_commands() {
        let mut exec = ExecutionEngine::new(ExecutionMode::Live, PaperSimConfig::default_crypto());
        let intent = sample_long_intent(3_000.0, 2_970.0, ExecutionStyle::MakerPostOnly);
        let verdict = RiskVerdict::approved(SymbolId(1), 300.0, 0.1, false);

        let out = exec.execute_intent(&intent, &verdict, 3_000.0, 3_001.0, false, 5_000);
        assert_eq!(out.cmd_count, 2);
        assert_eq!(out.live_cmds[0].map(|c| c.kind), Some(LiveCommandKind::PlaceLimitPostOnly));
        assert_eq!(out.live_cmds[1].map(|c| c.kind), Some(LiveCommandKind::PlaceHardStop));

        let cid = out.submitted_order_id.unwrap_or(0);
        assert!(exec.on_exchange_ack(cid, 99_001, 5_050));
        let fill = exec.on_exchange_fill(cid, 0.1, 3_000.0, 0.06, true, 5_200);
        assert!(fill.is_some());
        assert_eq!(exec.osm.get(cid).map(|o| o.state), Some(ExecOrderState::Filled));
    }
}
