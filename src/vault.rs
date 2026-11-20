#![allow(dead_code)]

use crate::execution::FillEvent;
use crate::sniper_strategy::{InventoryState, TradeDirection};
use crate::types::SymbolId;

pub const MAX_VAULT_SYMBOLS: usize = 8;
pub const MAX_JOURNAL_ENTRIES: usize = 128;

/// Stack-only live position ledger for a single symbol (BTC / ETH / SOL)
#[derive(Debug, Clone, Copy)]
pub struct PositionRecord {
    pub symbol_id: SymbolId,
    pub net_qty: f64, // Positive = Long, Negative = Short
    pub avg_entry_price: f64,
    pub last_mark_price: f64,
    pub realized_pnl_usd: f64,
    pub fees_paid_usd: f64,
    pub updated_ms: u64,
}

impl PositionRecord {
    pub const fn empty(symbol_id: SymbolId) -> Self {
        Self {
            symbol_id,
            net_qty: 0.0,
            avg_entry_price: 0.0,
            last_mark_price: 0.0,
            realized_pnl_usd: 0.0,
            fees_paid_usd: 0.0,
            updated_ms: 0,
        }
    }

    #[inline]
    pub fn is_open(&self) -> bool {
        self.net_qty.is_finite() && self.net_qty.abs() > 1e-12
    }

    #[inline]
    pub fn signed_notional_usd(&self) -> f64 {
        let px = if self.last_mark_price.is_finite() && self.last_mark_price > 0.0 {
            self.last_mark_price
        } else {
            self.avg_entry_price
        };
        if !self.net_qty.is_finite() || !px.is_finite() || px <= 0.0 {
            return 0.0;
        }
        self.net_qty * px
    }

    #[inline]
    pub fn unrealized_pnl_usd(&self) -> f64 {
        if !self.is_open()
            || !self.avg_entry_price.is_finite()
            || !self.last_mark_price.is_finite()
            || self.avg_entry_price <= 0.0
            || self.last_mark_price <= 0.0
        {
            return 0.0;
        }
        (self.last_mark_price - self.avg_entry_price) * self.net_qty
    }

    #[inline]
    pub fn unrealized_bps(&self) -> f64 {
        let entry_notional = self.net_qty.abs() * self.avg_entry_price;
        if !self.is_open() || !entry_notional.is_finite() || entry_notional <= 1e-9 {
            return 0.0;
        }
        (self.unrealized_pnl_usd() / entry_notional) * 10_000.0
    }

    #[inline]
    pub fn to_inventory_state(&self, has_open_order: bool) -> InventoryState {
        InventoryState {
            net_qty: self.net_qty,
            avg_entry_price: self.avg_entry_price,
            notional_usd: self.signed_notional_usd(),
            unrealized_bps: self.unrealized_bps(),
            has_open_order,
        }
    }
}

/// Immutable audit record stored in the fixed-size stack trade journal
#[derive(Debug, Clone, Copy)]
pub struct TradeJournalEntry {
    pub seq: u64,
    pub client_order_id: u64,
    pub symbol_id: SymbolId,
    pub direction: TradeDirection,
    pub fill_qty: f64,
    pub fill_price: f64,
    pub fee_usd: f64,
    pub realized_pnl_usd: f64,
    pub post_fee_pnl_bps: f64,
    pub is_maker: bool,
    pub is_hard_sl: bool,
    pub is_take_profit: bool,
    pub timestamp_ms: u64,
}

impl TradeJournalEntry {
    pub const fn empty() -> Self {
        Self {
            seq: 0,
            client_order_id: 0,
            symbol_id: SymbolId(0),
            direction: TradeDirection::Flat,
            fill_qty: 0.0,
            fill_price: 0.0,
            fee_usd: 0.0,
            realized_pnl_usd: 0.0,
            post_fee_pnl_bps: 0.0,
            is_maker: false,
            is_hard_sl: false,
            is_take_profit: false,
            timestamp_ms: 0,
        }
    }
}

/// Immediate post-fill summary fed back to AiRouter RL Controller & Telegram HUD
#[derive(Debug, Clone, Copy)]
pub struct FillOutcome {
    pub closed_position: bool,
    pub realized_pnl_usd: f64,
    pub post_fee_pnl_bps: f64,
    pub daily_pnl_pct: f64,
    pub is_hard_sl: bool,
    pub is_take_profit: bool,
}

impl FillOutcome {
    pub const fn none() -> Self {
        Self {
            closed_position: false,
            realized_pnl_usd: 0.0,
            post_fee_pnl_bps: 0.0,
            daily_pnl_pct: 0.0,
            is_hard_sl: false,
            is_take_profit: false,
        }
    }
}

/// Zero-allocation Capital Ledger & Ring-Buffer Trade Journal (Zero-Panic)
#[derive(Debug, Clone, Copy)]
pub struct VaultLedger {
    pub cash_balance_usd: f64,
    pub day_start_equity_usd: f64,
    pub peak_equity_usd: f64,
    pub total_realized_pnl_usd: f64,
    pub total_fees_usd: f64,
    pub total_closed_trades: u64,
    pub winning_trades: u64,
    pub positions: [PositionRecord; MAX_VAULT_SYMBOLS],
    journal: [TradeJournalEntry; MAX_JOURNAL_ENTRIES],
    journal_head: usize,
    journal_count: usize,
    next_seq: u64,
}

impl VaultLedger {
    pub const fn new(initial_capital_usd: f64) -> Self {
        let cap = if initial_capital_usd > 0.0 {
            initial_capital_usd
        } else {
            0.0
        };
        Self {
            cash_balance_usd: cap,
            day_start_equity_usd: cap,
            peak_equity_usd: cap,
            total_realized_pnl_usd: 0.0,
            total_fees_usd: 0.0,
            total_closed_trades: 0,
            winning_trades: 0,
            positions: [
                PositionRecord::empty(SymbolId(0)),
                PositionRecord::empty(SymbolId(1)),
                PositionRecord::empty(SymbolId(2)),
                PositionRecord::empty(SymbolId(3)),
                PositionRecord::empty(SymbolId(4)),
                PositionRecord::empty(SymbolId(5)),
                PositionRecord::empty(SymbolId(6)),
                PositionRecord::empty(SymbolId(7)),
            ],
            journal: [TradeJournalEntry::empty(); MAX_JOURNAL_ENTRIES],
            journal_head: 0,
            journal_count: 0,
            next_seq: 1,
        }
    }

    #[inline]
    fn sym_idx(symbol_id: SymbolId) -> Option<usize> {
        let idx = symbol_id.0 as usize;
        if idx < MAX_VAULT_SYMBOLS { Some(idx) } else { None }
    }

    pub fn update_mark_price(&mut self, symbol_id: SymbolId, mark_price: f64, now_ms: u64) {
        if !mark_price.is_finite() || mark_price <= 0.0 {
            return;
        }
        if let Some(i) = Self::sym_idx(symbol_id) {
            self.positions[i].last_mark_price = mark_price;
            self.positions[i].updated_ms = now_ms;
            let eq = self.total_equity_usd();
            if eq > self.peak_equity_usd {
                self.peak_equity_usd = eq;
            }
        }
    }

    #[inline]
    pub fn total_unrealized_pnl_usd(&self) -> f64 {
        let mut sum = 0.0;
        for pos in self.positions.iter() {
            sum += pos.unrealized_pnl_usd();
        }
        sum
    }

    #[inline]
    pub fn total_gross_exposure_usd(&self) -> f64 {
        let mut sum = 0.0;
        for pos in self.positions.iter() {
            sum += pos.signed_notional_usd().abs();
        }
        sum
    }

    #[inline]
    pub fn total_equity_usd(&self) -> f64 {
        (self.cash_balance_usd + self.total_unrealized_pnl_usd()).max(0.0)
    }

    #[inline]
    pub fn daily_pnl_pct(&self) -> f64 {
        if !self.day_start_equity_usd.is_finite() || self.day_start_equity_usd <= 1e-9 {
            return 0.0;
        }
        ((self.total_equity_usd() - self.day_start_equity_usd) / self.day_start_equity_usd) * 100.0
    }

    #[inline]
    pub fn position(&self, symbol_id: SymbolId) -> Option<&PositionRecord> {
        Self::sym_idx(symbol_id).map(|i| &self.positions[i])
    }

    #[inline]
    pub fn inventory_state(&self, symbol_id: SymbolId, has_open_order: bool) -> InventoryState {
        match self.position(symbol_id) {
            Some(p) => p.to_inventory_state(has_open_order),
            None => InventoryState::flat(),
        }
    }

    /// Applies a realized FillEvent (open, add, partial/full close, or flip) and logs journal entry
    pub fn apply_fill(&mut self, fill: &FillEvent) -> FillOutcome {
        if !fill.fill_qty.is_finite()
            || !fill.fill_price.is_finite()
            || !fill.fee_usd.is_finite()
            || fill.fill_qty <= 0.0
            || fill.fill_price <= 0.0
        {
            return FillOutcome::none();
        }

        let signed_qty = match fill.direction {
            TradeDirection::Long => fill.fill_qty,
            TradeDirection::Short => -fill.fill_qty,
            TradeDirection::Flat => return FillOutcome::none(),
        };

        let idx = match Self::sym_idx(fill.symbol_id) {
            Some(i) => i,
            None => return FillOutcome::none(),
        };

        let fee = fill.fee_usd.max(0.0);
        let prev_qty = self.positions[idx].net_qty;
        let prev_entry = self.positions[idx].avg_entry_price;

        let mut gross_pnl = 0.0;
        let mut closed_qty = 0.0;

        if prev_qty.abs() <= 1e-12 || prev_qty.signum() == signed_qty.signum() {
            // Opening or adding to same direction
            let new_qty = prev_qty + signed_qty;
            let new_avg = if new_qty.abs() > 1e-12 {
                ((prev_qty.abs() * prev_entry) + (fill.fill_qty * fill.fill_price)) / new_qty.abs()
            } else {
                0.0
            };
            self.positions[idx].net_qty = new_qty;
            self.positions[idx].avg_entry_price = new_avg;
        } else {
            // Reducing, closing, or flipping position
            closed_qty = prev_qty.abs().min(fill.fill_qty);
            gross_pnl = if prev_qty > 0.0 {
                (fill.fill_price - prev_entry) * closed_qty
            } else {
                (prev_entry - fill.fill_price) * closed_qty
            };

            let new_qty = prev_qty + signed_qty;
            if new_qty.abs() <= 1e-12 {
                self.positions[idx].net_qty = 0.0;
                self.positions[idx].avg_entry_price = 0.0;
            } else if new_qty.signum() == prev_qty.signum() {
                self.positions[idx].net_qty = new_qty;
            } else {
                self.positions[idx].net_qty = new_qty;
                self.positions[idx].avg_entry_price = fill.fill_price;
            }
        }

        let net_realized = gross_pnl - fee;
        self.positions[idx].last_mark_price = fill.fill_price;
        self.positions[idx].realized_pnl_usd += gross_pnl;
        self.positions[idx].fees_paid_usd += fee;
        self.positions[idx].updated_ms = fill.timestamp_ms;

        self.cash_balance_usd += net_realized;
        self.total_realized_pnl_usd += gross_pnl;
        self.total_fees_usd += fee;

        let post_fee_bps = if closed_qty > 1e-12 && prev_entry > 0.0 {
            (net_realized / (closed_qty * prev_entry)) * 10_000.0
        } else {
            0.0
        };

        if closed_qty > 1e-12 {
            self.total_closed_trades = self.total_closed_trades.saturating_add(1);
            if net_realized > 0.0 {
                self.winning_trades = self.winning_trades.saturating_add(1);
            }
        }

        let eq = self.total_equity_usd();
        if eq > self.peak_equity_usd {
            self.peak_equity_usd = eq;
        }

        let entry = TradeJournalEntry {
            seq: self.next_seq,
            client_order_id: fill.client_order_id,
            symbol_id: fill.symbol_id,
            direction: fill.direction,
            fill_qty: fill.fill_qty,
            fill_price: fill.fill_price,
            fee_usd: fee,
            realized_pnl_usd: gross_pnl,
            post_fee_pnl_bps: post_fee_bps,
            is_maker: fill.is_maker,
            is_hard_sl: fill.is_hard_sl,
            is_take_profit: fill.is_take_profit,
            timestamp_ms: fill.timestamp_ms,
        };
        self.next_seq = self.next_seq.saturating_add(1);
        self.journal[self.journal_head] = entry;
        self.journal_head = (self.journal_head + 1) % MAX_JOURNAL_ENTRIES;
        if self.journal_count < MAX_JOURNAL_ENTRIES {
            self.journal_count += 1;
        }

        FillOutcome {
            closed_position: closed_qty > 1e-12,
            realized_pnl_usd: net_realized,
            post_fee_pnl_bps: post_fee_bps,
            daily_pnl_pct: self.daily_pnl_pct(),
            is_hard_sl: fill.is_hard_sl,
            is_take_profit: fill.is_take_profit,
        }
    }

    pub fn on_utc_day_rollover(&mut self) {
        let eq = self.total_equity_usd();
        if eq.is_finite() && eq > 0.0 {
            self.day_start_equity_usd = eq;
        }
    }

    #[inline]
    pub fn journal_len(&self) -> usize {
        self.journal_count
    }

    #[inline]
    pub fn latest_journal_entry(&self) -> Option<&TradeJournalEntry> {
        if self.journal_count == 0 {
            return None;
        }
        let idx = if self.journal_head == 0 {
            MAX_JOURNAL_ENTRIES - 1
        } else {
            self.journal_head - 1
        };
        Some(&self.journal[idx])
    }
}
