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
use crate::risk::AccountRiskSnapshot;

pub const DEFAULT_SYNC_INTERVAL_MS: u64 = 30_000;

impl VaultLedger {
    /// Builds a zero-allocation AccountRiskSnapshot directly for RiskEngine
    #[inline]
    pub fn to_risk_snapshot(&self, open_orders_count: u32) -> AccountRiskSnapshot {
        AccountRiskSnapshot {
            equity_usd: self.total_equity_usd(),
            day_start_equity_usd: self.day_start_equity_usd,
            peak_equity_usd: self.peak_equity_usd.max(self.total_equity_usd()),
            total_gross_exposure_usd: self.total_gross_exposure_usd(),
            open_orders_count,
        }
    }
}

/// Authoritative position snapshot fetched from Live Exchange API (Single Source of Truth)
#[derive(Debug, Clone, Copy)]
pub struct ExchangePositionTruth {
    pub symbol_id: SymbolId,
    pub net_qty: f64,
    pub avg_entry_price: f64,
    pub mark_price: f64,
    pub is_present: bool,
}

impl ExchangePositionTruth {
    pub const fn empty() -> Self {
        Self {
            symbol_id: SymbolId(0),
            net_qty: 0.0,
            avg_entry_price: 0.0,
            mark_price: 0.0,
            is_present: false,
        }
    }
}

/// Authoritative account & positions snapshot returned by periodic 30s Exchange Sync
#[derive(Debug, Clone, Copy)]
pub struct ExchangeAccountTruth {
    pub wallet_balance_usd: f64,
    pub positions: [ExchangePositionTruth; MAX_VAULT_SYMBOLS],
    pub position_count: usize,
    pub exchange_timestamp_ms: u64,
}

impl ExchangeAccountTruth {
    pub const fn empty(wallet_balance_usd: f64, exchange_timestamp_ms: u64) -> Self {
        Self {
            wallet_balance_usd,
            positions: [ExchangePositionTruth::empty(); MAX_VAULT_SYMBOLS],
            position_count: 0,
            exchange_timestamp_ms,
        }
    }

    pub fn push_position(
        &mut self,
        symbol_id: SymbolId,
        net_qty: f64,
        avg_entry_price: f64,
        mark_price: f64,
    ) -> bool {
        if self.position_count >= MAX_VAULT_SYMBOLS
            || !net_qty.is_finite()
            || !avg_entry_price.is_finite()
            || !mark_price.is_finite()
        {
            return false;
        }
        self.positions[self.position_count] = ExchangePositionTruth {
            symbol_id,
            net_qty,
            avg_entry_price: avg_entry_price.max(0.0),
            mark_price: mark_price.max(0.0),
            is_present: true,
        };
        self.position_count += 1;
        true
    }
}

/// Audit summary produced after reconciling local VaultLedger against Exchange truth
#[derive(Debug, Clone, Copy)]
pub struct ReconciliationReport {
    pub balance_drift_usd: f64,
    pub max_qty_drift: f64,
    pub positions_corrected: usize,
    pub severe_drift_detected: bool,
    pub synced_at_ms: u64,
}

impl ReconciliationReport {
    pub const fn clean(synced_at_ms: u64) -> Self {
        Self {
            balance_drift_usd: 0.0,
            max_qty_drift: 0.0,
            positions_corrected: 0,
            severe_drift_detected: false,
            synced_at_ms,
        }
    }
}

/// Wintermute Rule #11: Periodic 30s Exchange-vs-Vault Reconciler (Zero-Heap, Zero-Panic)
#[derive(Debug, Clone, Copy)]
pub struct ExchangeReconciler {
    pub sync_interval_ms: u64,
    pub last_sync_ms: u64,
    pub balance_drift_alert_usd: f64,
    pub qty_drift_tol: f64,
    pub total_syncs: u64,
    pub total_corrections: u64,
}

impl ExchangeReconciler {
    pub const fn new(
        sync_interval_ms: u64,
        balance_drift_alert_usd: f64,
        qty_drift_tol: f64,
    ) -> Self {
        Self {
            sync_interval_ms,
            last_sync_ms: 0,
            balance_drift_alert_usd,
            qty_drift_tol,
            total_syncs: 0,
            total_corrections: 0,
        }
    }

    pub const fn default_30s() -> Self {
        Self::new(DEFAULT_SYNC_INTERVAL_MS, 2.0, 1e-6)
    }

    /// True when >= 30s have elapsed since last authoritative Exchange sync
    #[inline]
    pub fn is_sync_due(&self, now_ms: u64) -> bool {
        self.last_sync_ms == 0 || now_ms.saturating_sub(self.last_sync_ms) >= self.sync_interval_ms
    }

    /// Overwrites local VaultLedger drift using Exchange as Single Source of Truth
    pub fn reconcile(
        &mut self,
        vault: &mut VaultLedger,
        truth: &ExchangeAccountTruth,
        now_ms: u64,
    ) -> ReconciliationReport {
        let mut report = ReconciliationReport::clean(now_ms);
        if !truth.wallet_balance_usd.is_finite() || truth.wallet_balance_usd < 0.0 {
            report.severe_drift_detected = true;
            return report;
        }

        // 1. Reconcile cash/wallet balance
        let bal_drift = truth.wallet_balance_usd - vault.cash_balance_usd;
        report.balance_drift_usd = bal_drift;
        if bal_drift.abs() > 1e-6 {
            vault.cash_balance_usd = truth.wallet_balance_usd;
        }
        if bal_drift.abs() >= self.balance_drift_alert_usd.abs() {
            report.severe_drift_detected = true;
        }

        // 2. Track which symbols are reported open by Exchange
        let mut seen = [false; MAX_VAULT_SYMBOLS];
        let count = truth.position_count.min(MAX_VAULT_SYMBOLS);
        for i in 0..count {
            let ex_pos = truth.positions[i];
            if !ex_pos.is_present {
                continue;
            }
            let idx = ex_pos.symbol_id.0 as usize;
            if idx >= MAX_VAULT_SYMBOLS {
                continue;
            }
            seen[idx] = true;

            let local = &mut vault.positions[idx];
            let qty_drift = (ex_pos.net_qty - local.net_qty).abs();
            if qty_drift > report.max_qty_drift {
                report.max_qty_drift = qty_drift;
            }

            if qty_drift > self.qty_drift_tol.max(1e-12) {
                local.net_qty = ex_pos.net_qty;
                local.avg_entry_price = if ex_pos.net_qty.abs() > 1e-12 {
                    ex_pos.avg_entry_price
                } else {
                    0.0
                };
                report.positions_corrected += 1;
                report.severe_drift_detected = true;
            }
            if ex_pos.mark_price > 0.0 {
                local.last_mark_price = ex_pos.mark_price;
            }
            local.updated_ms = now_ms;
        }

        // 3. Clear phantom local positions that no longer exist on Exchange
        for idx in 0..MAX_VAULT_SYMBOLS {
            if !seen[idx] && vault.positions[idx].is_open() {
                let phantom_qty = vault.positions[idx].net_qty.abs();
                if phantom_qty > report.max_qty_drift {
                    report.max_qty_drift = phantom_qty;
                }
                vault.positions[idx].net_qty = 0.0;
                vault.positions[idx].avg_entry_price = 0.0;
                vault.positions[idx].updated_ms = now_ms;
                report.positions_corrected += 1;
                report.severe_drift_detected = true;
            }
        }

        // 4. Refresh peak equity watermark
        let eq = vault.total_equity_usd();
        if eq > vault.peak_equity_usd {
            vault.peak_equity_usd = eq;
        }

        self.last_sync_ms = now_ms;
        self.total_syncs = self.total_syncs.saturating_add(1);
        self.total_corrections = self
            .total_corrections
            .saturating_add(report.positions_corrected as u64);

        report
    }
}
use std::fmt::Write as FmtWrite;
use std::fs::{self, OpenOptions};
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};

impl VaultLedger {
    /// Serializes ledger & open positions into a deterministic, human-readable text snapshot
    pub fn serialize_snapshot(&self) -> String {
        let mut out = String::with_capacity(512);
        let _ = writeln!(out, "v=1");
        let _ = writeln!(out, "cash={:.8}", self.cash_balance_usd);
        let _ = writeln!(out, "day_start={:.8}", self.day_start_equity_usd);
        let _ = writeln!(out, "peak={:.8}", self.peak_equity_usd);
        let _ = writeln!(out, "realized={:.8}", self.total_realized_pnl_usd);
        let _ = writeln!(out, "fees={:.8}", self.total_fees_usd);
        let _ = writeln!(out, "closed={}", self.total_closed_trades);
        let _ = writeln!(out, "wins={}", self.winning_trades);
        let _ = writeln!(out, "next_seq={}", self.next_seq);

        for idx in 0..MAX_VAULT_SYMBOLS {
            let p = &self.positions[idx];
            if p.is_open() || p.realized_pnl_usd.abs() > 1e-12 || p.fees_paid_usd > 1e-12 {
                let _ = writeln!(
                    out,
                    "pos={},{:.10},{:.8},{:.8},{:.8},{:.8},{}",
                    idx,
                    p.net_qty,
                    p.avg_entry_price,
                    p.last_mark_price,
                    p.realized_pnl_usd,
                    p.fees_paid_usd,
                    p.updated_ms
                );
            }
        }
        out
    }

    /// Parses a persisted snapshot safely (ignores malformed/NaN lines, zero panic)
    pub fn deserialize_snapshot(raw: &str, fallback_capital_usd: f64) -> Self {
        let mut ledger = Self::new(fallback_capital_usd);
        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let (key, val) = match trimmed.split_once('=') {
                Some(pair) => pair,
                None => continue,
            };

            match key {
                "cash" => {
                    if let Ok(v) = val.parse::<f64>() {
                        if v.is_finite() && v >= 0.0 {
                            ledger.cash_balance_usd = v;
                        }
                    }
                }
                "day_start" => {
                    if let Ok(v) = val.parse::<f64>() {
                        if v.is_finite() && v > 0.0 {
                            ledger.day_start_equity_usd = v;
                        }
                    }
                }
                "peak" => {
                    if let Ok(v) = val.parse::<f64>() {
                        if v.is_finite() && v > 0.0 {
                            ledger.peak_equity_usd = v;
                        }
                    }
                }
                "realized" => {
                    if let Ok(v) = val.parse::<f64>() {
                        if v.is_finite() {
                            ledger.total_realized_pnl_usd = v;
                        }
                    }
                }
                "fees" => {
                    if let Ok(v) = val.parse::<f64>() {
                        if v.is_finite() && v >= 0.0 {
                            ledger.total_fees_usd = v;
                        }
                    }
                }
                "closed" => {
                    if let Ok(v) = val.parse::<u64>() {
                        ledger.total_closed_trades = v;
                    }
                }
                "wins" => {
                    if let Ok(v) = val.parse::<u64>() {
                        ledger.winning_trades = v;
                    }
                }
                "next_seq" => {
                    if let Ok(v) = val.parse::<u64>() {
                        if v > 0 {
                            ledger.next_seq = v;
                        }
                    }
                }
                "pos" => {
                    let mut parts = val.split(',');
                    let idx = match parts.next().and_then(|s| s.parse::<usize>().ok()) {
                        Some(i) if i < MAX_VAULT_SYMBOLS => i,
                        _ => continue,
                    };
                    let net_qty = parts.next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
                    let avg_entry = parts.next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
                    let last_mark = parts.next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
                    let realized = parts.next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
                    let fees = parts.next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
                    let updated_ms = parts.next().and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);

                    if net_qty.is_finite()
                        && avg_entry.is_finite()
                        && last_mark.is_finite()
                        && realized.is_finite()
                        && fees.is_finite()
                    {
                        ledger.positions[idx] = PositionRecord {
                            symbol_id: SymbolId(idx as u16),
                            net_qty,
                            avg_entry_price: avg_entry.max(0.0),
                            last_mark_price: last_mark.max(0.0),
                            realized_pnl_usd: realized,
                            fees_paid_usd: fees.max(0.0),
                            updated_ms,
                        };
                    }
                }
                _ => {}
            }
        }
        ledger
    }

    /// Atomic write (.tmp -> rename) so crash mid-write never corrupts persisted state
    pub fn save_atomic(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let mut tmp_os = path.as_os_str().to_os_string();
        tmp_os.push(".tmp");
        let tmp_path = PathBuf::from(tmp_os);

        let payload = self.serialize_snapshot();
        {
            let mut file = fs::File::create(&tmp_path)?;
            file.write_all(payload.as_bytes())?;
            file.sync_all()?;
        }
        fs::rename(&tmp_path, path)
    }

    /// Crash/Restart state recovery: loads persisted vault or falls back cleanly
    pub fn load_or_init(path: &Path, fallback_capital_usd: f64) -> Self {
        match fs::read_to_string(path) {
            Ok(raw) if !raw.trim().is_empty() => Self::deserialize_snapshot(&raw, fallback_capital_usd),
            _ => Self::new(fallback_capital_usd),
        }
    }

    /// Appends a single trade fill record to a durable CSV trade journal on disk
    pub fn append_journal_csv(path: &Path, entry: &TradeJournalEntry) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let write_header = !path.exists();
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        if write_header {
            writeln!(
                file,
                "seq,client_order_id,symbol_id,dir,fill_qty,fill_price,fee_usd,realized_pnl_usd,post_fee_pnl_bps,is_maker,is_hard_sl,is_tp,timestamp_ms"
            )?;
        }
        let dir_str = match entry.direction {
            TradeDirection::Long => "LONG",
            TradeDirection::Short => "SHORT",
            TradeDirection::Flat => "FLAT",
        };
        writeln!(
            file,
            "{},{},{},{},{:.10},{:.8},{:.6},{:.6},{:.4},{},{},{},{}",
            entry.seq,
            entry.client_order_id,
            entry.symbol_id.0,
            dir_str,
            entry.fill_qty,
            entry.fill_price,
            entry.fee_usd,
            entry.realized_pnl_usd,
            entry.post_fee_pnl_bps,
            entry.is_maker,
            entry.is_hard_sl,
            entry.is_take_profit,
            entry.timestamp_ms
        )
    }
}

#[cfg(test)]
mod vault_tests {
    use super::*;

    fn mk_fill(sym: u16, dir: TradeDirection, qty: f64, px: f64, fee: f64, ts: u64) -> FillEvent {
        FillEvent {
            client_order_id: ts,
            symbol_id: SymbolId(sym),
            direction: dir,
            fill_qty: qty,
            fill_price: px,
            fee_usd: fee,
            is_maker: true,
            is_hard_sl: false,
            is_take_profit: false,
            timestamp_ms: ts,
        }
    }

    #[test]
    fn test_vault_open_close_and_position_flip() {
        let mut vault = VaultLedger::new(1_000.0);

        // Open 0.01 BTC Long @ 90,000 (fee $0.18)
        let o1 = vault.apply_fill(&mk_fill(0, TradeDirection::Long, 0.01, 90_000.0, 0.18, 1_000));
        assert!(!o1.closed_position);
        assert!((vault.cash_balance_usd - 999.82).abs() < 1e-6);

        // Close 0.01 BTC Long @ 91,000 -> +$10 gross - $0.18 fee = +$9.82 net
        let o2 = vault.apply_fill(&mk_fill(0, TradeDirection::Short, 0.01, 91_000.0, 0.18, 2_000));
        assert!(o2.closed_position);
        assert!((o2.realized_pnl_usd - 9.82).abs() < 1e-6);
        assert_eq!(vault.total_closed_trades, 1);
        assert_eq!(vault.winning_trades, 1);
        assert!(!vault.positions[0].is_open());

        // Flip test: Open 1.0 SOL Long @ 150, then Sell 2.0 SOL @ 155 -> net -1.0 SOL Short @ 155
        vault.apply_fill(&mk_fill(2, TradeDirection::Long, 1.0, 150.0, 0.05, 3_000));
        let o3 = vault.apply_fill(&mk_fill(2, TradeDirection::Short, 2.0, 155.0, 0.10, 4_000));
        assert!(o3.closed_position);
        assert!((vault.positions[2].net_qty - (-1.0)).abs() < 1e-9);
        assert!((vault.positions[2].avg_entry_price - 155.0).abs() < 1e-6);
    }

    #[test]
    fn test_exchange_reconciler_corrects_drift_and_phantom() {
        let mut vault = VaultLedger::new(1_000.0);
        // Local thinks we hold 0.01 BTC and 1.0 ETH (ETH is a phantom)
        vault.apply_fill(&mk_fill(0, TradeDirection::Long, 0.01, 90_000.0, 0.0, 1_000));
        vault.apply_fill(&mk_fill(1, TradeDirection::Long, 1.0, 3_000.0, 0.0, 1_100));

        // Exchange truth: balance is 1015.0, BTC qty is 0.02, ETH is closed (not present)
        let mut truth = ExchangeAccountTruth::empty(1_015.0, 31_000);
        truth.push_position(SymbolId(0), 0.02, 90_200.0, 90_500.0);

        let mut rec = ExchangeReconciler::default_30s();
        assert!(rec.is_sync_due(31_000));
        let rep = rec.reconcile(&mut vault, &truth, 31_000);

        assert!(rep.severe_drift_detected);
        assert_eq!(rep.positions_corrected, 2);
        assert!((vault.cash_balance_usd - 1_015.0).abs() < 1e-6);
        assert!((vault.positions[0].net_qty - 0.02).abs() < 1e-9);
        assert!(!vault.positions[1].is_open());
        assert!(!rec.is_sync_due(40_000));
        assert!(rec.is_sync_due(62_000));
    }

    #[test]
    fn test_atomic_disk_persistence_roundtrip() {
        let mut vault = VaultLedger::new(2_500.0);
        vault.apply_fill(&mk_fill(0, TradeDirection::Long, 0.015, 92_000.0, 0.25, 5_000));
        vault.update_mark_price(SymbolId(0), 92_400.0, 6_000);

        let path = std::env::temp_dir().join("hmtrobot_vault_test_snapshot.txt");
        let saved = vault.save_atomic(&path);
        assert!(saved.is_ok());

        let loaded = VaultLedger::load_or_init(&path, 500.0);
        let _ = fs::remove_file(&path);

        assert!((loaded.cash_balance_usd - vault.cash_balance_usd).abs() < 1e-6);
        assert!((loaded.peak_equity_usd - vault.peak_equity_usd).abs() < 1e-6);
        assert!((loaded.positions[0].net_qty - 0.015).abs() < 1e-9);
        assert!((loaded.positions[0].avg_entry_price - 92_000.0).abs() < 1e-6);
    }
}
