#![allow(dead_code)]

use std::fmt;

/// Zero-leak wrapper for API keys and tokens (Optiver/Wintermute security standard)
#[derive(Clone, Default)]
pub struct SecretString(String);

impl SecretString {
    pub fn new(value: String) -> Self {
        Self(value.trim().to_string())
    }

    #[inline]
    pub fn expose(&self) -> &str {
        &self.0
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            f.write_str("[EMPTY]")
        } else {
            f.write_str("[REDACTED]")
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExchangeId {
    DeltaExchange,
    Bybit,
    Binance,
}

impl ExchangeId {
    pub fn ws_url(&self) -> &'static str {
        match self {
            Self::DeltaExchange => "wss://socket.india.delta.exchange",
            Self::Bybit => "wss://stream.bybit.com/v5/public/linear",
            Self::Binance => "wss://fstream.binance.com/ws",
        }
    }

    pub fn rest_url(&self) -> &'static str {
        match self {
            Self::DeltaExchange => "https://api.india.delta.exchange",
            Self::Bybit => "https://api.bybit.com",
            Self::Binance => "https://fapi.binance.com",
        }
    }

    /// Real baseline fee schedules (basis points) per exchange
    pub fn default_fees(&self) -> FeeConfig {
        match self {
            Self::DeltaExchange => FeeConfig {
                maker_fee_bps: 2.0,
                taker_fee_bps: 5.0,
                slippage_bps: 1.5,
            },
            Self::Bybit => FeeConfig {
                maker_fee_bps: 2.0,
                taker_fee_bps: 5.5,
                slippage_bps: 1.2,
            },
            Self::Binance => FeeConfig {
                maker_fee_bps: 2.0,
                taker_fee_bps: 5.0,
                slippage_bps: 1.0,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradingMode {
    Paper,
    Live,
}

#[derive(Debug, Clone, Copy)]
pub struct FeeConfig {
    pub maker_fee_bps: f64,
    pub taker_fee_bps: f64,
    pub slippage_bps: f64,
}

impl FeeConfig {
    #[inline]
    pub fn round_trip_taker_bps(&self) -> f64 {
        (self.taker_fee_bps * 2.0) + (self.slippage_bps * 2.0)
    }

    #[inline]
    pub fn round_trip_maker_bps(&self) -> f64 {
        (self.maker_fee_bps * 2.0) + self.slippage_bps
    }

    #[inline]
    pub fn min_required_edge_bps(&self, edge_ratio: f64, is_maker: bool) -> f64 {
        let cost = if is_maker {
            self.round_trip_maker_bps()
        } else {
            self.round_trip_taker_bps()
        };
        cost * edge_ratio.max(1.0)
    }
}

/// Jump Trading execution discipline parameters
#[derive(Debug, Clone, Copy)]
pub struct ExecutionConfig {
    pub maker_first: bool,
    pub maker_timeout_ms: u64,
    pub max_order_retries: u8,
    pub reconcile_interval_sec: u64,
    pub exchange_hard_sl_bps: f64,
    pub inventory_skew_bps: f64,
}

/// Hard pre-trade risk & adverse-selection controls
#[derive(Debug, Clone, Copy)]
pub struct RiskConfig {
    pub max_daily_loss_pct: f64,
    pub max_position_usd: f64,
    pub max_leverage: f64,
    pub max_spread_bps: f64,
    pub stale_feed_ms: u64,
    pub trade_cooldown_ms: u64,
    pub min_edge_to_fee_ratio: f64,
    pub toxic_flow_veto_threshold: f64,
    pub vol_spike_veto_bps: f64,
    pub max_consecutive_losses: u32,
}

#[derive(Debug, Clone)]
pub struct BotConfig {
    pub exchange: ExchangeId,
    pub mode: TradingMode,
    pub symbols: Vec<String>,
    pub initial_capital_usd: f64,
    pub fees: FeeConfig,
    pub execution: ExecutionConfig,
    pub risk: RiskConfig,
    pub api_key: SecretString,
    pub api_secret: SecretString,
    pub telegram_bot_token: SecretString,
    pub telegram_chat_id: String,
    pub health_port: u16,
}

impl Default for BotConfig {
    fn default() -> Self {
        let exchange = ExchangeId::DeltaExchange;
        Self {
            exchange,
            mode: TradingMode::Paper,
            symbols: vec!["BTCUSD".to_string(), "PAXGUSD".to_string()],
            initial_capital_usd: 100.0,
            fees: exchange.default_fees(),
            execution: ExecutionConfig {
                maker_first: true,
                maker_timeout_ms: 1500,
                max_order_retries: 3,
                reconcile_interval_sec: 30,
                exchange_hard_sl_bps: 45.0,
                inventory_skew_bps: 2.0,
            },
            risk: RiskConfig {
                max_daily_loss_pct: 3.0,
                max_position_usd: 25.0,
                max_leverage: 3.0,
                max_spread_bps: 4.5,
                stale_feed_ms: 3000,
                trade_cooldown_ms: 15_000,
                min_edge_to_fee_ratio: 3.0,
                toxic_flow_veto_threshold: 0.75,
                vol_spike_veto_bps: 35.0,
                max_consecutive_losses: 4,
            },
            api_key: SecretString::default(),
            api_secret: SecretString::default(),
            telegram_bot_token: SecretString::default(),
            telegram_chat_id: String::new(),
            health_port: 10000,
        }
    }
}
