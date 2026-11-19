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
fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite())
        .unwrap_or(default)
}

fn env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_bool(key: &str, default: bool) -> bool {
    std::env::var(key)
        .ok()
        .map(|v| match v.trim().to_uppercase().as_str() {
            "1" | "TRUE" | "YES" | "ON" => true,
            "0" | "FALSE" | "NO" | "OFF" => false,
            _ => default,
        })
        .unwrap_or(default)
}

fn env_secret(key: &str) -> SecretString {
    SecretString::new(std::env::var(key).unwrap_or_default())
}

impl BotConfig {
    /// Zero-panic loader for all exchange, fee, execution, risk, and secret ENV vars
    pub fn from_env() -> Result<Self, String> {
        let mut cfg = Self::default();

        if let Ok(ex) = std::env::var("EXCHANGE") {
            cfg.exchange = match ex.trim().to_uppercase().as_str() {
                "BYBIT" => ExchangeId::Bybit,
                "BINANCE" => ExchangeId::Binance,
                _ => ExchangeId::DeltaExchange,
            };
            cfg.fees = cfg.exchange.default_fees();
        }

        if let Ok(mode) = std::env::var("TRADING_MODE") {
            cfg.mode = match mode.trim().to_uppercase().as_str() {
                "LIVE" => TradingMode::Live,
                _ => TradingMode::Paper,
            };
        }

        if let Ok(syms) = std::env::var("SYMBOLS") {
            let parsed: Vec<String> = syms
                .split(',')
                .map(|s| s.trim().to_uppercase())
                .filter(|s| !s.is_empty())
                .collect();
            if !parsed.is_empty() {
                cfg.symbols = parsed;
            }
        }

        cfg.initial_capital_usd = env_f64("INITIAL_CAPITAL_USD", cfg.initial_capital_usd);

        // Fee overrides
        cfg.fees.maker_fee_bps = env_f64("MAKER_FEE_BPS", cfg.fees.maker_fee_bps);
        cfg.fees.taker_fee_bps = env_f64("TAKER_FEE_BPS", cfg.fees.taker_fee_bps);
        cfg.fees.slippage_bps = env_f64("SLIPPAGE_BPS", cfg.fees.slippage_bps);

        // Execution overrides (Jump Trading controls)
        cfg.execution.maker_first = env_bool("MAKER_FIRST", cfg.execution.maker_first);
        cfg.execution.maker_timeout_ms = env_u64("MAKER_TIMEOUT_MS", cfg.execution.maker_timeout_ms);
        cfg.execution.max_order_retries = env_u64(
            "MAX_ORDER_RETRIES",
            cfg.execution.max_order_retries as u64,
        )
        .min(u8::MAX as u64) as u8;
        cfg.execution.reconcile_interval_sec = env_u64(
            "RECONCILE_INTERVAL_SEC",
            cfg.execution.reconcile_interval_sec,
        );
        cfg.execution.exchange_hard_sl_bps =
            env_f64("EXCHANGE_HARD_SL_BPS", cfg.execution.exchange_hard_sl_bps);
        cfg.execution.inventory_skew_bps =
            env_f64("INVENTORY_SKEW_BPS", cfg.execution.inventory_skew_bps);

        // Risk & adverse-selection overrides
        cfg.risk.max_daily_loss_pct = env_f64("MAX_DAILY_LOSS_PCT", cfg.risk.max_daily_loss_pct);
        cfg.risk.max_position_usd = env_f64("MAX_POSITION_USD", cfg.risk.max_position_usd);
        cfg.risk.max_leverage = env_f64("MAX_LEVERAGE", cfg.risk.max_leverage);
        cfg.risk.max_spread_bps = env_f64("MAX_SPREAD_BPS", cfg.risk.max_spread_bps);
        cfg.risk.stale_feed_ms = env_u64("STALE_FEED_MS", cfg.risk.stale_feed_ms);
        cfg.risk.trade_cooldown_ms = env_u64("TRADE_COOLDOWN_MS", cfg.risk.trade_cooldown_ms);
        cfg.risk.min_edge_to_fee_ratio =
            env_f64("MIN_EDGE_TO_FEE_RATIO", cfg.risk.min_edge_to_fee_ratio);
        cfg.risk.toxic_flow_veto_threshold = env_f64(
            "TOXIC_FLOW_VETO_THRESHOLD",
            cfg.risk.toxic_flow_veto_threshold,
        );
        cfg.risk.vol_spike_veto_bps = env_f64("VOL_SPIKE_VETO_BPS", cfg.risk.vol_spike_veto_bps);
        cfg.risk.max_consecutive_losses = env_u64(
            "MAX_CONSECUTIVE_LOSSES",
            cfg.risk.max_consecutive_losses as u64,
        )
        .min(u32::MAX as u64) as u32;

        // Secrets & Render health port
        cfg.api_key = env_secret("API_KEY");
        cfg.api_secret = env_secret("API_SECRET");
        cfg.telegram_bot_token = env_secret("TELEGRAM_BOT_TOKEN");
        cfg.telegram_chat_id = std::env::var("TELEGRAM_CHAT_ID")
            .unwrap_or_default()
            .trim()
            .to_string();
        cfg.health_port = env_u64("PORT", cfg.health_port as u64).clamp(1, 65535) as u16;

        cfg.validate()?;
        Ok(cfg)
    }

    /// Institutional pre-flight safety validator (blocks any misconfiguration before startup)
    pub fn validate(&self) -> Result<(), String> {
        if self.symbols.is_empty() {
            return Err("Config Error: symbols list cannot be empty".into());
        }
        if !self.initial_capital_usd.is_finite() || self.initial_capital_usd <= 0.0 {
            return Err("Config Error: initial_capital_usd must be finite and > 0".into());
        }

        // Fee checks (maker fee allows small rebate down to -5 bps)
        if !self.fees.maker_fee_bps.is_finite()
            || self.fees.maker_fee_bps < -5.0
            || self.fees.maker_fee_bps > 50.0
        {
            return Err("Config Error: maker_fee_bps out of safe range [-5, 50]".into());
        }
        if !self.fees.taker_fee_bps.is_finite()
            || self.fees.taker_fee_bps < 0.0
            || self.fees.taker_fee_bps > 50.0
        {
            return Err("Config Error: taker_fee_bps out of safe range [0, 50]".into());
        }
        if !self.fees.slippage_bps.is_finite()
            || self.fees.slippage_bps < 0.0
            || self.fees.slippage_bps > 50.0
        {
            return Err("Config Error: slippage_bps out of safe range [0, 50]".into());
        }

        // Execution checks
        if self.execution.maker_timeout_ms < 100 || self.execution.maker_timeout_ms > 10_000 {
            return Err("Config Error: maker_timeout_ms must be in [100, 10000]".into());
        }
        if self.execution.max_order_retries == 0 || self.execution.max_order_retries > 10 {
            return Err("Config Error: max_order_retries must be in [1, 10]".into());
        }
        if self.execution.reconcile_interval_sec < 5 || self.execution.reconcile_interval_sec > 300
        {
            return Err("Config Error: reconcile_interval_sec must be in [5, 300]".into());
        }
        if !self.execution.exchange_hard_sl_bps.is_finite()
            || self.execution.exchange_hard_sl_bps < 5.0
            || self.execution.exchange_hard_sl_bps > 500.0
        {
            return Err("Config Error: exchange_hard_sl_bps must be in [5, 500]".into());
        }
        if !self.execution.inventory_skew_bps.is_finite()
            || self.execution.inventory_skew_bps < 0.0
            || self.execution.inventory_skew_bps > 50.0
        {
            return Err("Config Error: inventory_skew_bps must be in [0, 50]".into());
        }

        // Risk checks
        if !self.risk.max_daily_loss_pct.is_finite()
            || self.risk.max_daily_loss_pct <= 0.0
            || self.risk.max_daily_loss_pct > 20.0
        {
            return Err("Config Error: max_daily_loss_pct must be in (0, 20]".into());
        }
        if !self.risk.max_leverage.is_finite()
            || self.risk.max_leverage < 1.0
            || self.risk.max_leverage > 20.0
        {
            return Err("Config Error: max_leverage must be in [1, 20]".into());
        }
        if !self.risk.max_position_usd.is_finite()
            || self.risk.max_position_usd <= 0.0
            || self.risk.max_position_usd > (self.initial_capital_usd * self.risk.max_leverage)
        {
            return Err("Config Error: max_position_usd exceeds capital * leverage".into());
        }
        if !self.risk.max_spread_bps.is_finite()
            || self.risk.max_spread_bps <= 0.0
            || self.risk.max_spread_bps > 50.0
        {
            return Err("Config Error: max_spread_bps must be in (0, 50]".into());
        }
        if self.risk.stale_feed_ms < 500 || self.risk.stale_feed_ms > 10_000 {
            return Err("Config Error: stale_feed_ms must be in [500, 10000]".into());
        }
        if self.risk.trade_cooldown_ms < 500 || self.risk.trade_cooldown_ms > 300_000 {
            return Err("Config Error: trade_cooldown_ms must be in [500, 300000]".into());
        }
        if !self.risk.min_edge_to_fee_ratio.is_finite() || self.risk.min_edge_to_fee_ratio < 1.5 {
            return Err("Config Error: min_edge_to_fee_ratio must be >= 1.5".into());
        }
        if !self.risk.toxic_flow_veto_threshold.is_finite()
            || self.risk.toxic_flow_veto_threshold <= 0.0
            || self.risk.toxic_flow_veto_threshold > 1.0
        {
            return Err("Config Error: toxic_flow_veto_threshold must be in (0, 1]".into());
        }
        if !self.risk.vol_spike_veto_bps.is_finite()
            || self.risk.vol_spike_veto_bps < 5.0
            || self.risk.vol_spike_veto_bps > 500.0
        {
            return Err("Config Error: vol_spike_veto_bps must be in [5, 500]".into());
        }
        if self.risk.max_consecutive_losses == 0 || self.risk.max_consecutive_losses > 20 {
            return Err("Config Error: max_consecutive_losses must be in [1, 20]".into());
        }

        // Fatal live-mode guard
        if self.mode == TradingMode::Live
            && (self.api_key.is_empty() || self.api_secret.is_empty())
        {
            return Err("FATAL SAFETY BLOCK: LIVE mode requires non-empty API_KEY and API_SECRET".into());
        }

        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secret_string_redaction_and_trim() {
        let empty = SecretString::default();
        assert!(empty.is_empty());
        assert_eq!(format!("{:?}", empty), "[EMPTY]");

        let secret = SecretString::new("  live_api_key_999  ".to_string());
        assert!(!secret.is_empty());
        assert_eq!(secret.expose(), "live_api_key_999");
        assert_eq!(format!("{:?}", secret), "[REDACTED]");
        assert!(!format!("{:?}", secret).contains("999"));
    }

    #[test]
    fn test_default_config_is_valid_and_paper_mode() {
        let cfg = BotConfig::default();
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.mode, TradingMode::Paper);
        assert!(cfg.execution.maker_first);
    }

    #[test]
    fn test_exchange_endpoints_and_fee_math() {
        for ex in [ExchangeId::DeltaExchange, ExchangeId::Bybit, ExchangeId::Binance] {
            assert!(ex.ws_url().starts_with("wss://"));
            assert!(ex.rest_url().starts_with("https://"));
        }

        let delta = ExchangeId::DeltaExchange.default_fees();
        assert!((delta.round_trip_maker_bps() - 5.5).abs() < 1e-9);
        assert!((delta.round_trip_taker_bps() - 13.0).abs() < 1e-9);
        assert!((delta.min_required_edge_bps(3.0, true) - 16.5).abs() < 1e-9);
        assert!((delta.min_required_edge_bps(3.0, false) - 39.0).abs() < 1e-9);

        let bybit = ExchangeId::Bybit.default_fees();
        assert!((bybit.round_trip_taker_bps() - 13.4).abs() < 1e-9);

        let binance = ExchangeId::Binance.default_fees();
        assert!((binance.round_trip_taker_bps() - 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_live_mode_requires_non_empty_secrets() {
        let mut cfg = BotConfig::default();
        cfg.mode = TradingMode::Live;
        assert!(cfg.validate().is_err());

        cfg.api_key = SecretString::new("key_ok".to_string());
        assert!(cfg.validate().is_err());

        cfg.api_secret = SecretString::new("secret_ok".to_string());
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_execution_config_guards() {
        let mut cfg = BotConfig::default();
        cfg.execution.maker_timeout_ms = 50; // Too low (< 100ms)
        assert!(cfg.validate().is_err());

        let mut cfg2 = BotConfig::default();
        cfg2.execution.reconcile_interval_sec = 2; // Too frequent (< 5s)
        assert!(cfg2.validate().is_err());

        let mut cfg3 = BotConfig::default();
        cfg3.execution.exchange_hard_sl_bps = 2.0; // Too tight (< 5 bps)
        assert!(cfg3.validate().is_err());

        let mut cfg4 = BotConfig::default();
        cfg4.execution.max_order_retries = 0; // Must be >= 1
        assert!(cfg4.validate().is_err());
    }

    #[test]
    fn test_risk_and_adverse_selection_guards() {
        let mut cfg = BotConfig::default();
        cfg.risk.max_position_usd = 500.0; // Exceeds 100 * 3x = 300
        assert!(cfg.validate().is_err());

        let mut nan_cfg = BotConfig::default();
        nan_cfg.initial_capital_usd = f64::NAN;
        assert!(nan_cfg.validate().is_err());

        let mut toxic_cfg = BotConfig::default();
        toxic_cfg.risk.toxic_flow_veto_threshold = 1.5; // Must be <= 1.0
        assert!(toxic_cfg.validate().is_err());

        let mut vol_cfg = BotConfig::default();
        vol_cfg.risk.vol_spike_veto_bps = 1.0; // Must be >= 5.0
        assert!(vol_cfg.validate().is_err());
    }

    #[test]
    fn test_stale_feed_and_daily_loss_bounds() {
        let mut cfg = BotConfig::default();
        cfg.risk.stale_feed_ms = 100;
        assert!(cfg.validate().is_err());

        let mut loss_cfg = BotConfig::default();
        loss_cfg.risk.max_daily_loss_pct = 25.0;
        assert!(loss_cfg.validate().is_err());
    }

    #[test]
    fn test_bot_config_debug_hides_all_secrets() {
        let mut cfg = BotConfig::default();
        cfg.api_key = SecretString::new("SUPER_SECRET_KEY".to_string());
        cfg.api_secret = SecretString::new("SUPER_SECRET_PASS".to_string());
        cfg.telegram_bot_token = SecretString::new("TG_BOT_TOKEN_123".to_string());

        let dumped = format!("{:?}", cfg);
        assert!(!dumped.contains("SUPER_SECRET_KEY"));
        assert!(!dumped.contains("SUPER_SECRET_PASS"));
        assert!(!dumped.contains("TG_BOT_TOKEN_123"));
        assert!(dumped.contains("[REDACTED]"));
    }
}
