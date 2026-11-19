mod config;

fn main() {
    match config::BotConfig::from_env() {
        Ok(cfg) => println!(
            "HMTROBOT_LAB ready | Exchange: {:?} ({}) | Mode: {:?} | MinMakerEdge: {:.2}bps",
            cfg.exchange,
            cfg.exchange.ws_url(),
            cfg.mode,
            cfg.fees.min_required_edge_bps(cfg.risk.min_edge_to_fee_ratio, true)
        ),
        Err(e) => eprintln!("Startup blocked: {}", e),
    }
}
