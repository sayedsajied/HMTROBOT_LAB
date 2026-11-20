mod config;

fn main() {
    match config::BotConfig::from_env() {
        Ok(cfg) => println!(
            "HMTROBOT_LAB v2 OK | {:?} | MinMakerEdge: {:.2}bps",
            cfg,
            cfg.fees
                .min_required_edge_bps(cfg.risk.min_edge_to_fee_ratio, cfg.execution.maker_first)
        ),
        Err(e) => eprintln!("Startup blocked: {}", e),
    }
}
mod types;
mod feeds;
mod math_kernel;
mod ai_router;

