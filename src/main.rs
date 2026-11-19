mod config;

fn main() {
    let cfg = config::BotConfig::default();
    println!("HMTROBOT_LAB ready | Exchange: {:?} | Mode: {:?}", cfg.exchange, cfg.mode);
}
