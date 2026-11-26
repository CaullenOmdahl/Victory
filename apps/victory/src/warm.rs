use std::path::PathBuf;

use anyhow::Result;

pub fn run(
    paths: Vec<PathBuf>,
    used: bool,
    shader_cache_dir: Option<PathBuf>,
    pipeline_cache_dir: Option<PathBuf>,
) -> Result<()> {
    #[cfg(not(windows))]
    {
        let _ = (paths, used, shader_cache_dir, pipeline_cache_dir);
        println!(
            "victory warm compiles host D3D12 pipelines on Windows/Proton; \
             run `victory translate --cache` on Linux for native DXIL warming, \
             then `victory.exe warm --used` under Proton."
        );
        Ok(())
    }

    #[cfg(windows)]
    {
        use std::collections::HashMap;
        use std::sync::{Mutex, atomic::{AtomicUsize, Ordering}};

        if !used {
            anyhow::bail!("pass --used to warm pipelines recorded in used-pipelines.bin");
        }
        let cfg = victory_core::Config::load()?;
        let shader_dir = match shader_cache_dir {
            Some(d) => d,
            None => cfg.cache_subdir("shaders")?,
        };
        let pipeline_dir = match pipeline_cache_dir {
            Some(d) => d,
            None => cfg.cache_subdir("pipelines")?,
        };
        let needed = victory_gpu::needed_shader_hashes(&pipeline_dir);
        println!(
            "warm: {} needed shader container hashes in {}",
            needed.len(),
            pipeline_dir.display()
        );
        if needed.is_empty() {
            println!("warm: no used-pipelines.bin recipes found yet (run the title once first)");
            return Ok(());
        }

        let paths = crate::default_paths(paths)?;
        let containers: Mutex<HashMap<String, Vec<u8>>> = Mutex::new(HashMap::new());
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);

        for root in &paths {
            for entry in walkdir::WalkDir::new(root).into_iter().filter_map(Result::ok) {
                if !entry.file_type().is_file()
                    || entry.metadata().map_or(0, |m| m.len()) > crate::shader::MAX_FILE
                {
                    continue;
                }
                let Ok(data) = std::fs::read(entry.path()) else {
                    continue;
                };
                let jobs = crate::shader::sources(&data);
                let next = AtomicUsize::new(0);
                std::thread::scope(|s| {
                    for _ in 0..threads.min(jobs.len()) {
                        s.spawn(|| {
                            while let Some(job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                                let Some(bytes) = job.bytes() else {
                                    continue;
                                };
                                let hash = blake3::hash(&bytes).to_hex()[..16].to_owned();
                                if needed.contains(&hash) {
                                    containers
                                        .lock()
                                        .unwrap_or_else(|e| e.into_inner())
                                        .entry(hash)
                                        .or_insert_with(|| bytes.into_owned());
                                }
                            }
                        });
                    }
                });
            }
        }

        let collected = containers.into_inner().unwrap_or_else(|e| e.into_inner());
        println!(
            "warm: matched {} of {} needed shader containers from game files",
            collected.len(),
            needed.len()
        );
        let stats = victory_gpu::warm_used(&collected, shader_dir, pipeline_dir)
            .map_err(anyhow::Error::msg)?;
        println!(
            "warm: recipes={} matched={} built={} loaded={} missing_shaders={} shader_hits={} shader_misses={}",
            stats.recipes,
            stats.matched,
            stats.built,
            stats.loaded,
            stats.missing_shaders,
            stats.shader_hits,
            stats.shader_misses
        );
        Ok(())
    }
}
