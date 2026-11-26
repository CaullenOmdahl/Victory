use std::fs;
use std::path::{Path, PathBuf};

fn collect_rs(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<_> = fs::read_dir(dir)
        .expect("read src dir")
        .map(|e| e.expect("dir entry").path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            collect_rs(root, &path, out);
        } else if path.extension().and_then(|s| s.to_str()) == Some("rs") {
            let rel = path.strip_prefix(root).expect("relative path").to_path_buf();
            out.push(rel);
        }
    }
}

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=src");

    let mut files = vec![PathBuf::from("Cargo.toml")];
    collect_rs(&manifest_dir, &manifest_dir.join("src"), &mut files);

    files.sort_by(|a, b| {
        a.to_string_lossy()
            .replace('\\', "/")
            .cmp(&b.to_string_lossy().replace('\\', "/"))
    });

    let mut hasher = blake3::Hasher::new();
    for rel in &files {
        let norm = rel.to_string_lossy().replace('\\', "/");
        println!("cargo:rerun-if-changed={norm}");
        hasher.update(norm.as_bytes());
        hasher.update(&[0]);
        let bytes = fs::read(manifest_dir.join(rel)).expect("read shader source");
        hasher.update(&bytes);
    }

    println!(
        "cargo:rustc-env=VICTORY_SHADER_SOURCE_HASH={}",
        hasher.finalize().to_hex()
    );
}
