use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=shaders");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_dir = Path::new("shaders");
    let spv_dir = out_dir.join("spv");
    fs::create_dir_all(&spv_dir).expect("create OUT_DIR/spv");

    let compiler = find_shader_compiler();
    for entry in fs::read_dir(shader_dir).expect("read shaders/") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if !matches!(ext, "comp" | "vert" | "frag" | "glsl") {
            continue;
        }
        let stem = path.file_name().unwrap().to_string_lossy();
        let out_spv = spv_dir.join(format!("{stem}.spv"));
        compile_shader(&compiler, &path, &out_spv);
        println!("cargo:rustc-env=SHADER_{}={}", env_key(&stem), out_spv.display());
    }
}

fn env_key(stem: &str) -> String {
    stem.replace('.', "_").replace('-', "_").to_ascii_uppercase()
}

enum Compiler {
    Glslc(PathBuf),
    Glslang(PathBuf),
}

fn find_shader_compiler() -> Compiler {
    if let Ok(path) = which("glslc") {
        return Compiler::Glslc(path);
    }
    if let Ok(path) = which("glslangValidator") {
        return Compiler::Glslang(path);
    }
    panic!(
        "Need glslc or glslangValidator on PATH to compile GLSL → SPIR-V.\n\
         Install LunarG Vulkan SDK, or `brew install shaderc` / `brew install glslang`."
    );
}

fn which(bin: &str) -> Result<PathBuf, ()> {
    if let Ok(p) = env::var(format!("{}_PATH", bin.to_ascii_uppercase())) {
        let path = PathBuf::from(p);
        if path.exists() {
            return Ok(path);
        }
    }
    let output = Command::new("which").arg(bin).output().map_err(|_| ())?;
    if !output.status.success() {
        return Err(());
    }
    let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if path.is_empty() {
        Err(())
    } else {
        Ok(PathBuf::from(path))
    }
}

fn compile_shader(compiler: &Compiler, src: &Path, dst: &Path) {
    let status = match compiler {
        Compiler::Glslc(bin) => Command::new(bin)
            .args(["-O", "-o"])
            .arg(dst)
            .arg(src)
            .status(),
        Compiler::Glslang(bin) => Command::new(bin)
            .args(["-V", "-o"])
            .arg(dst)
            .arg(src)
            .status(),
    }
    .unwrap_or_else(|e| panic!("failed to spawn shader compiler for {}: {e}", src.display()));

    if !status.success() {
        panic!("shader compile failed: {}", src.display());
    }
}
