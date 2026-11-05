struct Params {
    width: u32,
    height: u32,
    planes: u32,
    steps: u32,
    projector_width: u32,
    pitch: f32,
    min_contrast: f32,
    min_modulation: f32,
    max_disagreement: f32,
    row_stride: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gray: array<u32>;
@group(0) @binding(2) var<storage, read> phase: array<u32>;
@group(0) @binding(3) var<storage, read_write> columns: array<f32>;
@group(0) @binding(4) var<storage, read_write> mask: array<u32>;

const TAU: f32 = 6.283185307179586;

fn gray_at(i: u32) -> f32 {
    return f32((gray[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu);
}

fn phase_at(i: u32) -> f32 {
    return f32((phase[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = params.width * params.height;
    let p = id.x + id.y * params.row_stride;
    if (p >= n) {
        return;
    }
    var ok = true;
    var code = 0u;
    for (var k = 0u; k < params.planes; k++) {
        let d = gray_at(2u * k * n + p) - gray_at((2u * k + 1u) * n + p);
        code = (code << 1u) | select(0u, 1u, d > 0.0);
        ok = ok && abs(d) >= params.min_contrast;
    }
    var column = code;
    var shift = code >> 1u;
    loop {
        if (shift == 0u) {
            break;
        }
        column = column ^ shift;
        shift = shift >> 1u;
    }
    ok = ok && column < params.projector_width;
    var num = 0.0;
    var den = 0.0;
    var saturated = false;
    for (var i = 0u; i < params.steps; i++) {
        let v = phase_at(i * n + p);
        saturated = saturated || v >= 255.0;
        let angle = TAU * f32(i) / f32(params.steps);
        num += v * sin(angle);
        den += v * cos(angle);
    }
    ok = ok && !saturated;
    ok = ok && 2.0 / f32(params.steps) * sqrt(num * num + den * den) >= params.min_modulation;
    let negated = -atan2(num, den);
    let wrapped = negated - TAU * floor(negated / TAU);
    let fraction = wrapped * params.pitch / TAU;
    let gray_column = f32(column);
    let u = fraction + params.pitch * round((gray_column - fraction) / params.pitch);
    ok = ok && abs(u - gray_column) <= params.max_disagreement;
    ok = ok && u >= -0.5 && u <= f32(params.projector_width) - 0.5;
    columns[p] = u;
    mask[p] = select(0u, 1u, ok);
}
