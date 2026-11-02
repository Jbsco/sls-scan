struct Params {
    width: u32,
    height: u32,
    stride: u32,
    pad0: u32,
    ir0: f32,
    ir1: f32,
    ir2: f32,
    ir3: f32,
    ir4: f32,
    ir5: f32,
    ir6: f32,
    ir7: f32,
    ir8: f32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    k1: f32,
    k2: f32,
    p1: f32,
    p2: f32,
    k3: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> raw: array<u32>;
@group(0) @binding(2) var<storage, read_write> rectified: array<u32>;

fn raw_at(x: i32, y: i32) -> f32 {
    if (x < 0 || y < 0 || x >= i32(params.width) || y >= i32(params.height)) {
        return 0.0;
    }
    let i = u32(y) * params.width + u32(x);
    return f32((raw[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu);
}

fn sample(j: f32, i: f32) -> f32 {
    let hx = j * params.ir0 + i * params.ir1 + params.ir2;
    let hy = j * params.ir3 + i * params.ir4 + params.ir5;
    let hw = j * params.ir6 + i * params.ir7 + params.ir8;
    let x = hx / hw;
    let y = hy / hw;
    let r2 = x * x + y * y;
    let radial = 1.0 + ((params.k3 * r2 + params.k2) * r2 + params.k1) * r2;
    let u = params.fx * (x * radial + 2.0 * params.p1 * x * y + params.p2 * (r2 + 2.0 * x * x))
        + params.cx;
    let v = params.fy * (y * radial + params.p1 * (r2 + 2.0 * y * y) + 2.0 * params.p2 * x * y)
        + params.cy;
    let x0 = floor(u);
    let y0 = floor(v);
    let tx = u - x0;
    let ty = v - y0;
    let xi = i32(x0);
    let yi = i32(y0);
    let top = (1.0 - tx) * raw_at(xi, yi) + tx * raw_at(xi + 1, yi);
    let bottom = (1.0 - tx) * raw_at(xi, yi + 1) + tx * raw_at(xi + 1, yi + 1);
    return (1.0 - ty) * top + ty * bottom;
}

@compute @workgroup_size(64)
fn rectify(@builtin(global_invocation_id) id: vec3<u32>) {
    let word = id.x + id.y * params.stride;
    let n = params.width * params.height;
    if (word * 4u >= n) {
        return;
    }
    var packed = 0u;
    for (var b = 0u; b < 4u; b++) {
        let p = word * 4u + b;
        if (p < n) {
            let value = sample(f32(p % params.width), f32(p / params.width));
            packed |= u32(clamp(round(value), 0.0, 255.0)) << (8u * b);
        }
    }
    rectified[word] = packed;
}
