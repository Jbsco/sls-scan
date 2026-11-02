struct Params {
    width: u32,
    height: u32,
    small_width: u32,
    small_height: u32,
    scale: u32,
    block: u32,
    lo: i32,
    count: u32,
    min_correlation: f32,
    margin: f32,
    small_stride: u32,
    volume_stride: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> left: array<u32>;
@group(0) @binding(2) var<storage, read> right: array<u32>;
@group(0) @binding(3) var<storage, read_write> small_left: array<f32>;
@group(0) @binding(4) var<storage, read_write> small_right: array<f32>;
@group(0) @binding(5) var<storage, read_write> volume: array<f32>;
@group(0) @binding(6) var<storage, read_write> source: array<f32>;
@group(0) @binding(7) var<storage, read_write> source_valid: array<u32>;
@group(0) @binding(8) var<storage, read_write> output: array<f32>;
@group(0) @binding(9) var<storage, read_write> output_valid: array<u32>;
@group(0) @binding(10) var<storage, read_write> block_stats: array<vec4<f32>>;

fn byte_at(image: u32, i: u32) -> u32 {
    if (image == 0u) {
        return (left[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu;
    }
    return (right[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu;
}

fn reflect101(i: i32, n: i32) -> i32 {
    if (i < 0) {
        return -i;
    }
    if (i >= n) {
        return 2 * n - 2 - i;
    }
    return i;
}

@compute @workgroup_size(64)
fn downsample(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = id.x + id.y * params.small_stride;
    if (p >= params.small_width * params.small_height) {
        return;
    }
    let x0 = (p % params.small_width) * params.scale;
    let y0 = (p / params.small_width) * params.scale;
    var sum_left = 0u;
    var sum_right = 0u;
    for (var j = 0u; j < params.scale; j++) {
        for (var i = 0u; i < params.scale; i++) {
            let k = (y0 + j) * params.width + x0 + i;
            sum_left += byte_at(0u, k);
            sum_right += byte_at(1u, k);
        }
    }
    let n = f32(params.scale * params.scale);
    small_left[p] = round(f32(sum_left) / n);
    small_right[p] = round(f32(sum_right) / n);
}

@compute @workgroup_size(64)
fn stats(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = id.x + id.y * params.small_stride;
    if (p >= params.small_width * params.small_height) {
        return;
    }
    let w = i32(params.small_width);
    let h = i32(params.small_height);
    let x = i32(p % params.small_width);
    let y = i32(p / params.small_width);
    let r = i32(params.block) / 2;
    var sums = vec4<f32>(0.0);
    for (var dy = -r; dy <= r; dy++) {
        let row = reflect101(y + dy, h) * w;
        for (var dx = -r; dx <= r; dx++) {
            let k = row + reflect101(x + dx, w);
            let l = small_left[k];
            let rv = small_right[k];
            sums += vec4<f32>(l, l * l, rv, rv * rv);
        }
    }
    block_stats[p] = sums;
}

@compute @workgroup_size(64)
fn correlate(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixels = params.small_width * params.small_height;
    let i = id.x + id.y * params.volume_stride;
    if (i >= pixels * params.count) {
        return;
    }
    let p = i % pixels;
    let k = i / pixels;
    let w = i32(params.small_width);
    let h = i32(params.small_height);
    let x = i32(p % params.small_width);
    let y = i32(p / params.small_width);
    let d = params.lo + i32(k);
    let b = i32(params.block);
    var out = -1.0;
    let masked = select((x >= w + d - b), (x < d + b), (d >= 0));
    if (abs(d) + b < w && !masked) {
        let r = b / 2;
        var slr = 0.0;
        for (var dy = -r; dy <= r; dy++) {
            let row = reflect101(y + dy, h) * w;
            for (var dx = -r; dx <= r; dx++) {
                let xl = reflect101(x + dx, w);
                let xr = reflect101(x - d + dx, w);
                slr += small_left[row + xl] * small_right[row + xr];
            }
        }
        let left = block_stats[p];
        let right = block_stats[u32(y) * params.small_width + u32(x - d)];
        let inv = 1.0 / f32(b * b);
        let ml = left.x * inv;
        let mr = right.z * inv;
        let vl = left.y * inv - ml * ml;
        let vr = right.w * inv - mr * mr;
        let cov = slr * inv - ml * mr;
        out = cov / sqrt(max(vl * vr, 1e-6));
    }
    volume[i] = out;
}

@compute @workgroup_size(64)
fn choose(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixels = params.small_width * params.small_height;
    let p = id.x + id.y * params.small_stride;
    if (p >= pixels) {
        return;
    }
    let n = params.count;
    var best = 0u;
    var peak = volume[p];
    for (var k = 1u; k < n; k++) {
        let v = volume[k * pixels + p];
        if (v > peak) {
            peak = v;
            best = k;
        }
    }
    let below = volume[select(best - 1u, 0u, best == 0u) * pixels + p];
    let above = volume[min(best + 1u, n - 1u) * pixels + p];
    let curvature = below - 2.0 * peak + above;
    var offset = 0.0;
    if (abs(curvature) > 1e-6) {
        offset = clamp(0.5 * (below - above) / curvature, -0.5, 0.5);
    }
    let d = f32(params.lo + i32(best)) + offset;
    var second = -1.0;
    for (var k = 0u; k < n; k++) {
        if (k + 1u < best || k > best + 1u) {
            second = max(second, volume[k * pixels + p]);
        }
    }
    var ok = peak >= params.min_correlation && peak - second >= params.margin;
    ok = ok && best > 0u && best < n - 1u;
    let w = i32(params.small_width);
    let x = i32(p % params.small_width);
    let row = p - u32(x);
    let xr = clamp(i32(round(f32(x) - d)), 0, w - 1);
    var back = 0u;
    var back_peak = -2.0;
    for (var k = 0u; k < n; k++) {
        let column = clamp(xr + params.lo + i32(k), 0, w - 1);
        let v = volume[k * pixels + row + u32(column)];
        if (v > back_peak) {
            back_peak = v;
            back = k;
        }
    }
    ok = ok && abs(f32(params.lo + i32(back)) - d) <= 1.0;
    output[p] = d;
    output_valid[p] = select(0u, 1u, ok);
}

@compute @workgroup_size(64)
fn fill(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = id.x + id.y * params.small_stride;
    if (p >= params.small_width * params.small_height) {
        return;
    }
    if (source_valid[p] != 0u) {
        output[p] = source[p];
        output_valid[p] = 1u;
        return;
    }
    let w = i32(params.small_width);
    let h = i32(params.small_height);
    let x = i32(p % params.small_width);
    let y = i32(p / params.small_width);
    var total = 0.0;
    var count = 0.0;
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let q = u32(reflect101(y + dy, h) * w + reflect101(x + dx, w));
            if (source_valid[q] != 0u) {
                total += source[q];
                count += 1.0;
            }
        }
    }
    output[p] = select(source[p], total / count, count >= 3.0);
    output_valid[p] = select(0u, 1u, count >= 3.0);
}
