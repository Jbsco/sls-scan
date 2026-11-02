struct Params {
    width: u32,
    height: u32,
    window: u32,
    iterations: u32,
    count: u32,
    pixel_stride: u32,
    sample_stride: u32,
    terms: u32,
    max_move: f32,
    min_correlation: f32,
    gate: u32,
    pad0: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> left: array<u32>;
@group(0) @binding(2) var<storage, read> right: array<u32>;
@group(0) @binding(3) var<storage, read_write> gx: array<f32>;
@group(0) @binding(4) var<storage, read_write> gy: array<f32>;
@group(0) @binding(5) var<storage, read> samples: array<f32>;
@group(0) @binding(6) var<storage, read_write> refined: array<f32>;
@group(0) @binding(7) var<storage, read_write> kept: array<u32>;

const N: u32 = 14u;
const LANES: u32 = 64u;
const CHUNK: u32 = 40u;
const RHS: u32 = 105u;

var<workgroup> partial: array<f32, 2560>;
var<workgroup> sums: array<f32, 120>;
var<workgroup> shared_q: array<f32, 14>;
var<workgroup> state: u32;

fn left_at(i: u32) -> f32 {
    return f32((left[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu);
}

fn right_at(i: u32) -> f32 {
    return f32((right[i >> 2u] >> ((i & 3u) * 8u)) & 0xffu);
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
fn gradients(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = id.x + id.y * params.pixel_stride;
    if (p >= params.width * params.height) {
        return;
    }
    let w = i32(params.width);
    let h = i32(params.height);
    let x = i32(p % params.width);
    let y = i32(p / params.width);
    let xm = u32(reflect101(x - 1, w));
    let xp = u32(reflect101(x + 1, w));
    let ym = u32(reflect101(y - 1, h)) * params.width;
    let yp = u32(reflect101(y + 1, h)) * params.width;
    let xc = u32(x);
    let yc = u32(y) * params.width;
    gx[p] = (3.0 * (right_at(ym + xp) - right_at(ym + xm))
        + 10.0 * (right_at(yc + xp) - right_at(yc + xm))
        + 3.0 * (right_at(yp + xp) - right_at(yp + xm))) / 32.0;
    gy[p] = (3.0 * (right_at(yp + xm) - right_at(ym + xm))
        + 10.0 * (right_at(yp + xc) - right_at(ym + xc))
        + 3.0 * (right_at(yp + xp) - right_at(ym + xp))) / 32.0;
}

fn cubic_weights(t: f32) -> vec4<f32> {
    let a = -0.75;
    let t0 = t + 1.0;
    let t2 = 1.0 - t;
    let c0 = ((a * t0 - 5.0 * a) * t0 + 8.0 * a) * t0 - 4.0 * a;
    let c1 = ((a + 2.0) * t - (a + 3.0)) * t * t + 1.0;
    let c2 = ((a + 2.0) * t2 - (a + 3.0)) * t2 * t2 + 1.0;
    return vec4<f32>(c0, c1, c2, 1.0 - c0 - c1 - c2);
}

fn right_cubic(x: f32, y: f32) -> vec3<f32> {
    let fx = floor(x);
    let fy = floor(y);
    let wx = cubic_weights(x - fx);
    let wy = cubic_weights(y - fy);
    let ix = i32(fx) - 1;
    let iy = i32(fy) - 1;
    var out = vec3<f32>(0.0);
    for (var j = 0; j < 4; j++) {
        let yy = iy + j;
        if (yy < 0 || yy >= i32(params.height)) {
            continue;
        }
        var row = vec3<f32>(0.0);
        for (var i = 0; i < 4; i++) {
            let xx = ix + i;
            if (xx < 0 || xx >= i32(params.width)) {
                continue;
            }
            let k = u32(yy) * params.width + u32(xx);
            row += wx[i] * vec3<f32>(right_at(k), gx[k], gy[k]);
        }
        out += wy[j] * row;
    }
    return out;
}

fn left_linear(x: f32, y: f32) -> f32 {
    let fx = floor(x);
    let fy = floor(y);
    let tx = x - fx;
    let ty = y - fy;
    var out = 0.0;
    for (var j = 0; j < 2; j++) {
        let yy = i32(fy) + j;
        if (yy < 0 || yy >= i32(params.height)) {
            continue;
        }
        for (var i = 0; i < 2; i++) {
            let xx = i32(fx) + i;
            if (xx < 0 || xx >= i32(params.width)) {
                continue;
            }
            let wt = select(1.0 - tx, tx, i == 1) * select(1.0 - ty, ty, j == 1);
            out += wt * left_at(u32(yy) * params.width + u32(xx));
        }
    }
    return out;
}

fn shape_at(k: u32, hw: i32, scale: f32) -> array<f32, 6> {
    let a = f32(i32(k % params.window) - hw) / scale;
    let b = f32(i32(k / params.window) - hw) / scale;
    let quad = select(0.0, 1.0, params.terms > 3u);
    return array<f32, 6>(1.0, a, b, quad * 0.5 * a * a, quad * a * b, quad * 0.5 * b * b);
}

fn upper(c: u32, d: u32) -> u32 {
    return c * N - (c * (c + 1u)) / 2u + d;
}

fn solve(u0: f32, v0: f32) -> u32 {
    var m = array<f32, 196>();
    var r = array<f32, 14>();
    var scaling = array<f32, 14>();
    for (var c = 0u; c < N; c++) {
        r[c] = sums[RHS + c];
        for (var d = c; d < N; d++) {
            m[c * N + d] = sums[upper(c, d)];
        }
        if (params.terms <= 3u && c < 12u && c % 6u >= 3u) {
            m[c * N + c] = 1.0;
            r[c] = 0.0;
        }
        if (!(m[c * N + c] > 0.0)) {
            return 2u;
        }
        scaling[c] = inverseSqrt(m[c * N + c]);
    }
    for (var c = 0u; c < N; c++) {
        r[c] *= scaling[c];
        for (var d = c; d < N; d++) {
            m[c * N + d] *= scaling[c] * scaling[d];
        }
    }
    for (var c = 0u; c < N; c++) {
        var pivot = m[c * N + c];
        for (var k = 0u; k < c; k++) {
            pivot -= m[k * N + c] * m[k * N + c];
        }
        if (!(pivot > 1e-10)) {
            return 2u;
        }
        m[c * N + c] = sqrt(pivot);
        for (var d = c + 1u; d < N; d++) {
            var e = m[c * N + d];
            for (var k = 0u; k < c; k++) {
                e -= m[k * N + c] * m[k * N + d];
            }
            m[c * N + d] = e / m[c * N + c];
        }
    }
    for (var c = 0u; c < N; c++) {
        var e = r[c];
        for (var k = 0u; k < c; k++) {
            e -= m[k * N + c] * r[k];
        }
        r[c] = e / m[c * N + c];
    }
    for (var c = N; c > 0u; c--) {
        let row = c - 1u;
        var e = r[row];
        for (var k = row + 1u; k < N; k++) {
            e -= m[row * N + k] * r[k];
        }
        r[row] = e / m[row * N + row];
    }
    for (var c = 0u; c < N; c++) {
        shared_q[c] += r[c] * scaling[c];
    }
    if (!(abs(shared_q[0] - u0) < 32.0 && abs(shared_q[6] - v0) < 32.0)) {
        return 2u;
    }
    if (max(abs(r[0] * scaling[0]), abs(r[6] * scaling[6])) < 1e-5) {
        return 1u;
    }
    return 0u;
}

@compute @workgroup_size(64)
fn refine(@builtin(workgroup_id) group: vec3<u32>,
          @builtin(local_invocation_index) lane: u32) {
    let s = group.x + group.y * params.sample_stride;
    if (s >= params.count) {
        return;
    }
    let base = s * 8u;
    let lx = samples[base];
    let ly = samples[base + 1u];
    let u0 = samples[base + 2u] - lx;
    let v0 = samples[base + 3u] - ly;
    let hw = i32(params.window / 2u);
    let scale = f32(hw);
    let pixels = params.window * params.window;
    if (lane == 0u) {
        for (var c = 0u; c < N; c++) {
            shared_q[c] = 0.0;
        }
        shared_q[0] = u0;
        shared_q[1] = (samples[base + 4u] - 1.0) * scale;
        shared_q[2] = samples[base + 5u] * scale;
        shared_q[6] = v0;
        shared_q[7] = samples[base + 6u] * scale;
        shared_q[8] = (samples[base + 7u] - 1.0) * scale;
        shared_q[12] = 1.0;
        state = 0u;
    }
    workgroupBarrier();
    for (var it = 0u; it < params.iterations; it++) {
        var q = array<f32, 14>();
        for (var c = 0u; c < N; c++) {
            q[c] = shared_q[c];
        }
        // Explicit initializers zero these sums on every iteration.
        var acc = array<f32, 120>();
        for (var k = lane; k < pixels; k += LANES) {
            let i = i32(k % params.window) - hw;
            let j = i32(k / params.window) - hw;
            let shape = shape_at(k, hw, scale);
            var u = 0.0;
            var v = 0.0;
            for (var c = 0u; c < 6u; c++) {
                u += q[c] * shape[c];
                v += q[6u + c] * shape[c];
            }
            let g = right_cubic(lx + f32(i) + u, ly + f32(j) + v);
            let f = left_linear(lx + f32(i), ly + f32(j));
            let residual = f - (q[12] * g.x + q[13]);
            var jac = array<f32, 14>();
            for (var c = 0u; c < 6u; c++) {
                jac[c] = q[12] * g.y * shape[c];
                jac[6u + c] = q[12] * g.z * shape[c];
            }
            jac[12] = g.x;
            jac[13] = 1.0;
            for (var c = 0u; c < N; c++) {
                acc[RHS + c] += jac[c] * residual;
                for (var d = c; d < N; d++) {
                    acc[upper(c, d)] += jac[c] * jac[d];
                }
            }
        }
        for (var chunk = 0u; chunk < 3u; chunk++) {
            for (var c = 0u; c < CHUNK; c++) {
                partial[lane * CHUNK + c] = acc[chunk * CHUNK + c];
            }
            workgroupBarrier();
            for (var stride = LANES / 2u; stride > 0u; stride /= 2u) {
                if (lane < stride) {
                    for (var c = 0u; c < CHUNK; c++) {
                        partial[lane * CHUNK + c] += partial[(lane + stride) * CHUNK + c];
                    }
                }
                workgroupBarrier();
            }
            if (lane == 0u) {
                for (var c = 0u; c < CHUNK; c++) {
                    sums[chunk * CHUNK + c] = partial[c];
                }
            }
            workgroupBarrier();
        }
        if (lane == 0u) {
            state = solve(u0, v0);
        }
        if (workgroupUniformLoad(&state) != 0u) {
            break;
        }
    }
    var correlated = true;
    if (params.gate != 0u) {
        var q = array<f32, 14>();
        for (var c = 0u; c < N; c++) {
            q[c] = shared_q[c];
        }
        var moments = array<f32, 5>();
        for (var k = lane; k < pixels; k += LANES) {
            let i = i32(k % params.window) - hw;
            let j = i32(k / params.window) - hw;
            let shape = shape_at(k, hw, scale);
            var u = 0.0;
            var v = 0.0;
            for (var c = 0u; c < 6u; c++) {
                u += q[c] * shape[c];
                v += q[6u + c] * shape[c];
            }
            let f = left_linear(lx + f32(i), ly + f32(j)) - 128.0;
            let g = right_cubic(lx + f32(i) + u, ly + f32(j) + v).x - 128.0;
            moments[0] += f;
            moments[1] += g;
            moments[2] += f * f;
            moments[3] += g * g;
            moments[4] += f * g;
        }
        for (var c = 0u; c < 5u; c++) {
            partial[lane * CHUNK + c] = moments[c];
        }
        workgroupBarrier();
        for (var stride = LANES / 2u; stride > 0u; stride /= 2u) {
            if (lane < stride) {
                for (var c = 0u; c < 5u; c++) {
                    partial[lane * CHUNK + c] += partial[(lane + stride) * CHUNK + c];
                }
            }
            workgroupBarrier();
        }
        let n = f32(pixels);
        let vf = partial[2] - partial[0] * partial[0] / n;
        let vg = partial[3] - partial[1] * partial[1] / n;
        let cov = partial[4] - partial[0] * partial[1] / n;
        correlated = cov / sqrt(max(vf * vg, 1e-12)) >= params.min_correlation;
    }
    if (lane == 0u) {
        refined[2u * s] = lx + shared_q[0];
        refined[2u * s + 1u] = ly + shared_q[6];
        let moved = abs(shared_q[0] - u0) < params.max_move;
        kept[s] = select(0u, 1u, state != 2u && moved && correlated);
    }
}
