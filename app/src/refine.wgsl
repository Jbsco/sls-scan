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
const RHS: u32 = 105u;
const SUMS: u32 = 119u;
const HALF: u32 = 60u;
// Each lane's accumulator row is 61 long, odd, so lanes land on different memory banks.
const STRIDE: u32 = 61u;

var<workgroup> acc: array<f32, 3904>;
var<workgroup> sums: array<f32, 120>;
var<workgroup> chol: array<f32, 196>;
var<workgroup> rhs: array<f32, 14>;
var<workgroup> scaling: array<f32, 14>;
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

fn outer(a: vec3<f32>, b: vec3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(a * b.x, a * b.y, a * b.z);
}

fn store(row: u32, lo: u32, c: u32, d: u32, v: f32) {
    let k = upper(c, d);
    if (k >= lo && k < lo + HALF) {
        acc[row + k - lo] = v;
    }
}

fn store_rhs(row: u32, lo: u32, c: u32, v: f32) {
    let k = RHS + c;
    if (k >= lo && k < lo + HALF) {
        acc[row + k - lo] = v;
    }
}

fn reduce(lane: u32, lo: u32, count: u32) {
    workgroupBarrier();
    for (var c = lane; c < count; c += LANES) {
        var total = 0.0;
        for (var l = 0u; l < LANES; l++) {
            total += acc[l * STRIDE + c];
        }
        sums[lo + c] = total;
    }
    workgroupBarrier();
}

fn solve(lane: u32, u0: f32, v0: f32) {
    if (lane < N) {
        let c = lane;
        for (var d = c; d < N; d++) {
            chol[c * N + d] = sums[upper(c, d)];
        }
        rhs[c] = sums[RHS + c];
        if (params.terms <= 3u && c < 12u && c % 6u >= 3u) {
            chol[c * N + c] = 1.0;
            rhs[c] = 0.0;
        }
        let diag = chol[c * N + c];
        scaling[c] = select(0.0, inverseSqrt(diag), diag > 0.0);
    }
    workgroupBarrier();
    if (lane == 0u) {
        state = 0u;
        for (var c = 0u; c < N; c++) {
            if (!(chol[c * N + c] > 0.0)) {
                state = 2u;
            }
        }
    }
    if (workgroupUniformLoad(&state) == 2u) {
        return;
    }
    if (lane < N) {
        let c = lane;
        rhs[c] *= scaling[c];
        for (var d = c; d < N; d++) {
            chol[c * N + d] *= scaling[c] * scaling[d];
        }
    }
    workgroupBarrier();
    for (var c = 0u; c < N; c++) {
        if (lane == 0u) {
            var pivot = chol[c * N + c];
            for (var k = 0u; k < c; k++) {
                pivot -= chol[k * N + c] * chol[k * N + c];
            }
            if (pivot > 1e-10) {
                chol[c * N + c] = sqrt(pivot);
            } else {
                state = 2u;
            }
        }
        if (workgroupUniformLoad(&state) == 2u) {
            return;
        }
        if (lane > c && lane < N) {
            let d = lane;
            var e = chol[c * N + d];
            for (var k = 0u; k < c; k++) {
                e -= chol[k * N + c] * chol[k * N + d];
            }
            chol[c * N + d] = e / chol[c * N + c];
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        for (var c = 0u; c < N; c++) {
            var e = rhs[c];
            for (var k = 0u; k < c; k++) {
                e -= chol[k * N + c] * rhs[k];
            }
            rhs[c] = e / chol[c * N + c];
        }
        for (var c = N; c > 0u; c--) {
            let row = c - 1u;
            var e = rhs[row];
            for (var k = row + 1u; k < N; k++) {
                e -= chol[row * N + k] * rhs[k];
            }
            rhs[row] = e / chol[row * N + row];
        }
        for (var c = 0u; c < N; c++) {
            shared_q[c] += rhs[c] * scaling[c];
        }
        if (!(abs(shared_q[0] - u0) < 32.0 && abs(shared_q[6] - v0) < 32.0)) {
            state = 2u;
        } else if (max(abs(rhs[0] * scaling[0]), abs(rhs[6] * scaling[6])) < 1e-5) {
            state = 1u;
        }
    }
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
        var xx11 = mat3x3<f32>();
        var xx12 = mat3x3<f32>();
        var xx22 = mat3x3<f32>();
        var xy11 = mat3x3<f32>();
        var xy12 = mat3x3<f32>();
        var xy21 = mat3x3<f32>();
        var xy22 = mat3x3<f32>();
        var yy11 = mat3x3<f32>();
        var yy12 = mat3x3<f32>();
        var yy22 = mat3x3<f32>();
        var xg1 = vec3<f32>();
        var xg2 = vec3<f32>();
        var x11 = vec3<f32>();
        var x12 = vec3<f32>();
        var yg1 = vec3<f32>();
        var yg2 = vec3<f32>();
        var y11 = vec3<f32>();
        var y12 = vec3<f32>();
        var rx1 = vec3<f32>();
        var rx2 = vec3<f32>();
        var ry1 = vec3<f32>();
        var ry2 = vec3<f32>();
        var gg = 0.0;
        var g1 = 0.0;
        var ones = 0.0;
        var rg = 0.0;
        var r1 = 0.0;
        for (var k = lane; k < pixels; k += LANES) {
            let i = i32(k % params.window) - hw;
            let j = i32(k / params.window) - hw;
            let shape = shape_at(k, hw, scale);
            let s1 = vec3<f32>(shape[0], shape[1], shape[2]);
            let s2 = vec3<f32>(shape[3], shape[4], shape[5]);
            let u = dot(vec3<f32>(q[0], q[1], q[2]), s1) + dot(vec3<f32>(q[3], q[4], q[5]), s2);
            let v = dot(vec3<f32>(q[6], q[7], q[8]), s1) + dot(vec3<f32>(q[9], q[10], q[11]), s2);
            let g = right_cubic(lx + f32(i) + u, ly + f32(j) + v);
            let f = left_linear(lx + f32(i), ly + f32(j));
            let residual = f - (q[12] * g.x + q[13]);
            let jx = q[12] * g.y;
            let jy = q[12] * g.z;
            let o11 = outer(s1, s1);
            let o12 = outer(s1, s2);
            let o22 = outer(s2, s2);
            xx11 += (jx * jx) * o11;
            xx12 += (jx * jx) * o12;
            xx22 += (jx * jx) * o22;
            xy11 += (jx * jy) * o11;
            xy12 += (jx * jy) * o12;
            xy21 += (jx * jy) * transpose(o12);
            xy22 += (jx * jy) * o22;
            yy11 += (jy * jy) * o11;
            yy12 += (jy * jy) * o12;
            yy22 += (jy * jy) * o22;
            xg1 += (jx * g.x) * s1;
            xg2 += (jx * g.x) * s2;
            x11 += jx * s1;
            x12 += jx * s2;
            yg1 += (jy * g.x) * s1;
            yg2 += (jy * g.x) * s2;
            y11 += jy * s1;
            y12 += jy * s2;
            rx1 += (jx * residual) * s1;
            rx2 += (jx * residual) * s2;
            ry1 += (jy * residual) * s1;
            ry2 += (jy * residual) * s2;
            gg += g.x * g.x;
            g1 += g.x;
            ones += 1.0;
            rg += g.x * residual;
            r1 += residual;
        }
        let row = lane * STRIDE;
        for (var half = 0u; half < 2u; half++) {
            let lo = half * HALF;
        for (var i = 0u; i < 3u; i++) {
            for (var j = 0u; j < 3u; j++) {
                if (i <= j) {
                    store(row, lo, i, j, xx11[j][i]);
                    store(row, lo, i + 3u, j + 3u, xx22[j][i]);
                    store(row, lo, i + 6u, j + 6u, yy11[j][i]);
                    store(row, lo, i + 9u, j + 9u, yy22[j][i]);
                }
                store(row, lo, i, j + 3u, xx12[j][i]);
                store(row, lo, i + 6u, j + 9u, yy12[j][i]);
                store(row, lo, i, j + 6u, xy11[j][i]);
                store(row, lo, i, j + 9u, xy12[j][i]);
                store(row, lo, i + 3u, j + 6u, xy21[j][i]);
                store(row, lo, i + 3u, j + 9u, xy22[j][i]);
            }
            store(row, lo, i, 12u, xg1[i]);
            store(row, lo, i + 3u, 12u, xg2[i]);
            store(row, lo, i, 13u, x11[i]);
            store(row, lo, i + 3u, 13u, x12[i]);
            store(row, lo, i + 6u, 12u, yg1[i]);
            store(row, lo, i + 9u, 12u, yg2[i]);
            store(row, lo, i + 6u, 13u, y11[i]);
            store(row, lo, i + 9u, 13u, y12[i]);
            store_rhs(row, lo, i, rx1[i]);
            store_rhs(row, lo, 3u + i, rx2[i]);
            store_rhs(row, lo, 6u + i, ry1[i]);
            store_rhs(row, lo, 9u + i, ry2[i]);
        }
        store(row, lo, 12u, 12u, gg);
        store(row, lo, 12u, 13u, g1);
        store(row, lo, 13u, 13u, ones);
        store_rhs(row, lo, 12u, rg);
        store_rhs(row, lo, 13u, r1);
            reduce(lane, lo, min(HALF, SUMS - lo));
        }
        solve(lane, u0, v0);
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
            acc[lane * STRIDE + c] = moments[c];
        }
        reduce(lane, 0u, 5u);
        let n = f32(pixels);
        let vf = sums[2] - sums[0] * sums[0] / n;
        let vg = sums[3] - sums[1] * sums[1] / n;
        let cov = sums[4] - sums[0] * sums[1] / n;
        correlated = cov / sqrt(max(vf * vg, 1e-12)) >= params.min_correlation;
    }
    if (lane == 0u) {
        refined[2u * s] = lx + shared_q[0];
        refined[2u * s + 1u] = ly + shared_q[6];
        let moved = abs(shared_q[0] - u0) < params.max_move;
        kept[s] = select(0u, 1u, state != 2u && moved && correlated);
    }
}
