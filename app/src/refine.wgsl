struct Params {
    width: u32,
    height: u32,
    window: u32,
    iterations: u32,
    count: u32,
    pixel_stride: u32,
    sample_stride: u32,
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

@compute @workgroup_size(64)
fn refine(@builtin(global_invocation_id) id: vec3<u32>) {
    let s = id.x + id.y * params.sample_stride;
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
    var q: array<f32, 14>;
    q[0] = u0;
    q[1] = (samples[base + 4u] - 1.0) * scale;
    q[2] = samples[base + 5u] * scale;
    q[6] = v0;
    q[7] = samples[base + 6u] * scale;
    q[8] = (samples[base + 7u] - 1.0) * scale;
    q[12] = 1.0;
    var ok = true;
    for (var it = 0u; it < params.iterations; it++) {
        // Explicit initializers zero these sums on every iteration.
        var m = array<f32, 196>();
        var r = array<f32, 14>();
        for (var j = -hw; j <= hw; j++) {
            for (var i = -hw; i <= hw; i++) {
                let a = f32(i) / scale;
                let b = f32(j) / scale;
                let shape = array<f32, 6>(1.0, a, b, 0.5 * a * a, a * b, 0.5 * b * b);
                var u = 0.0;
                var v = 0.0;
                for (var k = 0u; k < 6u; k++) {
                    u += q[k] * shape[k];
                    v += q[6u + k] * shape[k];
                }
                let g = right_cubic(lx + f32(i) + u, ly + f32(j) + v);
                let f = left_linear(lx + f32(i), ly + f32(j));
                let residual = f - (q[12] * g.x + q[13]);
                var jac = array<f32, 14>();
                for (var k = 0u; k < 6u; k++) {
                    jac[k] = q[12] * g.y * shape[k];
                    jac[6u + k] = q[12] * g.z * shape[k];
                }
                jac[12] = g.x;
                jac[13] = 1.0;
                for (var c = 0u; c < N; c++) {
                    r[c] += jac[c] * residual;
                    for (var d = c; d < N; d++) {
                        m[c * N + d] += jac[c] * jac[d];
                    }
                }
            }
        }
        var scaling = array<f32, 14>();
        for (var c = 0u; c < N; c++) {
            if (!(m[c * N + c] > 0.0)) {
                ok = false;
            }
            scaling[c] = select(0.0, inverseSqrt(m[c * N + c]), ok);
        }
        if (!ok) {
            break;
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
                ok = false;
                break;
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
        if (!ok) {
            break;
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
            q[c] += r[c] * scaling[c];
        }
        if (!(abs(q[0] - u0) < 4.0 && abs(q[6] - v0) < 4.0)) {
            ok = false;
            break;
        }
        if (max(abs(r[0] * scaling[0]), abs(r[6] * scaling[6])) < 1e-5) {
            break;
        }
    }
    refined[2u * s] = lx + q[0];
    refined[2u * s + 1u] = ly + q[6];
    kept[s] = select(0u, 1u, ok && abs(q[0] - u0) < 1.0);
}
