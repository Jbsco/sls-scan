struct Params {
    nx: u32,
    ny: u32,
    nz: u32,
    stride: u32,
    grid_width: u32,
    grid_height: u32,
    step: f32,
    window: f32,
    lower_x: f32,
    lower_y: f32,
    lower_z: f32,
    voxel: f32,
    truncation: f32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
    m0: f32,
    m1: f32,
    m2: f32,
    m3: f32,
    m4: f32,
    m5: f32,
    m6: f32,
    m7: f32,
    m8: f32,
    b0: f32,
    b1: f32,
    b2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grid: array<f32>;
@group(0) @binding(2) var<storage, read_write> tsdf: array<f32>;
@group(0) @binding(3) var<storage, read_write> weight: array<f32>;

fn depth_at(x: u32, y: u32) -> f32 {
    return grid[y * params.grid_width + x];
}

// One voxel per invocation: fold its truncated distance to the grid's surface into its mean.
@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.y * params.stride + id.x;
    if (i >= params.nx * params.ny * params.nz) {
        return;
    }
    let iz = i % params.nz;
    let iy = (i / params.nz) % params.ny;
    let ix = i / (params.nz * params.ny);
    let wx = params.lower_x + params.voxel * f32(ix);
    let wy = params.lower_y + params.voxel * f32(iy);
    let wz = params.lower_z + params.voxel * f32(iz);
    let x = params.m0 * wx + params.m1 * wy + params.m2 * wz + params.b0;
    let y = params.m3 * wx + params.m4 * wy + params.m5 * wz + params.b1;
    let z = params.m6 * wx + params.m7 * wy + params.m8 * wz + params.b2;
    if (z <= 1e-6) {
        return;
    }
    let gx = (params.fx * x / z + params.cx - params.window) / params.step;
    let gy = (params.fy * y / z + params.cy - params.window) / params.step;
    let fx0 = floor(gx);
    let fy0 = floor(gy);
    if (fx0 < 0.0 || fy0 < 0.0 || fx0 >= f32(params.grid_width - 1u)
        || fy0 >= f32(params.grid_height - 1u)) {
        return;
    }
    let x0 = u32(fx0);
    let y0 = u32(fy0);
    let d00 = depth_at(x0, y0);
    let d10 = depth_at(x0 + 1u, y0);
    let d01 = depth_at(x0, y0 + 1u);
    let d11 = depth_at(x0 + 1u, y0 + 1u);
    let low = min(min(d00, d10), min(d01, d11));
    if (low <= 0.0 || max(max(d00, d10), max(d01, d11)) - low > params.truncation) {
        return;
    }
    let tx = gx - fx0;
    let ty = gy - fy0;
    let depth = (d00 * (1.0 - tx) + d10 * tx) * (1.0 - ty) + (d01 * (1.0 - tx) + d11 * tx) * ty;
    let sdf = depth - z;
    if (sdf <= -params.truncation) {
        return;
    }
    let value = min(1.0, sdf / params.truncation);
    let w = weight[i];
    tsdf[i] = (tsdf[i] * w + value) / (w + 1.0);
    weight[i] = w + 1.0;
}
