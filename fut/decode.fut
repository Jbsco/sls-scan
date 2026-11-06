-- The decode of slscan/decode.py: Gray code and phase-shift stacks to a projector column
-- and a mask per camera pixel, written once for any float type. Stacks are plane-major,
-- each plane a flat row of camera pixels.

module decoder (R: real) = {
  type t = R.t

  -- A wrapped phase in [0, 2 pi), as np.mod of a value within one turn of zero.
  def wrap (x: t) : t =
    if R.(x > i64 0) then x else if R.(x < i64 0) then R.(x + i64 2 * pi) else R.i64 0

  def gray_to_binary (g: i64) : i64 =
    let (n, _) = loop (n, s) = (g, g >> 1) while s != 0 do (n ^ s, s >> 1)
    in n

  def pixel [m] [s]
            (gray: [m]u8)
            (phase: [s]u8)
            (projector_width: i64)
            (pitch: t)
            (min_contrast: t)
            (min_modulation: t)
            (max_disagreement: t) : (f32, u8) =
    let (code, contrasted) =
      loop (code, ok) = (0i64, true)
      for k < m / 2 do
        let d = f32.u8 gray[2 * k] - f32.u8 gray[2 * k + 1]
        let c = R.f32 (f32.abs d)
        in ((code << 1) | (if d > 0 then 1 else 0), ok && R.(c >= min_contrast))
    let column = gray_to_binary code
    let (num, den) =
      loop (num, den) = (R.i64 0, R.i64 0)
      for i < s do
        let v = R.u8 phase[i]
        let a = R.(i64 2 * pi * i64 i / i64 s)
        in if i == 0
           then (R.(v * sin a), R.(v * cos a))
           else (R.(num + v * sin a), R.(den + v * cos a))
    let modulation = R.(i64 2 / i64 s * hypot num den)
    let fraction = R.(wrap (neg (atan2 num den)) * pitch / (i64 2 * pi))
    let g = R.i64 column
    let u = R.(fraction + pitch * round ((g - fraction) / pitch))
    let kept =
      contrasted && column < projector_width && u8.maximum phase < 255
      && R.(modulation >= min_modulation)
      && R.(abs (u - g) <= max_disagreement)
      && R.(u >= f64 (-0.5))
      && R.(u <= i64 projector_width - f64 0.5)
    in (f32.f64 (R.to_f64 u), u8.bool kept)

  def decode [m] [s] [n]
             (gray: [m][n]u8)
             (phase: [s][n]u8)
             (projector_width: i64)
             (pitch: t)
             (min_contrast: t)
             (min_modulation: t)
             (max_disagreement: t) : ([n]f32, [n]u8) =
    unzip (map2 (\g p ->
                   pixel g
                         p
                         projector_width
                         pitch
                         min_contrast
                         min_modulation
                         max_disagreement)
                (transpose gray)
                (transpose phase))
}

module d64 = decoder f64
module d32 = decoder f32

entry decode_f64 [m] [s] [n]
                 (gray: [m][n]u8)
                 (phase: [s][n]u8)
                 (projector_width: i64)
                 (pitch: f64)
                 (min_contrast: f64)
                 (min_modulation: f64)
                 (max_disagreement: f64) : ([n]f32, [n]u8) =
  d64.decode gray phase projector_width pitch min_contrast min_modulation max_disagreement

entry decode_f32 [m] [s] [n]
                 (gray: [m][n]u8)
                 (phase: [s][n]u8)
                 (projector_width: i64)
                 (pitch: f32)
                 (min_contrast: f32)
                 (min_modulation: f32)
                 (max_disagreement: f32) : ([n]f32, [n]u8) =
  d32.decode gray phase projector_width pitch min_contrast min_modulation max_disagreement
