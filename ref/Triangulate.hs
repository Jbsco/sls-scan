-- | The triangulation of `slscan/triangulate.py` written again: each camera ray meets
-- the plane of its projector column, taken from the undistorted projector point at the
-- row where the previous estimate lands, over three iterations.
module Triangulate
  ( Device (..)
  , Scan (..)
  , undistort
  , project
  , pointOf
  ) where

-- | A pinhole device: row-major matrix and distortion k1, k2, p1, p2, k3.
data Device = Device
  { matrix :: [Double]
  , distortion :: [Double]
  }

-- | Camera, projector and the camera-to-projector rotation (row-major) and translation.
data Scan = Scan
  { camera :: Device
  , projector :: Device
  , rotation :: [Double]
  , translation :: [Double]
  }

-- | Undistorted normalized coordinates of a pixel by the five fixed iterations of
-- `cv2.undistortPoints`.
undistort :: Device -> Double -> Double -> (Double, Double)
undistort d u v = go (5 :: Int) x0 y0
  where
    (fx, cx, fy, cy) = intrinsics d
    (k1, k2, p1, p2, k3) = coefficients d
    x0 = (u - cx) * (1 / fx)
    y0 = (v - cy) * (1 / fy)
    go 0 x y = (x, y)
    go k x y =
      let r2 = x * x + y * y
          icdist = 1 / (1 + ((k3 * r2 + k2) * r2 + k1) * r2)
          dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
          dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
       in if icdist < 0 then (x0, y0) else go (k - 1) ((x0 - dx) * icdist) ((y0 - dy) * icdist)

-- | Pixel of a point in the device's frame, with distortion, as `cv2.projectPoints`.
project :: Device -> (Double, Double, Double) -> (Double, Double)
project d (px, py, pz) = (fx * xd + cx, fy * yd + cy)
  where
    (fx, cx, fy, cy) = intrinsics d
    (k1, k2, p1, p2, k3) = coefficients d
    x = px / pz
    y = py / pz
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

intrinsics :: Device -> (Double, Double, Double, Double)
intrinsics d = case matrix d of
  [fx, _, cx, _, fy, cy, _, _, _] -> (fx, cx, fy, cy)
  _ -> error "a device matrix has nine elements"

coefficients :: Device -> (Double, Double, Double, Double, Double)
coefficients d = case distortion d of
  [k1, k2, p1, p2, k3] -> (k1, k2, p1, p2, k3)
  _ -> error "a device has five distortion coefficients"

-- | The point in the camera frame for camera pixel (x, y) at projector column c, when it
-- lies in front of the camera.
pointOf :: Scan -> Int -> Int -> Double -> Maybe (Double, Double, Double)
pointOf s x y c = if pz > 0 then Just p else Nothing
  where
    (xn, yn) = undistort (camera s) (fromIntegral x) (fromIntegral y)
    (r0, r1, r2, r3, r4, r5, r6, r7, r8) = case rotation s of
      [a, b, cc, d, e, f, g, h, i] -> (a, b, cc, d, e, f, g, h, i)
      _ -> error "a rotation has nine elements"
    (t0, t1, t2) = case translation s of
      [a, b, cc] -> (a, b, cc)
      _ -> error "a translation has three elements"
    rot (a, b, cc) = (r0 * a + r1 * b + r2 * cc, r3 * a + r4 * b + r5 * cc, r6 * a + r7 * b + r8 * cc)
    (q0, _, q2) = rot (xn, yn, 1)
    (_, _, _, cyProjector) = intrinsics (projector s)
    step v =
      let (u', _) = undistort (projector s) c v
          scale = negate (t0 - u' * t2) / (q0 - u' * q2)
          point = (xn * scale, yn * scale, scale)
          (a, b, cc) = rot point
          (_, v') = project (projector s) (a + t0, b + t1, cc + t2)
       in (v', point)
    (_, p@(_, _, pz)) = step (fst (step (fst (step cyProjector))))
