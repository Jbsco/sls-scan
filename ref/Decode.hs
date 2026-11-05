-- | The decode of `slscan/decode.py` and `slscan/patterns.py` written again, for a
-- second implementation to check the NumPy reference and the GPU kernels against.
module Decode
  ( Settings (..)
  , decodePixel
  , wrap
  ) where

import Data.Bits (shiftL, shiftR, xor, (.|.))
import qualified Data.ByteString as B
import qualified Data.ByteString.Unsafe as BU

foreign import ccall unsafe "math.h hypot" c_hypot :: Double -> Double -> Double

-- | Camera size, the stacks' plane and step counts, and the decode thresholds.
data Settings = Settings
  { width :: !Int
  , height :: !Int
  , planes :: !Int
  , steps :: !Int
  , projectorWidth :: !Int
  , pitch :: !Double
  , minContrast :: !Double
  , minModulation :: !Double
  , maxDisagreement :: !Double
  }

saturated :: Double
saturated = 255

-- | A wrapped phase in [0, 2 pi), as `np.mod` of a value within one turn of zero.
wrap :: Double -> Double
wrap x
  | x > 0 = x
  | x < 0 = x + 2 * pi
  | otherwise = 0

grayToBinary :: Int -> Int
grayToBinary g = go g (g `shiftR` 1)
  where
    go n 0 = n
    go n s = go (n `xor` s) (s `shiftR` 1)

-- | The projector column of pixel p and whether it decoded, from Gray code and phase
-- stacks laid out plane-major, then rows, then columns.
decodePixel :: Settings -> B.ByteString -> B.ByteString -> Int -> (Double, Bool)
decodePixel s gray phase p = (column, kept)
  where
    n = width s * height s
    sample v k = fromIntegral (BU.unsafeIndex v (k * n + p)) :: Double
    pairs = [(sample gray (2 * k), sample gray (2 * k + 1)) | k <- [0 .. planes s - 1]]
    contrasted = all (\(a, b) -> abs (a - b) >= minContrast s) pairs
    code = foldl (\acc (a, b) -> (acc `shiftL` 1) .|. (if a - b > 0 then 1 else 0)) 0 pairs
    grayColumn = grayToBinary code
    samples = [sample phase i | i <- [0 .. steps s - 1]]
    angles = [2 * pi * fromIntegral i / fromIntegral (steps s) | i <- [0 .. steps s - 1]]
    numerator = foldl1 (+) (zipWith (\v a -> v * sin a) samples angles)
    denominator = foldl1 (+) (zipWith (\v a -> v * cos a) samples angles)
    modulation = 2 / fromIntegral (steps s) * c_hypot numerator denominator
    wrapped = wrap (negate (atan2 numerator denominator))
    fraction = wrapped * pitch s / (2 * pi)
    g = fromIntegral grayColumn :: Double
    column = fraction + pitch s * fromIntegral (round ((g - fraction) / pitch s) :: Integer)
    kept =
      contrasted
        && grayColumn < projectorWidth s
        && maximum samples < saturated
        && modulation >= minModulation s
        && abs (column - g) <= maxDisagreement s
        && column >= -0.5
        && column <= fromIntegral (projectorWidth s) - 0.5
