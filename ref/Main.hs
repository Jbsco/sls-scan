-- | sls-ref: the Haskell reference of the pure decode and triangulation, on GHC's own
-- libraries.
--
-- sls-ref decode-check DIR decodes the stacks `slscan decode-vectors` wrote and
-- compares the columns and masks with NumPy's. sls-ref scan-check DIR triangulates the
-- columns `slscan scan-vectors` wrote and compares the points with NumPy's.
module Main (main) where

import Data.Array.Unboxed (UArray, listArray, (!))
import Data.Bits (shiftL, (.|.))
import qualified Data.ByteString as B
import qualified Data.ByteString.Unsafe as BU
import Data.Word (Word32, Word64)
import Decode
import GHC.Float (castWord32ToFloat, castWord64ToDouble)
import Json
import System.Environment (getArgs)
import System.Exit (exitFailure)
import System.FilePath ((</>))
import System.IO (hPutStrLn, stderr)
import Text.Printf (printf)
import Triangulate

word :: B.ByteString -> Int -> Int -> Word64
word b size i = foldr (\k acc -> acc `shiftL` 8 .|. byte k) 0 [0 .. size - 1]
  where
    byte k = fromIntegral (BU.unsafeIndex b (size * i + k))

floatAt :: B.ByteString -> Int -> Float
floatAt b i = castWord32ToFloat (fromIntegral (word b 4 i) :: Word32)

doubleAt :: B.ByteString -> Int -> Double
doubleAt b i = castWord64ToDouble (word b 8 i)

u32At :: B.ByteString -> Int -> Int
u32At b i = fromIntegral (word b 4 i)

orDie :: Either String a -> IO a
orDie = either (\e -> hPutStrLn stderr e >> exitFailure) pure

int :: String -> Json -> Either String Int
int key j = round <$> (field key j >>= number)

settingsFrom :: Json -> Either String Settings
settingsFrom j =
  Settings
    <$> int "width" j
    <*> int "height" j
    <*> int "planes" j
    <*> int "steps" j
    <*> int "projector_width" j
    <*> real "pitch"
    <*> real "min_contrast"
    <*> real "min_modulation"
    <*> real "max_disagreement"
  where
    real key = field key j >>= number

decodeCheck :: FilePath -> IO ()
decodeCheck dir = do
  settings <- readFile (dir </> "meta.json") >>= orDie . (\t -> parseJson t >>= settingsFrom)
  gray <- B.readFile (dir </> "gray.bin")
  phase <- B.readFile (dir </> "phase.bin")
  columns <- B.readFile (dir </> "columns.bin")
  mask <- B.readFile (dir </> "mask.bin")
  let n = width settings * height settings
      sizes =
        [ B.length gray == 2 * planes settings * n
        , B.length phase == steps settings * n
        , B.length columns == 4 * n
        , B.length mask == n
        ]
  if and sizes then pure () else hPutStrLn stderr "vectors do not match meta.json" >> exitFailure
  let decoded = map (decodePixel settings gray phase) [0 .. n - 1]
      found = listArray (0, n - 1) (map (realToFrac . fst) decoded) :: UArray Int Float
      kept = listArray (0, n - 1) (map snd decoded) :: UArray Int Bool
      reference i = BU.unsafeIndex mask i /= 0
      disagreements = length [() | i <- [0 .. n - 1], kept ! i /= reference i]
      shared = [i | i <- [0 .. n - 1], kept ! i, reference i]
      differences = [abs (found ! i - floatAt columns i) | i <- shared]
      largest = if null differences then 0 else maximum differences
      unequal = length (filter (> 0) differences)
  printf
    "%d pixels: %d mask disagreements, %d of %d shared columns differ, largest %.6f px\n"
    n
    disagreements
    unequal
    (length shared)
    largest
  if disagreements == 0 && largest <= 1.0e-4
    then putStrLn "decode check: Haskell matches the NumPy reference"
    else putStrLn "decode check: Haskell differs from the NumPy reference" >> exitFailure

scanFrom :: Json -> Either String (Scan, Int, Int)
scanFrom j = do
  cam <- field "camera" j
  proj <- field "projector" j
  scan <-
    Scan
      <$> device cam
      <*> device proj
      <*> (field "rotation" j >>= numbers)
      <*> (field "translation" j >>= numbers)
  (,,) scan <$> int "width" cam <*> int "height" cam
  where
    device d = Device <$> (field "matrix" d >>= numbers) <*> (field "distortion" d >>= numbers)

scanCheck :: FilePath -> IO ()
scanCheck dir = do
  (scan, w, h) <- readFile (dir </> "scan.json") >>= orDie . (\t -> parseJson t >>= scanFrom)
  columns <- B.readFile (dir </> "columns.bin")
  mask <- B.readFile (dir </> "mask.bin")
  points <- B.readFile (dir </> "points.bin")
  pixels <- B.readFile (dir </> "pixels.bin")
  let n = w * h
      found =
        [ ((x, y), p)
        | i <- [0 .. n - 1]
        , BU.unsafeIndex mask i /= 0
        , let (y, x) = i `divMod` w
        , Just p <- [pointOf scan x y (realToFrac (floatAt columns i))]
        ]
      count = B.length points `div` 24
      reference = [((u32At pixels (2 * k), u32At pixels (2 * k + 1)), point k) | k <- [0 .. count - 1]]
      point k = (doubleAt points (3 * k), doubleAt points (3 * k + 1), doubleAt points (3 * k + 2))
      distance (a, b, c) (d, e, f) = sqrt ((a - d) ^ (2 :: Int) + (b - e) ^ (2 :: Int) + (c - f) ^ (2 :: Int))
      samePixels = map fst found == map fst reference
      distances = zipWith (\(_, p) (_, q) -> distance p q) found reference
      largest = maximum (0 : distances)
  printf
    "%d points from NumPy columns against %d, pixels %s, %d points differ, largest by %.3g nm\n"
    (length found)
    count
    (if samePixels then "equal" else "differ")
    (length (filter (> 0) distances))
    (1.0e9 * largest)
  if samePixels && largest <= 1.0e-8
    then putStrLn "scan check: Haskell matches the NumPy reference"
    else putStrLn "scan check: Haskell differs from the NumPy reference" >> exitFailure

main :: IO ()
main = do
  args <- getArgs
  case args of
    ["decode-check", dir] -> decodeCheck dir
    ["scan-check", dir] -> scanCheck dir
    _ -> hPutStrLn stderr "usage: sls-ref decode-check|scan-check <vector directory>" >> exitFailure
