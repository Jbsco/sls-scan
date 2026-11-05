-- | sls-ref: the Haskell reference of the pure decode core, on GHC's own libraries.
--
-- sls-ref decode-check DIR decodes the stacks `slscan decode-vectors` wrote and
-- compares the columns and masks with NumPy's.
module Main (main) where

import Data.Array.Unboxed (UArray, listArray, (!))
import Data.Bits (shiftL, (.|.))
import qualified Data.ByteString as B
import qualified Data.ByteString.Unsafe as BU
import Data.Word (Word32, Word64)
import Decode
import GHC.Float (castWord32ToFloat)
import Json
import System.Environment (getArgs)
import System.Exit (exitFailure)
import System.FilePath ((</>))
import System.IO (hPutStrLn, stderr)
import Text.Printf (printf)

word :: B.ByteString -> Int -> Int -> Word64
word b size i = foldr (\k acc -> acc `shiftL` 8 .|. byte k) 0 [0 .. size - 1]
  where
    byte k = fromIntegral (BU.unsafeIndex b (size * i + k))

floatAt :: B.ByteString -> Int -> Float
floatAt b i = castWord32ToFloat (fromIntegral (word b 4 i) :: Word32)

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

main :: IO ()
main = do
  args <- getArgs
  case args of
    ["decode-check", dir] -> decodeCheck dir
    _ -> hPutStrLn stderr "usage: sls-ref decode-check <vector directory>" >> exitFailure
