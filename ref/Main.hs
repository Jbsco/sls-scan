-- | sls-ref: the Haskell reference of the pure decode core, on GHC's own libraries.
--
-- sls-ref decode-check DIR decodes the stacks `slscan decode-vectors` wrote and
-- compares the columns and masks with NumPy's.
module Main (main) where

import Data.Array.Unboxed (UArray, listArray, (!))
import Data.Bits (shiftL, (.|.))
import qualified Data.ByteString as B
import qualified Data.ByteString.Unsafe as BU
import Data.Char (isSpace)
import Data.Word (Word32)
import Decode
import GHC.Float (castWord32ToFloat)
import System.Environment (getArgs)
import System.Exit (exitFailure)
import System.FilePath ((</>))
import System.IO (hPutStrLn, stderr)
import Text.Printf (printf)
import Text.Read (readMaybe)

-- | The numeric fields of a flat JSON object, as `json.dump` writes `meta.json`.
fields :: String -> [(String, String)]
fields text = [pair item | item <- splitOn ',' (strip (trimBraces (strip text)))]
  where
    trimBraces = takeWhile (/= '}') . drop 1 . dropWhile (/= '{')
    pair item =
      let (key, rest) = break (== ':') item
       in (filter (/= '"') (strip key), strip (drop 1 rest))
    strip = dropWhile isSpace . reverse . dropWhile isSpace . reverse
    splitOn c xs = case break (== c) xs of
      (a, []) -> [a]
      (a, _ : b) -> a : splitOn c b

settingsFrom :: String -> Either String Settings
settingsFrom text =
  Settings
    <$> number "width"
    <*> number "height"
    <*> number "planes"
    <*> number "steps"
    <*> number "projector_width"
    <*> number "pitch"
    <*> number "min_contrast"
    <*> number "min_modulation"
    <*> number "max_disagreement"
  where
    table = fields text
    number :: Read a => String -> Either String a
    number key = maybe (Left ("meta.json lacks " ++ key)) Right (lookup key table >>= readMaybe)

floatAt :: B.ByteString -> Int -> Float
floatAt b i =
  let at k = fromIntegral (BU.unsafeIndex b (4 * i + k)) :: Word32
   in castWord32ToFloat (at 0 .|. at 1 `shiftL` 8 .|. at 2 `shiftL` 16 .|. at 3 `shiftL` 24)

decodeCheck :: FilePath -> IO ()
decodeCheck dir = do
  meta <- readFile (dir </> "meta.json")
  settings <- either (\e -> hPutStrLn stderr e >> exitFailure) pure (settingsFrom meta)
  gray <- B.readFile (dir </> "gray.bin")
  phase <- B.readFile (dir </> "phase.bin")
  columns <- B.readFile (dir </> "columns.bin")
  mask <- B.readFile (dir </> "mask.bin")
  let n = width settings * height settings
      decoded = map (decodePixel settings gray phase) [0 .. n - 1]
      found = listArray (0, n - 1) (map (realToFrac . fst) decoded) :: UArray Int Float
      kept = listArray (0, n - 1) (map snd decoded) :: UArray Int Bool
      reference i = BU.unsafeIndex mask i /= 0
      disagreements = length [() | i <- [0 .. n - 1], kept ! i /= reference i]
      shared = [i | i <- [0 .. n - 1], kept ! i, reference i]
      differences = [abs (found ! i - floatAt columns i) | i <- shared]
      largest = if null differences then 0 else maximum differences
      unequal = length (filter (> 0) differences)
  let sizes =
        [ B.length gray == 2 * planes settings * n
        , B.length phase == steps settings * n
        , B.length columns == 4 * n
        , B.length mask == n
        ]
  if and sizes then pure () else hPutStrLn stderr "vectors do not match meta.json" >> exitFailure
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
