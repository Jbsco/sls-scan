-- | A small JSON reader for the vector files `slscan` writes.
module Json
  ( Json (..)
  , parseJson
  , field
  , number
  , numbers
  ) where

import Data.Char (isDigit, isSpace)

data Json
  = JNull
  | JBool Bool
  | JNumber Double
  | JString String
  | JArray [Json]
  | JObject [(String, Json)]

parseJson :: String -> Either String Json
parseJson text = case value (skip text) of
  Right (j, rest) | all isSpace rest -> Right j
  Right _ -> Left "trailing text after JSON value"
  Left e -> Left e

skip :: String -> String
skip = dropWhile isSpace

value :: String -> Either String (Json, String)
value ('{' : rest) = object (skip rest) []
value ('[' : rest) = array (skip rest) []
value ('"' : rest) = do
  (s, after) <- str rest
  pure (JString s, skip after)
value ('t' : 'r' : 'u' : 'e' : rest) = Right (JBool True, skip rest)
value ('f' : 'a' : 'l' : 's' : 'e' : rest) = Right (JBool False, skip rest)
value ('n' : 'u' : 'l' : 'l' : rest) = Right (JNull, skip rest)
value text =
  let (digits, rest) = span (\c -> isDigit c || c `elem` "+-.eE") text
   in case reads (normalize digits) of
        [(d, "")] -> Right (JNumber d, skip rest)
        _ -> Left ("bad JSON value at " ++ take 20 text)
  where
    normalize ds = case ds of
      '-' : more -> '-' : normalize more
      '.' : _ -> '0' : ds
      _ -> case break (== '.') ds of
        (whole, '.' : frac@(c : _)) | isDigit c -> whole ++ "." ++ frac
        (whole, ".") -> whole
        _ -> ds

str :: String -> Either String (String, String)
str = go []
  where
    go acc ('"' : rest) = Right (reverse acc, rest)
    go acc ('\\' : c : rest) = go (c : acc) rest
    go acc (c : rest) = go (c : acc) rest
    go _ [] = Left "unterminated JSON string"

object :: String -> [(String, Json)] -> Either String (Json, String)
object ('}' : rest) acc = Right (JObject (reverse acc), skip rest)
object ('"' : rest) acc = do
  (key, after) <- str rest
  case skip after of
    ':' : more -> do
      (v, after') <- value (skip more)
      case after' of
        ',' : next -> object (skip next) ((key, v) : acc)
        '}' : next -> Right (JObject (reverse ((key, v) : acc)), skip next)
        _ -> Left ("expected , or } after " ++ key)
    _ -> Left ("expected : after " ++ key)
object text _ = Left ("bad JSON object at " ++ take 20 text)

array :: String -> [Json] -> Either String (Json, String)
array (']' : rest) acc = Right (JArray (reverse acc), skip rest)
array text acc = do
  (v, after) <- value text
  case after of
    ',' : next -> array (skip next) (v : acc)
    ']' : next -> Right (JArray (reverse (v : acc)), skip next)
    _ -> Left "expected , or ] in JSON array"

-- | The member of an object at a key.
field :: String -> Json -> Either String Json
field key (JObject members) = maybe (Left ("missing " ++ key)) Right (lookup key members)
field key _ = Left ("not an object where " ++ key ++ " was expected")

number :: Json -> Either String Double
number (JNumber d) = Right d
number _ = Left "expected a number"

numbers :: Json -> Either String [Double]
numbers (JArray xs) = mapM number xs
numbers _ = Left "expected an array of numbers"
