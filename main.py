
from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel
from pydantic_core import from_json


class PredicateMatch(BaseModel):
    winteam: str
    loseteam: str
    winteam_score: int
    loseteam_score: int
    why_winteam: str
    why_loseteam: str

class PredicateMatchForChunk(BaseModel):
    winteam: Optional[str] = None
    loseteam: Optional[str] = None
    winteam_score: Optional[int] = None
    loseteam_score: Optional[int] = None
    why_winteam: Optional[str] = None
    why_loseteam: Optional[str] = None

client = genai.Client()
prompt = "本日のプロ野球の試合予想を教えてください。さらに、予想の理由も教えてください。"

response_stream = client.models.generate_content_stream(
    model="gemini-2.5-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=PredicateMatch,
    ),
)

buffer = ""
for chunk in response_stream:
    if chunk.candidates[0].content.parts:
        buffer += chunk.candidates[0].content.parts[0].text
        partial_json = PredicateMatchForChunk.model_validate(from_json(buffer, allow_partial=True))
        #ここでparial_jsonをフロントエンドに返す

complete_match = PredicateMatch.model_validate(from_json(buffer))

#ここでcomplete_matchをフロントエンドに返す