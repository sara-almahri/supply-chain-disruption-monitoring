from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import re
import html
from unidecode import unidecode

class TextProcessorInput(BaseModel):
    text: str = Field(..., description="Raw text to be cleaned.")

class TextProcessorTool(BaseTool):
    name: str = "TextProcessorTool"
    description: str = "Cleans and preprocesses extracted text for analysis."
    args_schema: type = TextProcessorInput

    def _run(self, text: str) -> str:
        if not text:
            return ""
        text = html.unescape(text)
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"\+?\d{1,3}?[-.\s]?\(?\d{1,4}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}", "", text)
        text = re.sub(r"[^a-zA-Z0-9.,!?;:\-\s]", "", text)
        text = unidecode(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @property
    def func(self):
        return self._run
