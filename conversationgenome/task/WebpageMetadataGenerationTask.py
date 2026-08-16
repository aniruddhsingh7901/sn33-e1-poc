import asyncio
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple

import bittensor as bt
from pydantic import BaseModel

from conversationgenome.llm.llm_factory import get_llm_backend
from conversationgenome.task.Task import Task


class WebpageMarkdownTaskInputData(BaseModel):
    window: List[Tuple[int, str]]
    participants: Optional[List[str]] = None

class WebpageMarkdownTaskInput(BaseModel):
    guid: str
    input_type: Literal["webpage_markdown"]
    data: WebpageMarkdownTaskInputData
    input_categories: Optional[List[str]] = None


class WebpageMetadataGenerationTask(Task):
    type: Literal["webpage_metadata_generation"] = "webpage_metadata_generation"
    input: Optional[WebpageMarkdownTaskInput] = None

    async def mine(self) -> dict[str, list]:
        llml = get_llm_backend()

        try:
            # Run all per-window-item LLM calls concurrently in a thread pool.
            # First item = main webpage; rest = enrichment search snippets.
            def _call(idx: int, content: str):
                if idx == 0:
                    return llml.website_to_metadata(
                        content,
                        generateEmbeddings=False,
                        input_categories=self.input.input_categories,
                    )
                return llml.enrichment_to_metadata(
                    content,
                    generateEmbeddings=False,
                    input_categories=self.input.input_categories,
                )

            results = await asyncio.gather(*[
                asyncio.to_thread(_call, idx, content)
                for idx, (_line_idx, content) in enumerate(self.input.data.window)
            ])

            all_tags = [r.tags for r in results if r and r.tags]

            if all_tags:
                combined_result = llml.combine_metadata_tags(all_tags, generateEmbeddings=False)
                output = {"tags": combined_result.tags if combined_result else [], "vectors": combined_result.vectors if combined_result else None}
            else:
                output = {"tags": [], "vectors": None}

        except Exception as e:
            bt.logging.error(f"Error during mining: {e}")
            raise e

        return output
