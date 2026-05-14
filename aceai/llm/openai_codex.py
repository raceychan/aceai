"""OpenAI Codex provider backed by the ChatGPT Codex Responses endpoint."""

import base64
import io
import json
import wave
from typing import Any, AsyncGenerator, BinaryIO, Callable

from openai import AsyncOpenAI
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from aceai.llm.errors import AceAIConfigurationError, AceAIRuntimeError
from aceai.llm.models import LLMInput, LLMResponse, LLMSegment, LLMStreamEvent
from aceai.llm.openai import OpenAI, OpenAIMeta, OpenAIPayload


OPENAI_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
OPENAI_CODEX_REALTIME_TRANSCRIPTION_URL = "wss://chatgpt.com/backend-api/codex/realtime"
OPENAI_CODEX_PROVIDER_NAME = "codex"
OPENAI_CODEX_CLIENT_TIMEOUT_SECONDS = 300.0
OPENAI_CODEX_REALTIME_AUDIO_RATE = 24_000
OPENAI_CODEX_REALTIME_AUDIO_CHANNELS = 1
OPENAI_CODEX_REALTIME_AUDIO_SAMPLE_WIDTH = 2
OPENAI_CODEX_REALTIME_AUDIO_CHUNK_BYTES = 32_000


class OpenAICodex(OpenAI):
    """OpenAI Codex provider using a resolved ChatGPT/Codex access token."""

    def __init__(
        self,
        *,
        api_key: str,
        default_meta: OpenAIMeta,
        base_url: str = OPENAI_CODEX_BASE_URL,
        realtime_transcription_url: str = OPENAI_CODEX_REALTIME_TRANSCRIPTION_URL,
        websocket_connect: Callable[..., Any] = connect,
        instructions: str = "",
    ):
        if api_key == "":
            raise AceAIConfigurationError("OpenAI Codex access token is required")
        self._api_key = api_key
        self._realtime_transcription_url = realtime_transcription_url
        self._websocket_connect = websocket_connect
        super().__init__(
            client=AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=OPENAI_CODEX_CLIENT_TIMEOUT_SECONDS,
            ),
            default_meta=default_meta,
            provider_name=OPENAI_CODEX_PROVIDER_NAME,
        )
        self._instructions = instructions

    async def stt(
        self,
        filename: str,
        file: BinaryIO,
        *,
        model: str,
        prompt: str | None = None,
    ) -> str:
        if prompt is not None:
            raise ValueError("Codex STT does not support transcription prompts")
        pcm = _pcm_from_wav(file.read())
        return await self._transcribe_pcm_with_realtime(
            pcm,
            model=model,
            session_id=f"aceai-stt-{filename}",
        )

    def _build_response_kwargs(self, payload: OpenAIPayload) -> dict[str, Any]:
        kwargs = super()._build_response_kwargs(payload)
        if self._instructions != "":
            kwargs["instructions"] = self._instructions
        kwargs["store"] = False
        if "max_output_tokens" in kwargs:
            del kwargs["max_output_tokens"]
        return kwargs

    async def complete(self, request: LLMInput) -> LLMResponse:
        response: LLMResponse | None = None
        async for event in self.stream(request):
            if event.event_type == "response.completed" and isinstance(
                event.response,
                LLMResponse,
            ):
                response = event.response
        if response is None:
            raise AceAIRuntimeError("OpenAI Codex stream did not complete")
        return response

    async def stream(self, request: LLMInput) -> AsyncGenerator[LLMStreamEvent, None]:
        text = ""
        async for event in super().stream(request):
            if event.event_type == "response.output_text.delta":
                if isinstance(event.text_delta, str):
                    text += event.text_delta
            elif event.event_type == "response.completed":
                response = event.response
                if (
                    isinstance(response, LLMResponse)
                    and response.text == ""
                    and text != ""
                ):
                    segments = [LLMSegment(type="text", content=text)]
                    segments.extend(response.segments)
                    patched_response = LLMResponse(
                        id=response.id,
                        model=response.model,
                        text=text,
                        tool_calls=response.tool_calls,
                        usage=response.usage,
                        segments=segments,
                        provider_meta=response.provider_meta,
                        status=response.status,
                        reasoning=response.reasoning,
                        reasoning_content=response.reasoning_content,
                    )
                    event = LLMStreamEvent(
                        event_type="response.completed",
                        response=patched_response,
                        segments=patched_response.segments,
                        provider_meta=patched_response.provider_meta,
                    )
            yield event

    async def _transcribe_pcm_with_realtime(
        self,
        pcm: bytes,
        *,
        model: str,
        session_id: str,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "x-session-id": session_id,
        }
        try:
            websocket_context = self._websocket_connect(
                self._realtime_transcription_url,
                additional_headers=headers,
            )
            async with websocket_context as websocket:
                return await self._transcribe_pcm_with_realtime_websocket(
                    websocket,
                    pcm,
                    model=model,
                )
        except InvalidStatus as exc:
            if exc.response.status_code == 403:
                raise AceAIRuntimeError(
                    "Codex subscription STT requires the Codex realtime call flow; "
                    "direct realtime websocket transcription was rejected."
                ) from exc
            raise

    async def _transcribe_pcm_with_realtime_websocket(
        self,
        websocket: Any,
        pcm: bytes,
        *,
        model: str,
    ) -> str:
        await websocket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "type": "transcription",
                        "audio": {
                            "input": {
                                "format": {
                                    "type": "audio/pcm",
                                    "rate": OPENAI_CODEX_REALTIME_AUDIO_RATE,
                                },
                                "transcription": {"model": model},
                            }
                        },
                    },
                }
            )
        )
        for offset in range(0, len(pcm), OPENAI_CODEX_REALTIME_AUDIO_CHUNK_BYTES):
            chunk = pcm[offset : offset + OPENAI_CODEX_REALTIME_AUDIO_CHUNK_BYTES]
            await websocket.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
            )
        await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        while True:
            raw_event = await websocket.recv()
            event = _realtime_event(raw_event)
            event_type = event["type"]
            if event_type == "conversation.item.input_audio_transcription.completed":
                transcript = event["transcript"]
                if type(transcript) is not str:
                    raise TypeError("Codex STT transcript must be str")
                return transcript
            if event_type == "error":
                error = event["error"]
                if type(error) is dict and type(error.get("message")) is str:
                    raise AceAIRuntimeError(error["message"])
                raise AceAIRuntimeError(json.dumps(event))


def _pcm_from_wav(audio: bytes) -> bytes:
    with wave.open(io.BytesIO(audio), "rb") as wav:
        if wav.getnchannels() != OPENAI_CODEX_REALTIME_AUDIO_CHANNELS:
            raise ValueError("Codex STT requires mono WAV audio")
        if wav.getframerate() != OPENAI_CODEX_REALTIME_AUDIO_RATE:
            raise ValueError("Codex STT requires 24 kHz WAV audio")
        if wav.getsampwidth() != OPENAI_CODEX_REALTIME_AUDIO_SAMPLE_WIDTH:
            raise ValueError("Codex STT requires 16-bit PCM WAV audio")
        if wav.getcomptype() != "NONE":
            raise ValueError("Codex STT requires uncompressed PCM WAV audio")
        return wav.readframes(wav.getnframes())


def _realtime_event(raw_event: str | bytes) -> dict[str, Any]:
    if type(raw_event) is bytes:
        raw_event = raw_event.decode("utf-8")
    event = json.loads(raw_event)
    if type(event) is not dict:
        raise TypeError("Codex realtime event must be an object")
    event_type = event["type"]
    if type(event_type) is not str:
        raise TypeError("Codex realtime event type must be str")
    return event
