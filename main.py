from fastapi import FastAPI, Query, HTTPException, StreamingResponse
import json

app=FastAPI()

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")
    conversation_id: str | None = Field(None, description="会话 ID，不传则新建")

@app.post("/api/v1/chat")
async def chat(req:ChatRequest):
    def event_stream():
        for ev in orchestrator.chat_stream(req.question,req.conversation_id):
            yield _sse(ev["event"],ev["data"])
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

def _sse(event:str,data:dict):
    payload=json.dumps(data,ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")