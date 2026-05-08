from service.llm_service import LLMService
from service.embedding_service import EmbeddingService
from service.milvus_service import MilvusService
from collections import defaultdict
import uuid
import json
import asyncio
import time
from config import settings
import logging
import os
import sys
class Orchestrator:
    def __init__(self,llm:LLMService,embedding:EmbeddingService,milvus:MilvusService):
        self.llm=llm
        self.embedding=embedding
        self.milvus=milvus
        self.sessions=defaultdict(list)
        self.top_k=settings.TOP_K

    def chat_stream(self,question:str,conversation_id:str|None=None):
        conv_id=conversation_id or uuid.uuid4().hex
        history=self.sessions[conv_id]
        contexts=self.retrieve(question)
        messages=self.build_messages(question,history,contexts)
        yield {"event":"meta","data":{"conversation_id":conv_id}}
        yield {"event":"retrieval","data":{"citations":[{"index":i+1,"filename":c["filename"],"doc_id":c["doc_id"],"score":c["score"],"snippet":c["text"][:120]} for i,c in enumerate(contexts)]}}
        full_answer=""
        try:
            for delta in self.llm.chat_stream(messages):
                full_answer+=delta
                yield {"event":"content","data":{"delta":delta}}
        except Exception as e:
            yield {"event":"error","data":{"message":str(e)}}
            yield {"event":"done","data":{"finish_reason":"error"}}
            return
    
    def retrieve(self,question:str):
        vector = self.embedding.embed_one(question)
        hits = self.milvus.search(vector,top_k=self.top_k)
        out:list[dict]=[]
        for h in hits:
            entity = h.get("entity",{})
            out.append({
                "text":entity.get("text",""),
                "filename":entity.get("filename",""),
                "doc_id":entity.get("doc_id",""),
                "score":float(h.get("distance",0.0)),
            })
        return out

    def build_messages(self,question:str,history:list[dict],contexts:list[dict]):
        if contexts:
            ctx_text="\n\n".join(f"[{i+1}] (来源：{c['filename']})\n{c['text']}" for i,c in enumerate(contexts))
        else:
            ctx_text="（无相关资料）"
        messages:list[dict]=[{"role":"system","content":SYSTEM_PROMPT}]
        messages.extend(history[-6:])
        messages.append({
            "role":"user",
            "content":f"【参考资料】\n{ctx_text}\n\n【用户问题】\n{question}"
        })
        return messages
