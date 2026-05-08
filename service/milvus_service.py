from pymilvus import DataType, MilvusClient
from config import settings

class MilvusService:

    def __init__(self):
        self.client = MilvusClient(uri=settings.MILVUS_URI)
        self.collection = settings.MILVUS_COLLECTION
        self._ensure_collection()

    def _ensure_collection(self):
        if self.client.has_collection(self.collection):
            return
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field("id", DataType.INT64, is_primary=True)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIM)
        schema.add_field("text", DataType.VARCHAR, max_length=16000)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=64)
        schema.add_field("filename", DataType.VARCHAR, max_length=512)
        schema.add_field("chunk_seq", DataType.INT64)
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type="COSINE",)

    def insert(self,rows:list[dict]):
        if not rows:
            return
        self.client.insert(collection_name=self.collection,data=rows)