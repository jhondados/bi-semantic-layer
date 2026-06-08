"""AI semantic layer generator."""
from langchain_google_vertexai import ChatVertexAI
from google.cloud import bigquery
import yaml
from typing import Dict

class SemanticLayerGenerator:
    def __init__(self, project_id: str):
        self.bq = bigquery.Client(project=project_id)
        self.llm = ChatVertexAI(model_name="gemini-1.5-pro-002")

    def get_table_schema(self, dataset: str, table: str) -> str:
        t = self.bq.get_table(f"{dataset}.{table}")
        return "\n".join([f"  - {f.name}: {f.field_type} ({f.description or 'no description'})" for f in t.schema])

    def generate_metric(self, table: str, schema: str, metric_name: str) -> Dict:
        prompt = f"""Given this BigQuery table schema:
{schema}

Generate a dbt Semantic Layer metric definition for: {metric_name}
Return valid YAML with: name, type, label, description, type_params, filter (if needed)
Use MetricFlow syntax (dbt 1.6+)."""
        response = self.llm.invoke(prompt).content
        if "```yaml" in response: response = response.split("```yaml")[1].split("```")[0]
        try: return yaml.safe_load(response)
        except: return {"raw": response}

    def generate_all_metrics(self, dataset: str, table: str) -> list:
        schema = self.get_table_schema(dataset, table)
        prompt = f"""For this BigQuery schema:\n{schema}\nList 10 business metrics to track (just names, one per line)"""
        metric_names = [l.strip("- 0123456789. ") for l in self.llm.invoke(prompt).content.split("\n") if l.strip()]
        return [self.generate_metric(table, schema, m) for m in metric_names[:10]]
