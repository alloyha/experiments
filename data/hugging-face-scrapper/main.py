from huggingface_hub import HfApi
import csv

api = HfApi()

# You can try a very large limit, but beware of memory and API rate limits
models = list(api.list_models(limit=1))

with open("hf_models.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "provider", "model_name", "task", "created_at", "downloads", "downloads_all_time", "likes"
    ])
    for m in models:
        provider, name = m.id.split("/") if "/" in m.id else ("N/A", m.id)
        writer.writerow([
            provider,
            name,
            m.pipeline_tag or "",
            m.created_at.isoformat() if m.created_at else "",
            getattr(m, "downloads", 0),
            getattr(m, "downloads_all_time", 0),
            getattr(m, "likes", 0),
        ])

print(models)
