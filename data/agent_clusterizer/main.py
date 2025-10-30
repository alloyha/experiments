import asyncio
from agentic_clusterizer import clusterize_texts, CONFIG_BALANCED_HYBRID


async def main():
    from agentic_clusterizer import clusterize_texts, ClusterConfig
    
    texts = [
        "Machine learning algorithms for data analysis",
        "Web development with React and Node.js",
        "Financial modeling and risk assessment",
        "Database design and optimization"
    ]
    
    config = ClusterConfig.balanced()
    result = await clusterize_texts(texts, config=config)
    
    # Validate: All assignments should reference existing categories
    category_ids = {c.id for c in result['categories']}
    for assignment in result['assignments']:
        assert assignment.category_id in category_ids, \
            f"Invalid assignment: {assignment.category_id} not in {category_ids}"
    
    print("✅ Fix validated: All assignments reference valid categories")
    print(f"Found {len(result['categories'])} categories:")
    for cat in result['categories']:
        print(f" - {cat.id}: {cat.name} ({len([a for a in result['assignments'] if a.category_id == cat.id])} texts)")
    print("\nSummary:")
    print(f"Texts: {len(texts)}")
    print(f"Categories: {len(result['categories'])}")
    print(f"Assignments: {len(result['assignments'])}")

asyncio.run(main())