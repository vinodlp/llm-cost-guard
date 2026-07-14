"""
Load test for LLM Cost Guard.
Sends 50 requests across simple, moderate and complex prompts.
Prints a cost saving summary at the end.
"""
import asyncio
import httpx
import json
import time

BASE_URL = "http://127.0.0.1:8000"

# 50 prompts across three complexity levels
PROMPTS = [
    # Simple — expect CHEAP tier (Haiku)
    ("simple", "Hi, how are you?"),
    ("simple", "What is the capital of France?"),
    ("simple", "Who is the CEO of Apple?"),
    ("simple", "What is 2 + 2?"),
    ("simple", "Thanks for your help"),
    ("simple", "What is Python?"),
    ("simple", "When was Google founded?"),
    ("simple", "What is HTTP?"),
    ("simple", "Define machine learning"),
    ("simple", "What is a REST API?"),
    ("simple", "Who invented the telephone?"),
    ("simple", "What is the speed of light?"),
    ("simple", "What is JSON?"),
    ("simple", "What is a database?"),
    ("simple", "What does API stand for?"),
    ("simple", "What is the capital of Japan?"),
    ("simple", "What is CSS?"),
    ("simple", "Who wrote Harry Potter?"),
    ("simple", "What is RAM?"),
    ("simple", "What is the internet?"),

    # Moderate — expect BALANCED tier (Sonnet)
    ("moderate", "Explain how LLMs work"),
    ("moderate", "Compare SQL vs NoSQL databases"),
    ("moderate", "Explain the difference between TCP and UDP"),
    ("moderate", "How does a neural network learn?"),
    ("moderate", "Explain how Docker containers work"),
    ("moderate", "Compare REST vs GraphQL APIs"),
    ("moderate", "How does OAuth 2.0 work?"),
    ("moderate", "Explain microservices architecture"),
    ("moderate", "How does Kubernetes orchestrate containers?"),
    ("moderate", "Explain the CAP theorem"),

    # Complex — expect CAPABLE tier (Opus)
    ("complex", "Design a distributed rate limiting system for a multi-tenant SaaS platform step by step"),
    ("complex", "Analyze and compare event sourcing vs CQRS architecture patterns with trade-offs"),
    ("complex", "Design a real-time recommendation engine for an e-commerce platform like Noon"),
    ("complex", "Architect a fault-tolerant payment processing system with distributed transactions"),
    ("complex", "Design a scalable data pipeline for processing 1 million events per second"),
    ("complex", "Analyze the trade-offs between microservices and monolithic architecture for a startup"),
    ("complex", "Design a distributed cache system with consistency guarantees step by step"),
    ("complex", "Prove and explain the Byzantine fault tolerance problem in distributed systems"),
    ("complex", "Design a multi-region active-active database architecture with conflict resolution"),
    ("complex", "Architect a real-time fraud detection system for a fintech platform step by step"),
]


async def send_request(
    client: httpx.AsyncClient,
    caller_id: str,
    prompt: str,
    expected_tier: str,
    index: int,
) -> dict:
    try:
        resp = await client.post(
            f"{BASE_URL}/v1/messages",
            headers={
                "Content-Type":      "application/json",
                "x-caller-id":       caller_id,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model":      "claude-opus-4-6",
                "max_tokens": 50,
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=30.0,
        )
        data = resp.json()
        routed_model = data.get("model", "unknown")

        tier = "CHEAP" if "haiku" in routed_model else \
               "BALANCED" if "sonnet" in routed_model else "CAPABLE"

        correct = (
            (expected_tier == "simple"   and tier == "CHEAP") or
            (expected_tier == "moderate" and tier == "BALANCED") or
            (expected_tier == "complex"  and tier == "CAPABLE")
        )

        print(
            f"[{index:02d}] {expected_tier:8} → {tier:8} "
            f"{'✓' if correct else '✗'} | {prompt[:50]}"
        )
        return {
            "expected": expected_tier,
            "tier":     tier,
            "correct":  correct,
            "model":    routed_model,
        }

    except Exception as e:
        print(f"[{index:02d}] ERROR: {e}")
        return {"expected": expected_tier, "tier": "ERROR", "correct": False}


async def run_load_test():
    print("=" * 65)
    print("LLM Cost Guard — Load Test")
    print("Sending 50 requests across simple, moderate, complex prompts")
    print("=" * 65)
    print()

    callers = ["search-service", "cart-service", "recommendation-service"]
    results = []
    start   = time.time()

    async with httpx.AsyncClient() as client:
        # Send in batches of 5 to avoid overwhelming the server
        for i in range(0, len(PROMPTS), 5):
            batch = PROMPTS[i:i+5]
            tasks = [
                send_request(
                    client,
                    callers[j % len(callers)],
                    prompt,
                    expected,
                    i + j + 1,
                )
                for j, (expected, prompt) in enumerate(batch)
            ]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)
            await asyncio.sleep(0.5)  # small pause between batches

    elapsed = round(time.time() - start, 1)

    # Summary
    total    = len(results)
    correct  = sum(1 for r in results if r["correct"])
    cheap    = sum(1 for r in results if r["tier"] == "CHEAP")
    balanced = sum(1 for r in results if r["tier"] == "BALANCED")
    capable  = sum(1 for r in results if r["tier"] == "CAPABLE")
    errors   = sum(1 for r in results if r["tier"] == "ERROR")

    print()
    print("=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"Total requests:     {total}")
    print(f"Completed in:       {elapsed}s")
    print(f"Routing accuracy:   {correct}/{total} ({round(correct/total*100)}%)")
    print()
    print("Tier distribution:")
    print(f"  CHEAP    (Haiku):  {cheap:3} requests  ({round(cheap/total*100)}%)")
    print(f"  BALANCED (Sonnet): {balanced:3} requests  ({round(balanced/total*100)}%)")
    print(f"  CAPABLE  (Opus):   {capable:3} requests  ({round(capable/total*100)}%)")
    if errors:
        print(f"  ERRORS:           {errors:3} requests")
    print()
    print("Cost estimate (at 100 output tokens per request):")
    opus_cost    = total * 100 * 25.00    / 1_000_000
    actual_cost  = (
        cheap    * 100 *  5.00  / 1_000_000 +
        balanced * 100 * 15.00  / 1_000_000 +
        capable  * 100 * 25.00  / 1_000_000
    )
    saving = opus_cost - actual_cost
    print(f"  Without routing (all Opus): ${opus_cost:.4f}")
    print(f"  With routing:               ${actual_cost:.4f}")
    print(f"  Saved on this test:         ${saving:.4f}")
    print(f"  Extrapolated (10k req/day): ${saving/total*10000:.2f}/day")
    print("=" * 65)
    print()
    print("View full dashboard: http://127.0.0.1:8000/dashboard")


if __name__ == "__main__":
    asyncio.run(run_load_test())