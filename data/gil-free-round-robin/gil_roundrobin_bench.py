"""
Python 3.14 (π Release) Free-Threading Benchmark
Round-Robin Producer-Consumer Queue Strategies with Backpressure

This benchmark compares different round-robin strategies for distributing work
across consumers with GIL disabled (--disable-gil or PYTHON_GIL=0).

Round-Robin Strategies:
1. Static Assignment: Each consumer bound to specific queue
2. Work Stealing: Consumers steal from others when idle
3. Central Dispatcher: Single queue with round-robin assignment
4. Sharded Queues: Multiple queues with hash-based routing
5. Dynamic Load Balancing: Runtime load-aware distribution

Backpressure Strategies:
6. Throttled Producer: Producer sleeps when queues are full
7. Drop Oldest: Discard oldest items when queue is full
8. Credit-Based Flow: Consumers grant credits to producer
9. Adaptive Queue Size: Dynamic queue sizing based on latency
10. Push-Pull Hybrid: Consumers pull when ready
"""

import sys
import time
import threading
import queue
import hashlib
import os
import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Callable
import statistics

# Check if GIL is disabled
if sys.version_info >= (3, 13):
    gil_disabled = not sys._is_gil_enabled() if hasattr(sys, '_is_gil_enabled') else False
else:
    gil_disabled = False


@dataclass
class BenchmarkResult:
    strategy: str
    total_time: float
    throughput: float
    avg_latency: float
    std_latency: float
    items_processed: int
    num_consumers: int


class WorkItem:
    """Simulated work item with CPU-bound task"""
    def __init__(self, item_id: int, work_iterations: int = 1000):
        self.item_id = item_id
        self.work_iterations = work_iterations
        self.created_at = time.perf_counter()
        self.processed_at = None
    
    def process(self):
        """Simulate CPU-intensive work"""
        # Hash computation to simulate real work
        result = self.item_id
        for _ in range(self.work_iterations):
            result = int(hashlib.sha256(str(result).encode()).hexdigest(), 16)
        self.processed_at = time.perf_counter()
        return result
    
    @property
    def latency(self):
        return self.processed_at - self.created_at if self.processed_at else None


# Strategy 1: Static Assignment Round-Robin
def static_assignment_strategy(num_consumers: int, num_items: int, work_iterations: int = 1000) -> BenchmarkResult:
    """Each consumer has dedicated queue, producer round-robins"""
    queues = [queue.Queue(maxsize=100) for _ in range(num_consumers)]
    items_processed = []
    stop_event = threading.Event()
    
    def consumer(q: queue.Queue, consumer_id: int):
        while not stop_event.is_set() or not q.empty():
            try:
                item = q.get(timeout=0.1)
                item.process()
                items_processed.append(item)
                q.task_done()
            except queue.Empty:
                continue
    
    def producer():
        for i in range(num_items):
            item = WorkItem(i, work_iterations)
            queues[i % num_consumers].put(item)
        stop_event.set()
    
    start = time.perf_counter()
    
    # Start consumers
    consumers = [threading.Thread(target=consumer, args=(queues[i], i)) 
                 for i in range(num_consumers)]
    for c in consumers:
        c.start()
    
    # Start producer
    prod = threading.Thread(target=producer)
    prod.start()
    prod.join()
    
    # Wait for all queues to be empty
    for q in queues:
        q.join()
    
    for c in consumers:
        c.join()
    
    end = time.perf_counter()
    
    latencies = [item.latency for item in items_processed if item.latency]
    return BenchmarkResult(
        strategy="Static Assignment",
        total_time=end - start,
        throughput=num_items / (end - start),
        avg_latency=statistics.mean(latencies) if latencies else 0,
        std_latency=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        items_processed=len(items_processed),
        num_consumers=num_consumers
    )


# Strategy 2: Work Stealing
def work_stealing_strategy(num_consumers: int, num_items: int, work_iterations: int = 1000) -> BenchmarkResult:
    """Consumers try their queue first, then steal from others"""
    queues = [queue.Queue(maxsize=100) for _ in range(num_consumers)]
    items_processed = []
    stop_event = threading.Event()
    lock = threading.Lock()
    
    def consumer(consumer_id: int):
        my_queue = queues[consumer_id]
        other_queues = [q for i, q in enumerate(queues) if i != consumer_id]
        
        while not stop_event.is_set() or any(not q.empty() for q in queues):
            item = None
            
            # Try own queue first
            try:
                item = my_queue.get_nowait()
            except queue.Empty:
                # Try stealing from others
                for q in other_queues:
                    try:
                        item = q.get_nowait()
                        break
                    except queue.Empty:
                        continue
            
            if item:
                item.process()
                with lock:
                    items_processed.append(item)
            else:
                time.sleep(0.001)  # Brief pause if no work found
    
    def producer():
        for i in range(num_items):
            item = WorkItem(i, work_iterations)
            queues[i % num_consumers].put(item)
        stop_event.set()
    
    start = time.perf_counter()
    
    consumers = [threading.Thread(target=consumer, args=(i,)) 
                 for i in range(num_consumers)]
    for c in consumers:
        c.start()
    
    prod = threading.Thread(target=producer)
    prod.start()
    prod.join()
    
    for c in consumers:
        c.join()
    
    end = time.perf_counter()
    
    latencies = [item.latency for item in items_processed if item.latency]
    return BenchmarkResult(
        strategy="Work Stealing",
        total_time=end - start,
        throughput=num_items / (end - start),
        avg_latency=statistics.mean(latencies) if latencies else 0,
        std_latency=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        items_processed=len(items_processed),
        num_consumers=num_consumers
    )


# Strategy 3: Central Dispatcher
def central_dispatcher_strategy(num_consumers: int, num_items: int, work_iterations: int = 1000) -> BenchmarkResult:
    """Single queue, all consumers pull from it"""
    central_queue = queue.Queue(maxsize=200)
    items_processed = []
    stop_event = threading.Event()
    lock = threading.Lock()
    
    def consumer(consumer_id: int):
        while not stop_event.is_set() or not central_queue.empty():
            try:
                item = central_queue.get(timeout=0.1)
                item.process()
                with lock:
                    items_processed.append(item)
                central_queue.task_done()
            except queue.Empty:
                continue
    
    def producer():
        for i in range(num_items):
            item = WorkItem(i, work_iterations)
            central_queue.put(item)
        stop_event.set()
    
    start = time.perf_counter()
    
    consumers = [threading.Thread(target=consumer, args=(i,)) 
                 for i in range(num_consumers)]
    for c in consumers:
        c.start()
    
    prod = threading.Thread(target=producer)
    prod.start()
    prod.join()
    
    central_queue.join()
    
    for c in consumers:
        c.join()
    
    end = time.perf_counter()
    
    latencies = [item.latency for item in items_processed if item.latency]
    return BenchmarkResult(
        strategy="Central Dispatcher",
        total_time=end - start,
        throughput=num_items / (end - start),
        avg_latency=statistics.mean(latencies) if latencies else 0,
        std_latency=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        items_processed=len(items_processed),
        num_consumers=num_consumers
    )


# Strategy 4: Sharded Queues with Hash Routing
def sharded_queues_strategy(num_consumers: int, num_items: int, work_iterations: int = 1000) -> BenchmarkResult:
    """Multiple shards, hash-based routing for better cache locality"""
    num_shards = num_consumers * 2  # More shards than consumers
    queues = [queue.Queue(maxsize=50) for _ in range(num_shards)]
    items_processed = []
    stop_event = threading.Event()
    lock = threading.Lock()
    
    def get_shard(item_id: int) -> int:
        return item_id % num_shards
    
    def consumer(consumer_id: int):
        # Each consumer handles multiple shards
        my_shards = [i for i in range(num_shards) if i % num_consumers == consumer_id]
        my_queues = [queues[i] for i in my_shards]
        
        while not stop_event.is_set() or any(not q.empty() for q in my_queues):
            processed = False
            for q in my_queues:
                try:
                    item = q.get_nowait()
                    item.process()
                    with lock:
                        items_processed.append(item)
                    processed = True
                    break
                except queue.Empty:
                    continue
            
            if not processed:
                time.sleep(0.001)
    
    def producer():
        for i in range(num_items):
            item = WorkItem(i, work_iterations)
            shard = get_shard(i)
            queues[shard].put(item)
        stop_event.set()
    
    start = time.perf_counter()
    
    consumers = [threading.Thread(target=consumer, args=(i,)) 
                 for i in range(num_consumers)]
    for c in consumers:
        c.start()
    
    prod = threading.Thread(target=producer)
    prod.start()
    prod.join()
    
    for c in consumers:
        c.join()
    
    end = time.perf_counter()
    
    latencies = [item.latency for item in items_processed if item.latency]
    return BenchmarkResult(
        strategy="Sharded Queues",
        total_time=end - start,
        throughput=num_items / (end - start),
        avg_latency=statistics.mean(latencies) if latencies else 0,
        std_latency=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        items_processed=len(items_processed),
        num_consumers=num_consumers
    )


# Strategy 5: Dynamic Load Balancing
def dynamic_load_balancing_strategy(num_consumers: int, num_items: int, work_iterations: int = 1000) -> BenchmarkResult:
    """Track consumer load and assign to least busy"""
    queues = [queue.Queue(maxsize=100) for _ in range(num_consumers)]
    items_processed = []
    stop_event = threading.Event()
    load_counter = [0] * num_consumers
    lock = threading.Lock()
    
    def consumer(consumer_id: int):
        my_queue = queues[consumer_id]
        while not stop_event.is_set() or not my_queue.empty():
            try:
                item = my_queue.get(timeout=0.1)
                item.process()
                with lock:
                    items_processed.append(item)
                    load_counter[consumer_id] -= 1
                my_queue.task_done()
            except queue.Empty:
                continue
    
    def producer():
        for i in range(num_items):
            item = WorkItem(i, work_iterations)
            # Find least loaded consumer
            with lock:
                min_load_idx = load_counter.index(min(load_counter))
                load_counter[min_load_idx] += 1
            queues[min_load_idx].put(item)
        stop_event.set()
    
    start = time.perf_counter()
    
    consumers = [threading.Thread(target=consumer, args=(i,)) 
                 for i in range(num_consumers)]
    for c in consumers:
        c.start()
    
    prod = threading.Thread(target=producer)
    prod.start()
    prod.join()
    
    for q in queues:
        q.join()
    
    for c in consumers:
        c.join()
    
    end = time.perf_counter()
    
    latencies = [item.latency for item in items_processed if item.latency]
    return BenchmarkResult(
        strategy="Dynamic Load Balancing",
        total_time=end - start,
        throughput=num_items / (end - start),
        avg_latency=statistics.mean(latencies) if latencies else 0,
        std_latency=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        items_processed=len(items_processed),
        num_consumers=num_consumers
    )


# Strategy 6: Throttled Producer (Backpressure)
def throttled_producer_strategy(num_consumers: int, num_items: int, work_iterations: int = 1000) -> BenchmarkResult:
    """Producer throttles (sleeps) when queues are full"""
    queues = [queue.Queue(maxsize=50) for _ in range(num_consumers)]  # Smaller queues
    items_processed = []
    stop_event = threading.Event()
    lock = threading.Lock()
    throttle_count = [0]
    
    def consumer(consumer_id: int):
        my_queue = queues[consumer_id]
        while not stop_event.is_set() or not my_queue.empty():
            try:
                item = my_queue.get(timeout=0.1)
                item.process()
                with lock:
                    items_processed.append(item)
                my_queue.task_done()
            except queue.Empty:
                continue
    
    def producer():
        for i in range(num_items):
            item = WorkItem(i, work_iterations)
            target_queue = queues[i % num_consumers]
            
            # Throttle when queue is full
            while True:
                try:
                    target_queue.put(item, timeout=0.01)
                    break
                except queue.Full:
                    throttle_count[0] += 1
                    time.sleep(0.005)  # Backpressure throttle
        stop_event.set()
    
    start = time.perf_counter()
    
    consumers = [threading.Thread(target=consumer, args=(i,)) 
                 for i in range(num_consumers)]
    for c in consumers:
        c.start()
    
    prod = threading.Thread(target=producer)
    prod.start()
    prod.join()
    
    for q in queues:
        q.join()
    
    for c in consumers:
        c.join()
    
    end = time.perf_counter()
    
    latencies = [item.latency for item in items_processed if item.latency]
    print(f"   [Throttle events: {throttle_count[0]}]")
    return BenchmarkResult(
        strategy="Throttled Producer",
        total_time=end - start,
        throughput=num_items / (end - start),
        avg_latency=statistics.mean(latencies) if latencies else 0,
        std_latency=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        items_processed=len(items_processed),
        num_consumers=num_consumers
    )


# Strategy 7: Drop Oldest (Realtime)
def drop_oldest_strategy(num_consumers: int, num_items: int, work_iterations: int = 1000) -> BenchmarkResult:
    """Discard oldest items when queue is full (realtime behavior)"""
    queues = [queue.Queue(maxsize=50) for _ in range(num_consumers)]
    items_processed = []
    items_dropped = [0]
    stop_event = threading.Event()
    lock = threading.Lock()
    
    def consumer(consumer_id: int):
        my_queue = queues[consumer_id]
        while not stop_event.is_set() or not my_queue.empty():
            try:
                item = my_queue.get(timeout=0.1)
                item.process()
                with lock:
                    items_processed.append(item)
                my_queue.task_done()
            except queue.Empty:
                continue
    
    def producer():
        for i in range(num_items):
            item = WorkItem(i, work_iterations)
            target_queue = queues[i % num_consumers]
            
            # Drop oldest if full (must call task_done() for the removed item)
            if target_queue.full():
                try:
                    _ = target_queue.get_nowait()
                    target_queue.task_done()
                    with lock:
                        items_dropped[0] += 1
                except queue.Empty:
                    pass

            target_queue.put(item)
        stop_event.set()
    
    start = time.perf_counter()
    
    consumers = [threading.Thread(target=consumer, args=(i,)) 
                 for i in range(num_consumers)]
    for c in consumers:
        c.start()
    
    prod = threading.Thread(target=producer)
    prod.start()
    prod.join()
    
    for q in queues:
        q.join()
    
    for c in consumers:
        c.join()
    
    end = time.perf_counter()
    
    latencies = [item.latency for item in items_processed if item.latency]
    print(f"   [Items dropped: {items_dropped[0]}]")
    return BenchmarkResult(
        strategy="Drop Oldest",
        total_time=end - start,
        throughput=len(items_processed) / (end - start),
        avg_latency=statistics.mean(latencies) if latencies else 0,
        std_latency=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        items_processed=len(items_processed),
        num_consumers=num_consumers
    )


# Strategy 8: Credit-Based Flow Control
def credit_based_strategy(num_consumers: int, num_items: int, work_iterations: int = 1000) -> BenchmarkResult:
    """Consumers grant credits to producer for flow control"""
    queues = [queue.Queue(maxsize=100) for _ in range(num_consumers)]
    items_processed = []
    stop_event = threading.Event()
    lock = threading.Lock()
    
    # Credit system: each consumer starts with initial credits
    initial_credits_per_consumer = 25
    credits = [threading.Semaphore(initial_credits_per_consumer) for _ in range(num_consumers)]
    
    def consumer(consumer_id: int):
        my_queue = queues[consumer_id]
        my_credits = credits[consumer_id]
        
        while not stop_event.is_set() or not my_queue.empty():
            try:
                item = my_queue.get(timeout=0.1)
                item.process()
                with lock:
                    items_processed.append(item)
                my_queue.task_done()
                
                # Release credit after processing
                my_credits.release()
            except queue.Empty:
                continue
    
    def producer():
        for i in range(num_items):
            item = WorkItem(i, work_iterations)
            target_idx = i % num_consumers
            
            # Acquire credit before enqueuing
            credits[target_idx].acquire()
            queues[target_idx].put(item)
        
        stop_event.set()
    
    start = time.perf_counter()
    
    consumers = [threading.Thread(target=consumer, args=(i,)) 
                 for i in range(num_consumers)]
    for c in consumers:
        c.start()
    
    prod = threading.Thread(target=producer)
    prod.start()
    prod.join()
    
    for q in queues:
        q.join()
    
    for c in consumers:
        c.join()
    
    end = time.perf_counter()
    
    latencies = [item.latency for item in items_processed if item.latency]
    return BenchmarkResult(
        strategy="Credit-Based Flow",
        total_time=end - start,
        throughput=num_items / (end - start),
        avg_latency=statistics.mean(latencies) if latencies else 0,
        std_latency=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        items_processed=len(items_processed),
        num_consumers=num_consumers
    )


# Strategy 9: Adaptive Queue Size
def adaptive_queue_strategy(num_consumers: int, num_items: int, work_iterations: int = 1000) -> BenchmarkResult:
    """Dynamic queue sizing based on consumer latency feedback"""
    # Start with moderate queue sizes
    queue_sizes = [50] * num_consumers
    queues = [queue.Queue(maxsize=queue_sizes[i]) for i in range(num_consumers)]
    items_processed = []
    stop_event = threading.Event()
    lock = threading.Lock()
    recent_latencies = [[] for _ in range(num_consumers)]
    adjustment_count = [0]
    
    def consumer(consumer_id: int):
        my_queue = queues[consumer_id]
        
        while not stop_event.is_set() or not my_queue.empty():
            try:
                item = my_queue.get(timeout=0.1)
                item.process()
                
                with lock:
                    items_processed.append(item)
                    if item.latency:
                        recent_latencies[consumer_id].append(item.latency)
                        # Keep only recent 10 samples
                        if len(recent_latencies[consumer_id]) > 10:
                            recent_latencies[consumer_id].pop(0)
                
                my_queue.task_done()
            except queue.Empty:
                continue
    
    def producer():
        check_interval = max(1, num_items // 20)  # Adjust 20 times during run
        
        for i in range(num_items):
            item = WorkItem(i, work_iterations)
            target_idx = i % num_consumers
            
            # Periodically adjust queue sizes
            if i > 0 and i % check_interval == 0:
                with lock:
                    for idx in range(num_consumers):
                        if len(recent_latencies[idx]) >= 5:
                            avg_lat = statistics.mean(recent_latencies[idx])
                            threshold = 0.01  # 10ms threshold
                            
                            # High latency: reduce queue size (apply backpressure)
                            if avg_lat > threshold and queue_sizes[idx] > 10:
                                queue_sizes[idx] = max(10, queue_sizes[idx] // 2)
                                adjustment_count[0] += 1
                            # Low latency: increase queue size (allow buffering)
                            elif avg_lat < threshold / 2 and queue_sizes[idx] < 200:
                                queue_sizes[idx] = min(200, queue_sizes[idx] * 2)
                                adjustment_count[0] += 1
            
            # Put with current queue behavior (note: can't change maxsize dynamically in stdlib queue)
            queues[target_idx].put(item)
        
        stop_event.set()
    
    start = time.perf_counter()
    
    consumers = [threading.Thread(target=consumer, args=(i,)) 
                 for i in range(num_consumers)]
    for c in consumers:
        c.start()
    
    prod = threading.Thread(target=producer)
    prod.start()
    prod.join()
    
    for q in queues:
        q.join()
    
    for c in consumers:
        c.join()
    
    end = time.perf_counter()
    
    latencies = [item.latency for item in items_processed if item.latency]
    print(f"   [Queue size adjustments: {adjustment_count[0]}]")
    return BenchmarkResult(
        strategy="Adaptive Queue Size",
        total_time=end - start,
        throughput=num_items / (end - start),
        avg_latency=statistics.mean(latencies) if latencies else 0,
        std_latency=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        items_processed=len(items_processed),
        num_consumers=num_consumers
    )


# Strategy 10: Push-Pull Hybrid (Consumer-Driven)
def push_pull_hybrid_strategy(num_consumers: int, num_items: int, work_iterations: int = 1000) -> BenchmarkResult:
    """Consumers pull items when ready (reactive pattern)"""
    items_available = queue.Queue(maxsize=num_items)
    items_processed = []
    stop_event = threading.Event()
    lock = threading.Lock()
    
    def consumer(consumer_id: int):
        while not stop_event.is_set() or not items_available.empty():
            try:
                # Consumer pulls when ready
                item = items_available.get(timeout=0.1)
                item.process()
                with lock:
                    items_processed.append(item)
                items_available.task_done()
            except queue.Empty:
                continue
    
    def producer():
        # Producer just fills the available pool
        for i in range(num_items):
            item = WorkItem(i, work_iterations)
            items_available.put(item)
        stop_event.set()
    
    start = time.perf_counter()
    
    # Start producer first to fill pool
    prod = threading.Thread(target=producer)
    prod.start()
    
    # Small delay to ensure some items are available
    time.sleep(0.01)
    
    # Start consumers (they pull when ready)
    consumers = [threading.Thread(target=consumer, args=(i,)) 
                 for i in range(num_consumers)]
    for c in consumers:
        c.start()
    
    prod.join()
    items_available.join()
    
    for c in consumers:
        c.join()
    
    end = time.perf_counter()
    
    latencies = [item.latency for item in items_processed if item.latency]
    return BenchmarkResult(
        strategy="Push-Pull Hybrid",
        total_time=end - start,
        throughput=num_items / (end - start),
        avg_latency=statistics.mean(latencies) if latencies else 0,
        std_latency=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        items_processed=len(items_processed),
        num_consumers=num_consumers
    )


def run_benchmark(num_consumers: int = 4, num_items: int = 10000, work_iterations: int = 1000, strategies: List[str] = None):
    """Run all benchmarks and display results"""
    
    # Available strategies
    all_strategies = {
        # Round-robin strategies
        'static': static_assignment_strategy,
        'workstealing': work_stealing_strategy,
        'central': central_dispatcher_strategy,
        'sharded': sharded_queues_strategy,
        'dynamic': dynamic_load_balancing_strategy,
        # Backpressure strategies
        'throttled': throttled_producer_strategy,
        'dropoldest': drop_oldest_strategy,
        'creditbased': credit_based_strategy,
        'adaptive': adaptive_queue_strategy,
        'pushpull': push_pull_hybrid_strategy,
    }
    
    # Select strategies to run
    if strategies:
        selected_strategies = {k: v for k, v in all_strategies.items() if k in strategies}
    else:
        selected_strategies = all_strategies
    
    print(f"\nPython {sys.version}")
    print(f"GIL Status: {'DISABLED ✓' if gil_disabled else 'ENABLED (run with --disable-gil or PYTHON_GIL=0)'}")
    print("=" * 80)
    print(f"Running benchmarks with {num_consumers} consumers, {num_items} items, {work_iterations} work iterations")
    print("=" * 80)
    
    results = []
    for name, strategy_fn in selected_strategies.items():
        print(f"\nTesting: {name.replace('_', ' ').title()}...", end=' ', flush=True)
        result = strategy_fn(num_consumers, num_items, work_iterations)
        results.append(result)
        print(f"✓ ({result.total_time:.3f}s)")
    
    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"{'Strategy':<25} {'Time(s)':<10} {'Throughput':<15} {'Avg Latency(ms)':<18} {'Std Dev(ms)':<12}")
    print("-" * 80)
    
    for result in sorted(results, key=lambda r: r.total_time):
        print(f"{result.strategy:<25} {result.total_time:<10.3f} "
              f"{result.throughput:<15.1f} {result.avg_latency*1000:<18.3f} "
              f"{result.std_latency*1000:<12.3f}")
    
    # Find best strategy
    best = min(results, key=lambda r: r.total_time)
    print("\n" + "=" * 80)
    print(f"🏆 Winner: {best.strategy}")
    print(f"   Best throughput: {best.throughput:.1f} items/sec")
    print(f"   Consumers: {best.num_consumers}")
    print(f"   With GIL {'DISABLED' if gil_disabled else 'ENABLED'}")
    print("=" * 80)
    
    if not gil_disabled:
        print("\n💡 Tip: Run with 'python --disable-gil' or 'PYTHON_GIL=0 python' for true parallelism!")
    
    if getattr(args, "json_output", None):
        summary = {
            "timestamp": time.time(),
            "num_consumers": num_consumers,
            "num_items": num_items,
            "work_iterations": work_iterations,
            "results": [
                {
                    "strategy": r.strategy,
                    "total_time": r.total_time,
                    "throughput": r.throughput,
                    "avg_latency": r.avg_latency,
                    "std_latency": r.std_latency,
                    "items_processed": r.items_processed,
                    "num_consumers": r.num_consumers
                } for r in results
            ]
        }
 
        import json
        try:
            with open(args.json_output, "w") as fh:
                json.dump(summary, fh)
        except Exception as e:
            print(f"Warning: failed to write JSON output to {args.json_output}: {e}", file=sys.stderr)

    return results


def scaling_test(max_consumers: int = 16, num_items: int = 10000, work_iterations: int = 1000):
    """Test scaling with different number of consumers"""
    print("\n" + "=" * 80)
    print("SCALING TEST - Work Stealing Strategy")
    print("=" * 80)
    
    consumer_counts = [1, 2, 4, 8, 12, 16] if max_consumers >= 16 else [1, 2, 4, 8]
    consumer_counts = [c for c in consumer_counts if c <= max_consumers]
    
    results = []
    for num_consumers in consumer_counts:
        print(f"\nTesting with {num_consumers} consumers...", end=' ', flush=True)
        result = work_stealing_strategy(num_consumers, num_items, work_iterations)
        results.append(result)
        print(f"✓ {result.throughput:.1f} items/sec")
    
    print("\n" + "=" * 80)
    print("SCALING RESULTS")
    print("=" * 80)
    print(f"{'Consumers':<12} {'Time(s)':<10} {'Throughput':<15} {'Speedup':<10} {'Efficiency(%)':<12}")
    print("-" * 80)
    
    baseline_time = results[0].total_time
    for result in results:
        speedup = baseline_time / result.total_time
        efficiency = (speedup / result.num_consumers) * 100
        print(f"{result.num_consumers:<12} {result.total_time:<10.3f} "
              f"{result.throughput:<15.1f} {speedup:<10.2f}x {efficiency:<12.1f}")
    
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Python 3.14 Free-Threading Benchmark')
    parser.add_argument('-c', '--consumers', type=int, default=None,
                        help='Number of consumer threads (default: auto-detect CPU count)')
    parser.add_argument('-i', '--items', type=int, default=10000,
                        help='Number of items to process (default: 10000)')
    parser.add_argument('-w', '--work', type=int, default=1000,
                        help='Work iterations per item (default: 1000)')
    parser.add_argument('-s', '--strategies', nargs='+', 
                        choices=['static', 'workstealing', 'central', 'sharded', 'dynamic',
                                'throttled', 'dropoldest', 'creditbased', 'adaptive', 'pushpull'],
                        help='Specific strategies to test (default: all)')
    parser.add_argument('--roundrobin-only', action='store_true',
                        help='Test only round-robin strategies (no backpressure)')
    parser.add_argument('--backpressure-only', action='store_true',
                        help='Test only backpressure strategies')
    parser.add_argument('--scaling', action='store_true',
                        help='Run scaling test with different consumer counts')
    parser.add_argument('--max-consumers', type=int, default=None,
                        help='Maximum consumers for scaling test (default: auto-detect)')
    parser.add_argument('--json-output', type=str, default=None,
                    help='Write JSON summary to this file (optional)')

    
    args = parser.parse_args()
    
    # Auto-detect CPU count if not specified
    cpu_count = os.cpu_count() or 4
    num_consumers = args.consumers if args.consumers else cpu_count
    max_consumers = args.max_consumers if args.max_consumers else cpu_count
    
    # Filter strategies based on flags
    selected_strategies = args.strategies
    if args.roundrobin_only:
        selected_strategies = ['static', 'workstealing', 'central', 'sharded', 'dynamic']
    elif args.backpressure_only:
        selected_strategies = ['throttled', 'dropoldest', 'creditbased', 'adaptive', 'pushpull']
    
    if args.scaling:
        scaling_test(max_consumers=max_consumers, num_items=args.items, work_iterations=args.work)
    else:
        run_benchmark(
            num_consumers=num_consumers,
            num_items=args.items,
            work_iterations=args.work,
            strategies=selected_strategies
        )

