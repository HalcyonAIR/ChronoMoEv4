# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright 2026 Halcyon AI Research (jeff@halcyon.ie)
"""Shared prompt corpus for MoE experiments.

Contains 200 categorized prompts (5 categories x 40 prompts each) covering
code, math, dialogue, reasoning, and factual domains. Originally developed
for the OLMoE Bob Phase 1 governed experiment and extracted here for reuse
across experiment harnesses.

Also provides the commit-then-violate (CTV) metric computation used to
measure whether cheap-path commits are followed by loss spikes.
"""

from collections import defaultdict
from typing import Dict, List


# ─── Prompt Categories ──────────────────────────────────────────────
# 40 prompts per category = 200 total. Each step processes one prompt.
# Categories cycle in blocks. With 200 unique prompts and ~300 active
# steps, each prompt is seen ~1.5 times. Enough routing diversity for
# meaningful CTV measurement.

PROMPTS = {
    "code": [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\ndef main():\n    for i in range(10):\n        print(fibonacci(i))",
        "import numpy as np\n\ndef matrix_multiply(A, B):\n    return np.dot(A, B)\n\nresult = matrix_multiply(np.eye(3), np.ones((3, 3)))",
        "class LinkedList:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\n    def append(self, val):\n        node = self\n        while node.next:\n            node = node.next\n        node.next = LinkedList(val)",
        "async function fetchData(url) {\n    const response = await fetch(url);\n    const data = await response.json();\n    return data.results.map(item => item.name);\n}",
        "SELECT u.name, COUNT(o.id) as order_count\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nWHERE u.created_at > '2024-01-01'\nGROUP BY u.name\nHAVING COUNT(o.id) > 5\nORDER BY order_count DESC;",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
        "import torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self, d_model, nhead):\n        super().__init__()\n        self.attention = nn.MultiheadAttention(d_model, nhead)\n        self.norm1 = nn.LayerNorm(d_model)\n        self.ffn = nn.Sequential(\n            nn.Linear(d_model, 4 * d_model),\n            nn.GELU(),\n            nn.Linear(4 * d_model, d_model),\n        )\n        self.norm2 = nn.LayerNorm(d_model)",
        "fn binary_search(arr: &[i32], target: i32) -> Option<usize> {\n    let mut low = 0;\n    let mut high = arr.len();\n    while low < high {\n        let mid = low + (high - low) / 2;\n        match arr[mid].cmp(&target) {\n            Ordering::Equal => return Some(mid),\n            Ordering::Less => low = mid + 1,\n            Ordering::Greater => high = mid,\n        }\n    }\n    None\n}",
        "class MaxHeap:\n    def __init__(self):\n        self.heap = []\n\n    def push(self, val):\n        self.heap.append(val)\n        self._sift_up(len(self.heap) - 1)\n\n    def pop(self):\n        if len(self.heap) == 1:\n            return self.heap.pop()\n        root = self.heap[0]\n        self.heap[0] = self.heap.pop()\n        self._sift_down(0)\n        return root",
        "CREATE TABLE users (\n    id SERIAL PRIMARY KEY,\n    username VARCHAR(50) UNIQUE NOT NULL,\n    email VARCHAR(100) UNIQUE NOT NULL,\n    created_at TIMESTAMP DEFAULT NOW(),\n    is_active BOOLEAN DEFAULT TRUE\n);\n\nCREATE INDEX idx_users_email ON users(email);\nCREATE INDEX idx_users_active ON users(is_active) WHERE is_active = TRUE;",
        "from typing import Generator\n\ndef prime_sieve(limit: int) -> Generator[int, None, None]:\n    sieve = [True] * (limit + 1)\n    sieve[0] = sieve[1] = False\n    for i in range(2, int(limit**0.5) + 1):\n        if sieve[i]:\n            for j in range(i*i, limit + 1, i):\n                sieve[j] = False\n    for i in range(2, limit + 1):\n        if sieve[i]:\n            yield i",
        "interface User {\n    id: string;\n    name: string;\n    email: string;\n    roles: Role[];\n}\n\ntype Role = 'admin' | 'editor' | 'viewer';\n\nfunction hasPermission(user: User, requiredRole: Role): boolean {\n    const hierarchy: Record<Role, number> = { admin: 3, editor: 2, viewer: 1 };\n    return user.roles.some(r => hierarchy[r] >= hierarchy[requiredRole]);\n}",
        "package main\n\nimport (\n    \"fmt\"\n    \"sync\"\n)\n\nfunc fanOut(input <-chan int, workers int) []<-chan int {\n    channels := make([]<-chan int, workers)\n    for i := 0; i < workers; i++ {\n        channels[i] = worker(input)\n    }\n    return channels\n}\n\nfunc worker(input <-chan int) <-chan int {\n    out := make(chan int)\n    go func() {\n        for n := range input {\n            out <- n * n\n        }\n        close(out)\n    }()\n    return out\n}",
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i])\n            i += 1\n        else:\n            result.append(right[j])\n            j += 1\n    result.extend(left[i:])\n    result.extend(right[j:])\n    return result",
        "class LRUCache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n        self.cache = {}\n        self.order = []\n\n    def get(self, key):\n        if key not in self.cache:\n            return -1\n        self.order.remove(key)\n        self.order.append(key)\n        return self.cache[key]\n\n    def put(self, key, value):\n        if key in self.cache:\n            self.order.remove(key)\n        elif len(self.cache) >= self.capacity:\n            oldest = self.order.pop(0)\n            del self.cache[oldest]\n        self.cache[key] = value\n        self.order.append(key)",
        "import asyncio\nimport aiohttp\n\nasync def fetch_all(urls):\n    async with aiohttp.ClientSession() as session:\n        tasks = [fetch_one(session, url) for url in urls]\n        return await asyncio.gather(*tasks)\n\nasync def fetch_one(session, url):\n    async with session.get(url) as response:\n        return await response.json()",
        "def depth_first_search(graph, start, visited=None):\n    if visited is None:\n        visited = set()\n    visited.add(start)\n    for neighbor in graph[start]:\n        if neighbor not in visited:\n            depth_first_search(graph, neighbor, visited)\n    return visited",
        "from dataclasses import dataclass, field\nfrom typing import List, Optional\n\n@dataclass\nclass TreeNode:\n    value: int\n    children: List['TreeNode'] = field(default_factory=list)\n    parent: Optional['TreeNode'] = None\n\n    def add_child(self, value: int) -> 'TreeNode':\n        child = TreeNode(value=value, parent=self)\n        self.children.append(child)\n        return child\n\n    def depth(self) -> int:\n        if self.parent is None:\n            return 0\n        return 1 + self.parent.depth()",
        "def dijkstra(graph, start):\n    import heapq\n    distances = {node: float('inf') for node in graph}\n    distances[start] = 0\n    pq = [(0, start)]\n    while pq:\n        dist, node = heapq.heappop(pq)\n        if dist > distances[node]:\n            continue\n        for neighbor, weight in graph[node].items():\n            new_dist = dist + weight\n            if new_dist < distances[neighbor]:\n                distances[neighbor] = new_dist\n                heapq.heappush(pq, (new_dist, neighbor))\n    return distances",
        "class EventEmitter:\n    def __init__(self):\n        self._listeners = {}\n\n    def on(self, event, callback):\n        if event not in self._listeners:\n            self._listeners[event] = []\n        self._listeners[event].append(callback)\n        return self\n\n    def emit(self, event, *args, **kwargs):\n        for callback in self._listeners.get(event, []):\n            callback(*args, **kwargs)\n\n    def off(self, event, callback):\n        if event in self._listeners:\n            self._listeners[event].remove(callback)",
        "const express = require('express');\nconst app = express();\n\napp.use(express.json());\n\nconst items = [];\n\napp.get('/items', (req, res) => {\n    res.json(items);\n});\n\napp.post('/items', (req, res) => {\n    const item = { id: items.length + 1, ...req.body };\n    items.push(item);\n    res.status(201).json(item);\n});\n\napp.listen(3000, () => console.log('Server running'));",
        "def topological_sort(graph):\n    visited = set()\n    stack = []\n    def dfs(node):\n        visited.add(node)\n        for neighbor in graph.get(node, []):\n            if neighbor not in visited:\n                dfs(neighbor)\n        stack.append(node)\n    for node in graph:\n        if node not in visited:\n            dfs(node)\n    return stack[::-1]",
        "class BTreeNode:\n    def __init__(self, leaf=False):\n        self.leaf = leaf\n        self.keys = []\n        self.children = []\n\n    def split_child(self, i, child):\n        t = len(child.keys) // 2\n        new_node = BTreeNode(leaf=child.leaf)\n        self.keys.insert(i, child.keys[t])\n        self.children.insert(i + 1, new_node)\n        new_node.keys = child.keys[t+1:]\n        child.keys = child.keys[:t]\n        if not child.leaf:\n            new_node.children = child.children[t+1:]\n            child.children = child.children[:t+1]",
        "#!/bin/bash\nset -euo pipefail\n\nREPO_DIR=\"/opt/deploy\"\nBRANCH=\"main\"\n\ncd \"$REPO_DIR\"\ngit fetch origin\ngit checkout \"$BRANCH\"\ngit pull origin \"$BRANCH\"\n\ndocker-compose build --no-cache\ndocker-compose down\ndocker-compose up -d\n\necho \"Deployment complete at $(date)\"",
        "def knapsack(weights, values, capacity):\n    n = len(weights)\n    dp = [[0] * (capacity + 1) for _ in range(n + 1)]\n    for i in range(1, n + 1):\n        for w in range(capacity + 1):\n            dp[i][w] = dp[i-1][w]\n            if weights[i-1] <= w:\n                dp[i][w] = max(dp[i][w], dp[i-1][w - weights[i-1]] + values[i-1])\n    return dp[n][capacity]",
        "import re\nfrom typing import Dict, List\n\ndef parse_log_entry(line: str) -> Dict:\n    pattern = r'(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}) \\[(\\w+)\\] (.+)'\n    match = re.match(pattern, line)\n    if not match:\n        return {'raw': line, 'level': 'UNKNOWN'}\n    return {\n        'timestamp': match.group(1),\n        'level': match.group(2),\n        'message': match.group(3),\n    }",
        "class Matrix:\n    def __init__(self, data):\n        self.data = data\n        self.rows = len(data)\n        self.cols = len(data[0]) if data else 0\n\n    def __matmul__(self, other):\n        assert self.cols == other.rows\n        result = [[0] * other.cols for _ in range(self.rows)]\n        for i in range(self.rows):\n            for j in range(other.cols):\n                for k in range(self.cols):\n                    result[i][j] += self.data[i][k] * other.data[k][j]\n        return Matrix(result)\n\n    def transpose(self):\n        return Matrix([[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)])",
        "def find_bridges(graph, n):\n    visited = [False] * n\n    disc = [0] * n\n    low = [0] * n\n    parent = [-1] * n\n    bridges = []\n    timer = [0]\n    def dfs(u):\n        visited[u] = True\n        disc[u] = low[u] = timer[0]\n        timer[0] += 1\n        for v in graph[u]:\n            if not visited[v]:\n                parent[v] = u\n                dfs(v)\n                low[u] = min(low[u], low[v])\n                if low[v] > disc[u]:\n                    bridges.append((u, v))\n            elif v != parent[u]:\n                low[u] = min(low[u], disc[v])\n    for i in range(n):\n        if not visited[i]:\n            dfs(i)\n    return bridges",
        "class Observable:\n    def __init__(self, value):\n        self._value = value\n        self._subscribers = []\n\n    @property\n    def value(self):\n        return self._value\n\n    @value.setter\n    def value(self, new_value):\n        old_value = self._value\n        self._value = new_value\n        for callback in self._subscribers:\n            callback(old_value, new_value)\n\n    def subscribe(self, callback):\n        self._subscribers.append(callback)\n        return lambda: self._subscribers.remove(callback)",
        "import unittest\nfrom unittest.mock import Mock, patch\n\nclass TestUserService(unittest.TestCase):\n    def setUp(self):\n        self.db = Mock()\n        self.service = UserService(self.db)\n\n    def test_create_user_success(self):\n        self.db.insert.return_value = {'id': 1, 'name': 'Alice'}\n        result = self.service.create_user('Alice', 'alice@test.com')\n        self.assertEqual(result['name'], 'Alice')\n        self.db.insert.assert_called_once()\n\n    def test_create_user_duplicate(self):\n        self.db.insert.side_effect = DuplicateKeyError()\n        with self.assertRaises(UserExistsError):\n            self.service.create_user('Alice', 'alice@test.com')",
        "def trie_insert(root, word):\n    node = root\n    for char in word:\n        if char not in node:\n            node[char] = {}\n        node = node[char]\n    node['$'] = True\n\ndef trie_search(root, word):\n    node = root\n    for char in word:\n        if char not in node:\n            return False\n        node = node[char]\n    return '$' in node\n\ndef trie_prefix(root, prefix):\n    node = root\n    for char in prefix:\n        if char not in node:\n            return False\n        node = node[char]\n    return True",
        "from functools import wraps\nimport time\n\ndef retry(max_attempts=3, delay=1.0, backoff=2.0):\n    def decorator(func):\n        @wraps(func)\n        def wrapper(*args, **kwargs):\n            current_delay = delay\n            for attempt in range(max_attempts):\n                try:\n                    return func(*args, **kwargs)\n                except Exception as e:\n                    if attempt == max_attempts - 1:\n                        raise\n                    time.sleep(current_delay)\n                    current_delay *= backoff\n        return wrapper\n    return decorator",
        "class MinStack:\n    def __init__(self):\n        self.stack = []\n        self.min_stack = []\n\n    def push(self, val):\n        self.stack.append(val)\n        if not self.min_stack or val <= self.min_stack[-1]:\n            self.min_stack.append(val)\n\n    def pop(self):\n        val = self.stack.pop()\n        if val == self.min_stack[-1]:\n            self.min_stack.pop()\n        return val\n\n    def get_min(self):\n        return self.min_stack[-1]",
        "def longest_common_subsequence(text1, text2):\n    m, n = len(text1), len(text2)\n    dp = [[0] * (n + 1) for _ in range(m + 1)]\n    for i in range(1, m + 1):\n        for j in range(1, n + 1):\n            if text1[i-1] == text2[j-1]:\n                dp[i][j] = dp[i-1][j-1] + 1\n            else:\n                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n    return dp[m][n]",
        "class RateLimiter:\n    def __init__(self, max_requests, window_seconds):\n        self.max_requests = max_requests\n        self.window = window_seconds\n        self.requests = {}\n\n    def allow(self, client_id):\n        now = time.time()\n        if client_id not in self.requests:\n            self.requests[client_id] = []\n        self.requests[client_id] = [\n            t for t in self.requests[client_id]\n            if now - t < self.window\n        ]\n        if len(self.requests[client_id]) < self.max_requests:\n            self.requests[client_id].append(now)\n            return True\n        return False",
        "class BloomFilter:\n    def __init__(self, size, num_hashes):\n        self.size = size\n        self.num_hashes = num_hashes\n        self.bit_array = [False] * size\n\n    def _hashes(self, item):\n        h1 = hash(item)\n        h2 = hash(str(item) + 'salt')\n        return [(h1 + i * h2) % self.size for i in range(self.num_hashes)]\n\n    def add(self, item):\n        for idx in self._hashes(item):\n            self.bit_array[idx] = True\n\n    def __contains__(self, item):\n        return all(self.bit_array[idx] for idx in self._hashes(item))",
        "def a_star(grid, start, end):\n    import heapq\n    rows, cols = len(grid), len(grid[0])\n    open_set = [(0, start)]\n    came_from = {}\n    g_score = {start: 0}\n    f_score = {start: heuristic(start, end)}\n    while open_set:\n        _, current = heapq.heappop(open_set)\n        if current == end:\n            return reconstruct_path(came_from, current)\n        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:\n            neighbor = (current[0]+dx, current[1]+dy)\n            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:\n                if grid[neighbor[0]][neighbor[1]] == 1:\n                    continue\n                tentative = g_score[current] + 1\n                if tentative < g_score.get(neighbor, float('inf')):\n                    came_from[neighbor] = current\n                    g_score[neighbor] = tentative\n                    f_score[neighbor] = tentative + heuristic(neighbor, end)\n                    heapq.heappush(open_set, (f_score[neighbor], neighbor))\n    return None",
        "class SegmentTree:\n    def __init__(self, data):\n        self.n = len(data)\n        self.tree = [0] * (4 * self.n)\n        self._build(data, 1, 0, self.n - 1)\n\n    def _build(self, data, node, start, end):\n        if start == end:\n            self.tree[node] = data[start]\n        else:\n            mid = (start + end) // 2\n            self._build(data, 2*node, start, mid)\n            self._build(data, 2*node+1, mid+1, end)\n            self.tree[node] = self.tree[2*node] + self.tree[2*node+1]",
        "class UnionFind:\n    def __init__(self, n):\n        self.parent = list(range(n))\n        self.rank = [0] * n\n        self.components = n\n\n    def find(self, x):\n        if self.parent[x] != x:\n            self.parent[x] = self.find(self.parent[x])\n        return self.parent[x]\n\n    def union(self, x, y):\n        px, py = self.find(x), self.find(y)\n        if px == py:\n            return False\n        if self.rank[px] < self.rank[py]:\n            px, py = py, px\n        self.parent[py] = px\n        if self.rank[px] == self.rank[py]:\n            self.rank[px] += 1\n        self.components -= 1\n        return True",
    ],
    "math": [
        "The derivative of f(x) = x^3 + 2x^2 - 5x + 1 is f'(x) = 3x^2 + 4x - 5. Setting f'(x) = 0 and using the quadratic formula gives critical points at x = (-4 +/- sqrt(76))/6.",
        "To prove that sqrt(2) is irrational, assume sqrt(2) = p/q where p and q are coprime integers. Then 2q^2 = p^2, so p^2 is even, meaning p is even. Write p = 2k, then 2q^2 = 4k^2, so q^2 = 2k^2, meaning q is also even. Contradiction with coprimality.",
        "The integral of e^(-x^2) from negative infinity to infinity equals sqrt(pi). This is the Gaussian integral and can be evaluated by squaring it and converting to polar coordinates: I^2 = integral of r * e^(-r^2) dr dtheta from 0 to 2pi and 0 to infinity.",
        "Consider a 3x3 matrix A with eigenvalues 1, 2, 3. The determinant of A equals the product of eigenvalues: det(A) = 1 * 2 * 3 = 6. The trace equals the sum: tr(A) = 1 + 2 + 3 = 6. The characteristic polynomial is (lambda-1)(lambda-2)(lambda-3).",
        "By the pigeonhole principle, if we have 13 people in a room, at least two of them were born in the same month, since there are only 12 months. More generally, if n items are put into m containers with n > m, at least one container has more than one item.",
        "Euler's formula states that e^(ix) = cos(x) + i*sin(x). Setting x = pi gives e^(i*pi) = -1, which is Euler's identity: e^(i*pi) + 1 = 0. This connects five fundamental constants: e, i, pi, 1, and 0.",
        "The Cauchy-Schwarz inequality states that for vectors u and v, |<u,v>|^2 <= <u,u><v,v>. Equality holds if and only if u and v are linearly dependent. This is equivalent to |cos(theta)| <= 1 where theta is the angle between u and v.",
        "The binomial theorem: (a+b)^n = sum from k=0 to n of C(n,k) * a^(n-k) * b^k. For example, (1+x)^4 = 1 + 4x + 6x^2 + 4x^3 + x^4. The coefficients form Pascal's triangle.",
        "The fundamental theorem of calculus connects differentiation and integration: if F'(x) = f(x), then the integral from a to b of f(x)dx = F(b) - F(a). This means every continuous function has an antiderivative.",
        "A group (G, *) satisfies four axioms: closure, associativity, identity element, and inverse element. The symmetric group S_n has n! elements. The cyclic group Z_n has n elements and is generated by a single element.",
        "The Taylor series of sin(x) around x=0 is x - x^3/3! + x^5/5! - x^7/7! + ... This converges for all x. The Maclaurin series of e^x is 1 + x + x^2/2! + x^3/3! + ..., also converging everywhere.",
        "In probability, Bayes' theorem states P(A|B) = P(B|A)P(A)/P(B). For example, if a medical test has 99% sensitivity and 95% specificity, and 1% of the population has the disease, the probability of actually having the disease given a positive test is about 17%.",
        "The Fibonacci sequence satisfies the recurrence F(n) = F(n-1) + F(n-2) with F(0)=0, F(1)=1. The closed-form solution involves the golden ratio phi = (1+sqrt(5))/2: F(n) = (phi^n - psi^n)/sqrt(5) where psi = (1-sqrt(5))/2.",
        "Green's theorem relates a line integral around a simple closed curve C to a double integral over the region D enclosed by C: the circulation of F around C equals the integral of curl(F) over D.",
        "The central limit theorem states that the sum of n independent identically distributed random variables, properly normalized, converges in distribution to a standard normal as n approaches infinity. This holds regardless of the original distribution, provided it has finite variance.",
        "The Riemann zeta function zeta(s) = sum from n=1 to infinity of 1/n^s converges for Re(s) > 1. The famous Riemann hypothesis conjectures that all non-trivial zeros have real part 1/2.",
        "Lagrange multipliers: to optimize f(x,y) subject to g(x,y)=0, we solve the system grad(f) = lambda * grad(g) and g(x,y) = 0. Geometrically, the gradients of f and g must be parallel at the optimum.",
        "The divergence theorem in 3D: the flux of a vector field F through a closed surface S equals the volume integral of div(F) over the region enclosed by S. This generalizes Green's theorem to three dimensions.",
        "Stirling's approximation: n! is approximately sqrt(2*pi*n) * (n/e)^n for large n. The relative error decreases as 1/(12n). This is useful for approximating binomial coefficients and in statistical mechanics.",
        "The Jordan normal form theorem: every square matrix over the complex numbers is similar to a block diagonal matrix with Jordan blocks. The eigenvalues appear on the diagonal, and ones may appear on the superdiagonal.",
        "The Chinese remainder theorem: if m_1, m_2, ..., m_k are pairwise coprime, then the system x = a_1 (mod m_1), ..., x = a_k (mod m_k) has a unique solution modulo M = m_1 * m_2 * ... * m_k.",
        "Galois theory connects field extensions to group theory. A polynomial equation is solvable by radicals if and only if its Galois group is solvable. This proves the quintic is not generally solvable.",
        "The spectral theorem for symmetric matrices: every real symmetric matrix A can be diagonalized by an orthogonal matrix: A = Q * D * Q^T, where D is diagonal with real eigenvalues and Q's columns are orthonormal eigenvectors.",
        "Stokes' theorem generalizes several classical theorems: the integral of d(omega) over a manifold M equals the integral of omega over the boundary of M. Green's theorem, the divergence theorem, and the fundamental theorem of calculus are all special cases.",
        "The rank-nullity theorem: for a linear map T from V to W, dim(V) = rank(T) + nullity(T). This means the dimension of the domain equals the dimension of the image plus the dimension of the kernel.",
        "Fermat's little theorem: if p is prime and gcd(a,p)=1, then a^(p-1) = 1 (mod p). This can be used for modular exponentiation and is the basis of primality tests like the Miller-Rabin test.",
        "The Hahn-Banach theorem is a foundational result in functional analysis: every bounded linear functional defined on a subspace of a normed vector space can be extended to the whole space without increasing its norm.",
        "Monte Carlo integration approximates integrals using random sampling. The estimate is sum(f(x_i))/n * volume, where x_i are uniform random points. The error decreases as 1/sqrt(n), independent of dimension.",
        "The moment generating function M_X(t) = E[e^(tX)] uniquely determines the distribution. For a normal distribution with mean mu and variance sigma^2, M_X(t) = exp(mu*t + sigma^2*t^2/2).",
        "Cantor's diagonal argument shows that the set of real numbers is uncountable. Given any list of real numbers, we can construct a real number not in the list by making the n-th digit of our number differ from the n-th digit of the n-th listed number.",
        "The Cauchy integral formula: if f is analytic inside and on a simple closed contour C, then f(z_0) = (1/2pi*i) * integral_C f(z)/(z-z_0) dz for any z_0 inside C. This implies analytic functions are infinitely differentiable.",
        "The isoperimetric inequality states that among all closed curves of a given perimeter, the circle encloses the maximum area. Equivalently, 4*pi*A <= L^2, with equality if and only if the curve is a circle.",
        "Linear programming: the simplex method finds the optimum of a linear objective function subject to linear constraints. The optimal solution occurs at a vertex of the feasible polytope. Interior point methods provide an alternative with polynomial time complexity.",
        "The Lebesgue dominated convergence theorem: if f_n converges pointwise to f and |f_n| <= g for some integrable g, then the integral of f_n converges to the integral of f. This is stronger than the monotone convergence theorem.",
        "Graph coloring: the chromatic number chi(G) is the minimum number of colors needed to color vertices so no adjacent vertices share a color. By the four color theorem, chi(G) <= 4 for any planar graph.",
        "The Bolzano-Weierstrass theorem: every bounded sequence in R^n has a convergent subsequence. This is equivalent to the completeness of R and is fundamental in analysis for proving existence results.",
        "Information theory: the entropy H(X) = -sum p(x) log p(x) measures the average information content. For a fair coin, H = 1 bit. Shannon's coding theorem relates entropy to the minimum average code length.",
        "The Perron-Frobenius theorem: a positive matrix has a unique largest eigenvalue, which is positive and real. The corresponding eigenvector has all positive entries. This is the mathematical basis of PageRank.",
        "Noether's theorem connects symmetries to conservation laws: every continuous symmetry of the action of a physical system corresponds to a conservation law. Time translation symmetry gives conservation of energy.",
        "The law of large numbers: the sample mean converges to the population mean as sample size increases. The strong version says convergence is almost sure; the weak version says convergence is in probability.",
    ],
    "dialogue": [
        "Hey, how's it going? I was thinking we could grab coffee tomorrow afternoon if you're free. The new place on Main Street has really good lattes.",
        "I can't believe the weather today! It was supposed to rain all week but it's actually sunny. Want to go for a walk in the park later?",
        "So I was telling my friend about that movie we saw last weekend, and she said she's been wanting to see it too. Maybe we should all go together next time.",
        "Thanks for helping me move last weekend! I really appreciate it. Let me know when you want me to return the favor. I owe you dinner at least.",
        "Did you hear about the concert next month? Apparently tickets go on sale Friday. We should try to get some before they sell out again.",
        "I just got back from vacation and my inbox is completely overflowing. I don't even know where to start. How was your week while I was gone?",
        "My neighbor's dog keeps barking at three in the morning. I've tried talking to them about it but nothing changes. Any advice on what to do?",
        "Remember when we used to play video games all weekend? Those were the days. We should have a gaming night sometime soon, for old times' sake.",
        "I'm thinking about learning to cook more. Right now I basically just make pasta and order takeout. Do you know any good beginner recipes?",
        "The traffic was absolutely terrible this morning. It took me almost two hours to get to work. I'm seriously considering switching to public transit.",
        "Have you tried that new restaurant downtown? I heard the sushi is amazing but the wait times can be pretty long, especially on weekends.",
        "I finally finished that book you recommended. You were right, the ending was completely unexpected. I did not see that twist coming at all.",
        "My car broke down again yesterday. The mechanic said it needs a new transmission. At this point I'm wondering if I should just get a new car.",
        "I can't decide whether to go back to school or not. The idea of more student debt is scary, but I feel like I need the degree to advance.",
        "Hey, are you free this Saturday? My sister is having a barbecue and she said I could bring friends. There'll be burgers and a pool.",
        "I just adopted a cat from the shelter! She's a calico and her name is Mochi. She's been hiding under the bed since I brought her home though.",
        "The power went out for six hours last night during the storm. I had to throw away everything in my freezer. At least I had some candles.",
        "I'm training for a half marathon in April. Today I ran eight miles and I'm already sore. Not sure how I'm going to do thirteen by race day.",
        "My parents are visiting next week and I haven't cleaned my apartment in way too long. I need to at least make the guest room presentable.",
        "Can you believe it's already February? This year is flying by. It feels like New Year's was just yesterday. Time really speeds up as you get older.",
        "I've been binge-watching that show everyone's been talking about. I'm on season three now and I still have no idea what's actually happening.",
        "The kids in the apartment above me have been playing drums. Actual drums. In an apartment. I don't know how their parents allow it.",
        "I got a really nice compliment from my boss today. She said my presentation was the best one she'd seen all quarter. Made my whole day.",
        "I'm thinking about getting a houseplant but I've killed every plant I've ever owned. Maybe a cactus? Those are supposed to be hard to kill.",
        "Did you see the sunset last night? The sky was this incredible shade of orange and pink. I wish I had taken a photo but I just stood there watching.",
        "I've been trying to get into meditation but I can't seem to quiet my mind. Every time I sit down to meditate, I just think about my to-do list.",
        "My coworker brought homemade cookies to the office today and they were incredible. Chocolate chip with sea salt. I need to get that recipe.",
        "I locked myself out of my apartment again. Third time this month. I really need to make a spare key and give it to someone I trust.",
        "We should plan a road trip this summer. I've been wanting to drive along the coast. We could stop at all the little beach towns along the way.",
        "I just found out that my favorite coffee shop is closing. Apparently the rent went up too much. I'm genuinely sad about it, I went there every morning.",
        "My phone screen cracked when I dropped it at the grocery store. It still works but there's a huge crack right across the middle. So frustrating.",
        "I tried making sourdough bread during the weekend. It turned out okay, a bit dense, but the taste was actually really good. Practice makes perfect.",
        "Have you ever been to Japan? My friend just came back and she said it was the most incredible trip she's ever taken. Now I really want to go.",
        "I'm trying to spend less time on social media. I deleted the apps from my phone but I keep going to the websites on my browser instead.",
        "The farmers market was amazing this morning. I got fresh strawberries, some artisan cheese, and this incredible loaf of sourdough.",
        "I just realized I forgot my mom's birthday. It was yesterday. I feel terrible. I called her this morning and she said it was fine but I know she was hurt.",
        "My internet has been so slow lately. I'm paying for high-speed but I can barely load a webpage. I need to call the provider and complain.",
        "I signed up for a pottery class on a whim. The first session is next Tuesday. I have absolutely no artistic talent but it sounds fun.",
        "The landlord raised the rent again. That's the third increase in two years. I love this neighborhood but I might have to start looking for a new place.",
        "I found a twenty dollar bill on the ground today. Normally I'd try to find the owner but there was nobody around. Guess I'm buying lunch.",
    ],
    "reasoning": [
        "If all dogs are mammals, and all mammals are animals, then all dogs are animals. This is a valid syllogism following the transitive property of set inclusion.",
        "The prisoner's dilemma shows that individually rational decisions can lead to collectively suboptimal outcomes. Both players defecting gives worse payoffs than both cooperating.",
        "Consider a trolley heading toward five people. You can pull a lever to divert it to a side track where it will hit one person. The utilitarian calculus suggests pulling the lever, maximizing total welfare.",
        "Correlation does not imply causation. Ice cream sales and drowning deaths both increase in summer, but ice cream doesn't cause drowning. The common cause is warm weather, a confounding variable.",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? The answer is still 5 minutes, since each machine makes one widget in 5 minutes.",
        "The Monty Hall problem: after you choose a door, the host opens a different door revealing a goat. You should switch. The probability of winning by switching is 2/3, not 1/2, because the host's action provides information.",
        "Survivorship bias: we study successful companies to find common traits, but we don't study failed companies with the same traits. The visible survivors don't represent the full population.",
        "The base rate fallacy: even with a 99% accurate test, if only 1 in 1000 people have a condition, a positive result means only about a 9% chance of actually having it. Prior probability matters enormously.",
        "Simpson's paradox: a trend that appears in several groups of data can reverse when the groups are combined. A treatment can appear better in every subgroup but worse overall, due to unequal group sizes.",
        "The gambler's fallacy: the belief that past random events affect future ones. After flipping ten heads in a row, the probability of the next flip being tails is still exactly 50%. The coin has no memory.",
        "Occam's razor suggests that among competing hypotheses that predict equally well, the one with the fewest assumptions should be selected. Simplicity is not proof, but it's a useful heuristic for model selection.",
        "The sorites paradox: removing one grain from a heap still leaves a heap. But repeated application would mean one grain is a heap. This challenges classical logic's binary categories and motivates fuzzy logic.",
        "Confirmation bias: we tend to search for and interpret information in ways that confirm our existing beliefs. A fair evaluation requires actively seeking disconfirming evidence, which is psychologically difficult.",
        "The tragedy of the commons: when individuals act in self-interest with shared resources, the resource gets depleted. Overfishing, air pollution, and traffic congestion are real-world examples.",
        "Modus ponens is the inference rule: if P implies Q, and P is true, then Q is true. Modus tollens is: if P implies Q, and Q is false, then P is false. Affirming the consequent is a fallacy.",
        "The frame problem in AI: how does a reasoning system determine which facts change as a result of an action and which remain the same? Turning on a light doesn't change the color of the walls.",
        "Newcomb's problem: a predictor has put $1M in a box if they predicted you'd take only that box. Do you take one box or two? One-boxers use evidential decision theory; two-boxers use causal decision theory.",
        "The regression to the mean: extreme observations tend to be followed by more moderate ones. A student who scores highest on one test will likely score lower on the next, purely statistically.",
        "Cognitive dissonance theory: when beliefs and actions conflict, we experience discomfort and tend to change our beliefs to match our actions, rather than vice versa. This explains post-purchase rationalization.",
        "The sunk cost fallacy: continuing an endeavor because of previously invested resources rather than future value. You shouldn't finish a bad movie just because you paid for the ticket.",
        "Arrow's impossibility theorem: no voting system with three or more candidates can simultaneously satisfy all fairness criteria: unanimity, independence of irrelevant alternatives, and non-dictatorship.",
        "The selection effect: we observe X because X is the type of thing we can observe. The anthropic principle is an example: the universe appears fine-tuned for life because only such universes contain observers.",
        "Goodhart's law: when a measure becomes a target, it ceases to be a good measure. If teachers are evaluated by test scores, they teach to the test rather than for understanding.",
        "The is-ought problem (Hume's guillotine): you cannot derive a moral statement (what ought to be) from a factual statement (what is). The existence of suffering doesn't logically imply we ought to reduce it.",
        "Inductive reasoning is ampliative: the conclusion goes beyond the premises. Every swan I've seen is white, therefore all swans are white. This was refuted by the discovery of black swans in Australia.",
        "The paradox of tolerance: a tolerant society must be intolerant of intolerance. If a society tolerates those who seek to destroy tolerance, it will eventually be destroyed.",
        "Anchoring bias: initial exposure to a number influences subsequent judgments. If asked whether Gandhi died before or after age 9, people estimate a younger death age than if asked about age 140.",
        "The problem of induction (Hume): past regularities don't logically guarantee future ones. The sun has risen every day, but this doesn't prove it will rise tomorrow. Science relies on induction without proving it valid.",
        "Counterfactual reasoning: what would have happened if things were different? If I hadn't missed the bus, I wouldn't have been late. This requires reasoning about non-actual but possible worlds.",
        "The availability heuristic: we judge probability based on how easily examples come to mind. Plane crashes seem more likely than car accidents because they're more memorable, despite being far rarer.",
        "Bayesian updating: start with a prior probability, observe evidence, and update to a posterior. P(H|E) = P(E|H)*P(H)/P(E). This is how rational agents should change beliefs in light of new evidence.",
        "The fallacy of composition: assuming what's true of parts is true of the whole. Every player on the team is excellent doesn't mean the team is excellent. Chemistry and coordination also matter.",
        "Occam's razor applied to scientific theories: epicycles could explain planetary orbits, but Kepler's ellipses are simpler and more predictive. The simpler theory won not because it's true, but because it's more useful.",
        "The trolley problem variant (fat man): pushing a large person off a bridge to stop the trolley feels different from pulling a lever, even though the outcome (one dies to save five) is the same. This challenges pure consequentialism.",
        "Epistemic humility: recognizing the limits of our knowledge. We should assign probabilities to our beliefs rather than treating them as certainties. Being 95% confident leaves room for being wrong.",
        "The Dunning-Kruger effect: incompetent individuals tend to overestimate their abilities, while experts tend to underestimate theirs. This is because evaluating one's performance requires the same skills as performing well.",
        "Decision under uncertainty: expected utility theory says we should maximize the expected value of outcomes weighted by probabilities. But Allais' paradox shows that people often violate this in practice.",
        "The problem of other minds: we can observe behavior but not consciousness. We infer that other people are conscious by analogy with ourselves, but this is an assumption, not a proof.",
        "Formal vs informal fallacies: formal fallacies are errors in logical structure (affirming the consequent). Informal fallacies are errors in content or context (appeal to authority, ad hominem).",
        "The precautionary principle: when an action raises threats of harm, precautionary measures should be taken even if cause-and-effect relationships are not fully established scientifically.",
    ],
    "factual": [
        "The speed of light in vacuum is approximately 299,792,458 meters per second. This is a fundamental constant of nature denoted by the letter c. Nothing with mass can reach this speed.",
        "Water molecules consist of two hydrogen atoms covalently bonded to one oxygen atom, forming an angle of about 104.5 degrees. This bent geometry gives water its polar properties and high boiling point.",
        "The Great Wall of China was built over many centuries, with the most well-known sections constructed during the Ming Dynasty (1368-1644). It stretches over 13,000 miles across northern China.",
        "Photosynthesis converts carbon dioxide and water into glucose and oxygen using light energy. The overall equation is 6CO2 + 6H2O + light -> C6H12O6 + 6O2. It occurs in chloroplasts.",
        "The human genome contains approximately 3 billion base pairs of DNA, organized into 23 pairs of chromosomes. Only about 1.5% of the genome codes for proteins. The rest includes regulatory sequences and more.",
        "The Earth is approximately 4.54 billion years old. The oldest known rocks are about 4 billion years old, found in northern Canada. Life first appeared roughly 3.5 billion years ago.",
        "Jupiter is the largest planet in our solar system, with a mass more than twice that of all other planets combined. It has at least 95 known moons, including the four large Galilean moons.",
        "The periodic table has 118 confirmed elements as of 2024. Elements 1-94 occur naturally; elements 95-118 are synthetic. The most recently confirmed element is oganesson (element 118).",
        "DNA replication is semiconservative: each daughter molecule contains one old strand and one new strand. The enzyme DNA polymerase adds nucleotides in the 5' to 3' direction, reading the template 3' to 5'.",
        "The Amazon rainforest covers approximately 5.5 million square kilometers and produces about 6% of the world's oxygen. It contains roughly 10% of all species on Earth and spans nine countries.",
        "The Mariana Trench is the deepest oceanic trench on Earth, reaching a depth of approximately 36,000 feet (about 11,000 meters). The deepest point is called Challenger Deep, located near the Mariana Islands.",
        "Black holes form when massive stars collapse at the end of their lives. The boundary beyond which nothing can escape is called the event horizon. The supermassive black hole at the center of the Milky Way, Sagittarius A*, has a mass of about 4 million Suns.",
        "The human brain contains approximately 86 billion neurons, connected by roughly 100 trillion synapses. It consumes about 20% of the body's energy despite being only 2% of body weight.",
        "The pH scale ranges from 0 to 14, with 7 being neutral. Acids have pH below 7 and bases have pH above 7. Each unit represents a tenfold change in hydrogen ion concentration.",
        "Plate tectonics describes the movement of Earth's lithosphere. The plates move at speeds of a few centimeters per year. Where plates collide, mountains form; where they diverge, new crust is created.",
        "The mitochondrion is the powerhouse of the cell, producing ATP through oxidative phosphorylation. It has its own DNA, suggesting it originated as an endosymbiotic bacterium billions of years ago.",
        "The Hubble constant measures the rate of expansion of the universe, currently estimated at about 70 km/s per megaparsec. This means galaxies are moving apart faster the farther away they are.",
        "Antibiotics work by targeting specific bacterial processes. Penicillin inhibits cell wall synthesis. Tetracycline blocks ribosomal protein synthesis. Fluoroquinolones inhibit DNA replication enzymes.",
        "The ozone layer, located in the stratosphere at 15-35 km altitude, absorbs 97-99% of the Sun's medium-frequency ultraviolet light. CFCs caused a hole in the ozone layer over Antarctica, discovered in 1985.",
        "Absolute zero is -273.15 degrees Celsius or 0 Kelvin. At this temperature, atoms would have minimum thermal motion. The third law of thermodynamics states that absolute zero can never be reached exactly.",
        "The speed of sound in air at sea level and 20 degrees Celsius is approximately 343 meters per second, or about 1,235 km/h. Sound travels faster in water (about 1,480 m/s) and even faster in steel (about 5,960 m/s).",
        "The sun is approximately 4.6 billion years old and is classified as a G-type main-sequence star. It converts about 600 million tons of hydrogen into helium every second through nuclear fusion.",
        "Human blood is classified into four main groups: A, B, AB, and O, based on the presence of antigens on red blood cells. The Rh factor adds positive or negative designation. Type O negative is the universal donor.",
        "The theory of general relativity, published by Einstein in 1915, describes gravity as the curvature of spacetime caused by mass and energy. It predicts gravitational time dilation, gravitational lensing, and gravitational waves.",
        "CRISPR-Cas9 is a gene editing technology that allows precise modification of DNA sequences. It uses a guide RNA to direct the Cas9 enzyme to a specific location in the genome, where it makes a double-strand break.",
        "The Dead Sea, located between Jordan and Israel, is the lowest point on Earth's surface at about 430 meters below sea level. Its salinity is about 34%, making it nearly ten times saltier than the ocean.",
        "Quantum entanglement is a phenomenon where two particles become correlated such that the quantum state of one instantly influences the other, regardless of distance. Einstein famously called this spooky action at a distance.",
        "The International Space Station orbits Earth at an altitude of approximately 400 km, traveling at about 28,000 km/h. It completes one orbit every 90 minutes and has been continuously occupied since November 2000.",
        "Photons are massless particles that carry electromagnetic radiation. They exhibit wave-particle duality: they behave as waves (diffraction, interference) and as particles (photoelectric effect). Their energy is proportional to frequency.",
        "The cerebral cortex is divided into four lobes: frontal (planning, motor control), parietal (sensory processing), temporal (auditory processing, memory), and occipital (visual processing). It is about 2-4 mm thick.",
        "Diamond and graphite are both made entirely of carbon atoms but have vastly different properties due to their crystal structures. Diamond has a tetrahedral sp3 bonding pattern; graphite has planar sp2 layers.",
        "The tides are primarily caused by the gravitational pull of the Moon on Earth's oceans. The Sun also contributes. Spring tides occur during full and new moons when the Sun and Moon are aligned.",
        "RNA differs from DNA in three key ways: it uses ribose instead of deoxyribose sugar, uracil instead of thymine as a base, and is usually single-stranded. Messenger RNA carries genetic information from DNA to ribosomes.",
        "The Coriolis effect causes moving objects on Earth to deflect to the right in the Northern Hemisphere and to the left in the Southern Hemisphere. It influences weather patterns, ocean currents, and cyclone rotation.",
        "Superconductors are materials that conduct electricity with zero resistance below a critical temperature. MgB2 superconducts below 39K. High-temperature superconductors like YBCO work below about 93K.",
        "The human eye can distinguish approximately 10 million different colors. Color vision is mediated by three types of cone cells sensitive to short (blue), medium (green), and long (red) wavelengths.",
        "Penicillin was discovered by Alexander Fleming in 1928 when he noticed that Penicillium mold inhibited bacterial growth. It became the first widely used antibiotic and has saved an estimated 200 million lives.",
        "The magnetic field of Earth is generated by convection currents in the liquid iron outer core. This geodynamo has reversed polarity hundreds of times over geological history, with the last reversal about 780,000 years ago.",
        "Continental drift was proposed by Alfred Wegener in 1912. Evidence includes matching coastlines, fossil distributions, rock formations, and paleoclimatic data. It was not widely accepted until the 1960s when plate tectonics was developed.",
        "The Voyager 1 spacecraft, launched in 1977, is the most distant human-made object from Earth, now over 24 billion kilometers away. It carries a Golden Record with sounds and images representing life on Earth.",
    ],
}

NUM_CATEGORIES = len(PROMPTS)
CATEGORY_NAMES = sorted(PROMPTS.keys())
CATEGORY_TO_ID = {name: i for i, name in enumerate(CATEGORY_NAMES)}


# ─── Halting Metrics ─────────────────────────────────────────────────

def compute_commit_then_violate(
    traces: List[Dict], success_multiplier: float = 1.2, K: int = 5,
) -> Dict:
    """Compute commit-then-violate (CTV) rate from experiment traces.

    A "commit" is a cheap-path step that the governor allowed. A "violation"
    occurs when any of the next K steps (within the same context class) has
    loss exceeding the running expensive-path baseline * success_multiplier.

    Args:
        traces: List of step dicts, each containing at minimum:
            - "context_class" (int): category ID for the step
            - "path" (str): "cheap" or "full"
            - "loss" (float): step loss value
            - "governor_decision" (str, optional): "allow" or "block"
        success_multiplier: Loss threshold multiplier over baseline.
        K: Number of future steps to check for violations.

    Returns:
        Dict with keys: total_commits, violated_commits,
        commit_then_violate_rate, K, success_multiplier.
    """
    by_class: Dict[int, List[Dict]] = defaultdict(list)
    for t in traces:
        by_class[t["context_class"]].append(t)

    expensive_sums: Dict[int, float] = defaultdict(float)
    expensive_counts: Dict[int, int] = defaultdict(int)
    total_commits = 0
    violated_commits = 0

    for cc, class_traces in by_class.items():
        for i, t in enumerate(class_traces):
            if t["path"] == "full":
                expensive_sums[cc] += t["loss"]
                expensive_counts[cc] += 1

            is_commit = t["path"] == "cheap"
            if t.get("governor_decision") is not None:
                is_commit = is_commit and t["governor_decision"] == "allow"

            if not is_commit:
                continue

            total_commits += 1

            if expensive_counts[cc] == 0:
                continue
            baseline = expensive_sums[cc] / expensive_counts[cc]
            threshold = baseline * success_multiplier

            for j in range(i + 1, min(i + 1 + K, len(class_traces))):
                future = class_traces[j]
                if future["loss"] > threshold:
                    violated_commits += 1
                    break

    return {
        "total_commits": total_commits,
        "violated_commits": violated_commits,
        "commit_then_violate_rate": (
            round(violated_commits / total_commits, 4)
            if total_commits > 0 else 0.0
        ),
        "K": K,
        "success_multiplier": success_multiplier,
    }
