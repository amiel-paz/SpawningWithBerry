"""Deterministic dependency-aware local task queue."""

from __future__ import annotations

import dataclasses
import enum
import heapq
import itertools
from collections.abc import Callable
from typing import Any, Protocol


class TaskKind(enum.StrEnum):
    TRAJECTORY = "trajectory"
    BACKPROPAGATE = "backpropagate"
    CENTROID = "centroid"
    QUANTUM = "quantum"
    SPAWN = "spawn"
    OUTPUT = "output"


@dataclasses.dataclass(order=True)
class Task:
    priority: tuple[float, int, str]
    kind: TaskKind = dataclasses.field(compare=False)
    identifier: str = dataclasses.field(compare=False)
    dependencies: frozenset[str] = dataclasses.field(default_factory=frozenset, compare=False)
    payload: dict[str, Any] = dataclasses.field(default_factory=dict, compare=False)


class Executor(Protocol):
    def execute(self, task: Task, action: Callable[[Task], Any]) -> Any: ...


class SerialExecutor:
    def execute(self, task: Task, action: Callable[[Task], Any]) -> Any:
        return action(task)


class TaskQueue:
    def __init__(self) -> None:
        self._heap: list[Task] = []
        self.completed: set[str] = set()
        self.failed: dict[str, str] = {}
        self._counter = itertools.count()

    def add(self, kind: TaskKind, time: float, identifier: str, *, dependencies=(), payload=None) -> Task:
        order = list(TaskKind).index(kind)
        task = Task((float(time), order, f"{next(self._counter):012d}"), kind, identifier, frozenset(dependencies), payload or {})
        heapq.heappush(self._heap, task)
        return task

    def pop_ready(self) -> Task | None:
        blocked: list[Task] = []
        selected = None
        while self._heap:
            task = heapq.heappop(self._heap)
            if task.dependencies <= self.completed:
                selected = task
                break
            blocked.append(task)
        for task in blocked:
            heapq.heappush(self._heap, task)
        return selected

    def complete(self, task: Task) -> None:
        self.completed.add(task.identifier)
        self._heap = [queued for queued in self._heap if queued.identifier != task.identifier]
        heapq.heapify(self._heap)

    def fail(self, task: Task, error: BaseException) -> None:
        self.failed[task.identifier] = str(error)
        self._heap = [queued for queued in self._heap if queued.identifier != task.identifier]
        heapq.heapify(self._heap)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pending": [
                {"priority": list(t.priority), "kind": t.kind.value, "identifier": t.identifier,
                 "dependencies": sorted(t.dependencies), "payload": t.payload}
                for t in sorted(self._heap)
            ],
            "completed": sorted(self.completed),
            "failed": self.failed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskQueue":
        queue = cls()
        queue.completed = set(data.get("completed", []))
        queue.failed = dict(data.get("failed", {}))
        for item in data.get("pending", []):
            task = Task(tuple(item["priority"]), TaskKind(item["kind"]), item["identifier"], frozenset(item["dependencies"]), item["payload"])
            heapq.heappush(queue._heap, task)
        return queue

    def __bool__(self) -> bool:
        return bool(self._heap)
