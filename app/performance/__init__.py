"""Phase 9: performance measurement, smart caching, resource limits, hardware capability and large-project handling.

Layering: nothing here imports Qt. ``metrics`` / ``profiler`` / ``resource_monitor`` are dependency-free building blocks that every other
package may import; the services built on them (cache manager, hardware service, scheduler hooks) live in sibling modules.
"""
