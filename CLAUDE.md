Add as a top-level section near the top of CLAUDE.md so it's always in context\n\n## Platform
This is a Windows development environment. Always account for Windows-specific limitations:
- `torch.compile` and Triton are NOT supported on Windows
- Use single slashes for PowerShell flags (e.g., `taskkill /F /PID`), not double slashes
- `os.execl` is unreliable on Windows; avoid it
- `persistent_workers=True` in DataLoader can cause page file exhaustion on low-RAM systems
- Always use `num_workers=0` when RAM caching is enabled
Add under a ## Hardware Constraints section, right after the Platform section\n\n## Hardware Constraints
System has 16GB RAM and limited VRAM. When writing ML code:
- Cache datasets as uint8, never float32 (float32 caching can use 25GB+ RAM)
- Set `num_workers=0` when using in-memory caching
- Avoid duplicate DataLoader rebuilds that cause double caching
- Always consider memory footprint before suggesting caching strategies
Add under a ## Debugging Approach section in CLAUDE.md\n\n## Debugging Approach
When fixing errors, always check the actual codebase first before giving generic advice. When a fix fails, do NOT just tweak the same approach — consider whether the approach itself is wrong for this platform/hardware. After applying a fix, verify there are no inconsistencies with the rest of the codebase (e.g., mismatched checkpoint keys, missing imports, mismatched normalization stats).