# Contributing

This is primarily a personal learning project, but bug fixes and improvements are welcome.

## Quick Guidelines

- Follow existing code style (see below)
- Add tests for new functionality
- Keep it simple - this is an educational implementation

## Code Style

**Naming**:
- Types/Classes: `PascalCase`
- Functions: `snake_case`
- Member variables: `snake_case_` (trailing underscore)

**Organization**:
- Header files minimal, implementation in .cpp
- Includes: standard library, then project headers

## Building

```bash
mkdir build && cd build
cmake ..
cmake --build .
./tests/example_usage
```
