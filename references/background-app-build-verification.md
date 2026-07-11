# Background App Build Verification Pattern

Use this when a user asks super-router to implement a GUI/native app in background mode, especially when the artifact must be directly runnable by clicking an icon.

## Durable lesson

For implementation tasks, a router run that creates source files or an app-bundle scaffold but reports the build or smoke test as `BLOCKED` is not a completed deliverable. Treat it as an incomplete artifact until the executable has been built and verified directly.

## Prompt requirements for the router task

Include explicit language that the router must:

1. Create the source/project files.
2. Build the app with the installed toolchain.
3. Package the clickable artifact (`.app`, installer, binary bundle, etc.).
4. Verify the artifact directly, not just the source tree:
   - bundle/app directory exists
   - metadata file exists (`Info.plist`, manifest, package metadata)
   - executable exists and has executable permissions
   - build command returns success with actual output
   - for macOS `.app` bundles, `plutil -lint <app>/Contents/Info.plist` passes
   - for macOS `.app` bundles, `file <app>/Contents/MacOS/<exe>` reports the expected Mach-O executable architecture
   - for macOS `.app` bundles, `codesign --verify --deep --strict <app>` passes when code signing is available/applicable
   - non-interactive smoke test or launch probe does not immediately crash
5. Prefer implementing a deterministic app-level smoke mode such as `--smoke` that prints a launch-confirmation line and exits 0. Verify it by invoking the bundle executable directly (for example `<app>/Contents/MacOS/<exe> --smoke`) instead of only opening the GUI and inferring success from process state.
6. If a smoke probe opens the GUI without an exit path, check for and clean up any lingering test process before reporting success. A still-running app can prove launch viability, but it should not be the only smoke evidence when a deterministic smoke path can be added.
7. Report any blocked step as a blocker, not as success.

## Post-run handling

After the background process finishes:

1. Read the final router output for explicit `SUCCESS`, `FAILED`, or `BLOCKED` statuses.
2. If build or smoke-test steps are `BLOCKED`, `FAILED`, or absent, do not claim the app is runnable.
3. If the user required "only through super-router", do not build or patch the artifact outside the router. The only allowed continuation is another super-router task that specifically completes build/debug/verification.
4. Before reporting success, inspect the declared artifacts directly when tool permissions allow. If artifact verification is unavailable, say exactly what evidence is missing.

## Correction-pass prompt shape

When a previous router run scaffolded files but failed to build, the correction prompt should be narrow and concrete:

```text
Continue and complete the existing app implementation strictly inside this super-router run.
Previous run created files under <project-path> but reported build/smoke test BLOCKED.
Do not summarize. Inspect existing files, fix compile/package issues, execute the actual build, verify the clickable artifact, run endpoint/smoke checks, and report command evidence.
```

## Final response rule

For app-delivery tasks, final status should be one of:

- `Router result: verified runnable artifact` — only when build and artifact checks passed.
- `Router failed: incomplete artifact` — when source/scaffold exists but build/smoke verification did not complete.
- `Router blocked: needs another router run` — when continuation is required but cannot be launched.
