<script lang="ts">
  import { onMount } from "svelte";
  import Editor from "./Editor.svelte";
  import Viewer from "./Viewer.svelte";
  import {
    api,
    setToken,
    type Capture,
    type CompileInput,
    type Job,
    type WorkbenchInfo,
  } from "./api";

  const draftKey = "cuteviz-python-draft-v1";
  let info = $state<WorkbenchInfo | null>(null);
  let source = $state(""),
    entry = $state("entry"),
    argsFactory = $state("make_args"),
    target = $state("");
  let tab = $state("code"),
    example = $state("copy"),
    error = $state("");
  let capture = $state<Capture | null>(null),
    captureId = $state("");
  let captureInput = $state(""),
    captureName = $state("Loaded capture");
  let job = $state<Job | null>(null),
    history = $state<Job[]>([]),
    submitting = $state(false);
  let ready = $state(false),
    draftSaved = $state(false),
    disposed = false;
  let sourceFile = $state<HTMLInputElement>();
  const running = $derived(
    submitting || (!!job && ["queued", "running"].includes(job.status)),
  );
  const input = $derived({ source, entry, args_factory: argsFactory, target });
  const stale = $derived(
    !!captureInput && captureInput !== JSON.stringify(input),
  );
  const statusLabel: Record<string, string> = {
    queued: "Queued",
    running: "Compiling…",
    succeeded: "Capture ready",
    failed: "Compile failed",
    cancelled: "Cancelled",
    timed_out: "Time limit reached",
  };

  onMount(() => {
    initialize();
    return () => {
      disposed = true;
    };
  });
  async function initialize() {
    try {
      info = await api<WorkbenchInfo>("/api/workbench");
      setToken(info.token);
      source = info.templates.find((t) => t.id === "copy")?.source ?? "";
      try {
        const saved = JSON.parse(localStorage.getItem(draftKey) ?? "null");
        if (
          saved &&
          [saved.source, saved.entry, saved.args_factory, saved.target].every(
            (x) => typeof x === "string",
          )
        ) {
          source = saved.source;
          entry = saved.entry;
          argsFactory = saved.args_factory;
          target = saved.target;
        }
      } catch {
        /* A missing or malformed local draft does not prevent editing. */
      }
      if (info.has_capture) {
        capture = await api<Capture>("/api/capture");
        tab = "inspect";
      }
      history = await api<Job[]>("/api/compilations");
      const active = history.find((j) =>
        ["queued", "running"].includes(j.status),
      );
      ready = true;
      if (active) {
        job = active;
        poll(active.id);
      }
    } catch (e) {
      error = String(e);
    }
  }
  $effect(() => {
    const draft = input;
    if (!ready) return;
    draftSaved = false;
    const timer = setTimeout(() => {
      try {
        localStorage.setItem(draftKey, JSON.stringify(draft));
        draftSaved = true;
      } catch {
        draftSaved = false;
      }
    }, 300);
    return () => clearTimeout(timer);
  });
  function remember(next: Job) {
    job = next;
    history = [next, ...history.filter((j) => j.id !== next.id)];
  }
  async function compile() {
    if (running || !info?.can_compile) return;
    error = "";
    tab = "code";
    submitting = true;
    try {
      const next = await api<Job>("/api/compilations", input);
      remember(next);
      poll(next.id);
    } catch (e) {
      error = String(e);
    } finally {
      submitting = false;
    }
  }
  async function poll(id: string) {
    while (!disposed) {
      try {
        const next = await api<Job>(`/api/compilations/${id}`);
        if (disposed) return;
        remember(next);
        if (!["queued", "running"].includes(next.status)) {
          if (next.status === "succeeded") await inspectRun(next);
          return;
        }
      } catch (e) {
        error = `Connection interrupted; retrying. ${String(e)}`;
      }
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
  async function inspectRun(run: Job) {
    try {
      const [data, submitted] = await Promise.all([
        api<Capture>(`/api/capture?capture_id=${run.id}`),
        api<CompileInput>(`/api/compilations/${run.id}/source`),
      ]);
      capture = data;
      captureId = run.id;
      captureInput = JSON.stringify(submitted);
      captureName = `${run.entry} · ${run.id.slice(0, 8)}${run.status !== "succeeded" ? " · partial run" : ""}`;
      tab = "inspect";
    } catch (e) {
      error = String(e);
    }
  }
  async function cancel() {
    if (!job) return;
    try {
      remember(await api<Job>(`/api/compilations/${job.id}/cancel`, {}));
    } catch (e) {
      error = String(e);
    }
  }
  function loadExample() {
    const template = info?.templates.find((t) => t.id === example);
    if (!template) return;
    source = template.source;
    entry = "entry";
    argsFactory = "make_args";
    if (example === "mma" && !target) target = "sm_80";
  }
  async function importSource(event: Event) {
    const element = event.target as HTMLInputElement;
    const file = element.files?.[0];
    try {
      if (file) {
        if (file.size > 256_000)
          throw new Error("Python source is limited to 256 KB.");
        source = await file.text();
      }
    } catch (e) {
      error = String(e);
    }
    element.value = "";
  }
  function downloadSource() {
    const url = URL.createObjectURL(
      new Blob([source], { type: "text/x-python" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = "kernel.py";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
</script>

<header class="topbar">
  <a class="brand" href="/" aria-label="CuteViz home"
    ><span class="brand-icon">▦</span> Cute<span>Viz</span></a
  >
  <span class="top-description">KERNEL WORKBENCH</span>
  <span class="capture-badge">Compile-time observations</span>
</header>
<div class="workbench-bar">
  <div class="workbench-tabs" role="tablist" aria-label="Workbench view">
    <button
      role="tab"
      id="code-tab"
      aria-selected={tab === "code"}
      aria-controls="code-panel"
      onclick={() => (tab = "code")}>01 <span>Code</span></button
    >
    <button
      role="tab"
      id="inspect-tab"
      aria-selected={tab === "inspect"}
      aria-controls="inspect-panel"
      disabled={!capture}
      onclick={() => (tab = "inspect")}
      >02 <span>Inspect</span>{#if capture}<small
          >{capture.objects.length}</small
        >{/if}</button
    >
  </div>
  <div class="run-actions">
    {#if job}<span class={`run-status ${job.status}`} role="status"
        >{statusLabel[job.status]}</span
      >{/if}
    {#if running}<button onclick={cancel} disabled={submitting}
        >Cancel compilation</button
      >{/if}
    <button
      class="primary"
      onclick={compile}
      disabled={!info?.can_compile || running || !source.trim()}
      >▶ Compile &amp; capture</button
    >
  </div>
</div>
{#if error}<p class="notice app-notice" role="alert">
    {error} <button onclick={() => (error = "")}>Dismiss</button>
  </p>{/if}
{#if !info}<main><p>Connecting to the local workbench…</p></main>{:else}
  <div
    id="code-panel"
    role="tabpanel"
    aria-labelledby="code-tab"
    hidden={tab !== "code"}
  >
    <div class="editor-page">
      <div class="editor-main">
        <div class="editor-heading">
          <div>
            <p class="eyebrow">WRITE · COMPILE · EXPLORE</p>
            <h1>Your kernel, made visible.</h1>
            <p class="muted">
              Add inspection probes, then follow their mappings in the
              inspector.
            </p>
          </div>
          <span class="tag">Python / CuTe DSL</span>
        </div>
        <div class="compile-config">
          <label
            >Entry function<input
              aria-label="Entry function"
              bind:value={entry}
              placeholder="entry"
              spellcheck="false"
            /></label
          >
          <label
            >Arguments factory<input
              aria-label="Arguments factory"
              bind:value={argsFactory}
              placeholder="Optional"
              spellcheck="false"
            /></label
          >
          <label
            >Target architecture<input
              aria-label="Target architecture"
              bind:value={target}
              list="targets"
              placeholder="Auto / environment"
              spellcheck="false"
            /></label
          >
          <datalist id="targets"
            ><option value="sm_80">Ampere</option><option value="sm_90a"
              >Hopper</option
            ><option value="sm_100a">Blackwell tcgen05</option></datalist
          >
        </div>
        <div class="editor-card">
          <div class="editor-toolbar">
            <span>kernel.py</span><span class="muted"
              >{draftSaved ? "Draft saved locally" : "Editing"}</span
            >
            <div class="editor-file-actions">
              <button onclick={() => sourceFile?.click()}>Open .py</button
              ><button onclick={downloadSource}>Save .py</button>
            </div>
          </div>
          <input
            type="file"
            accept=".py,text/x-python,text/plain"
            bind:this={sourceFile}
            onchange={importSource}
            hidden
          />
          <Editor
            value={source}
            onchange={(text) => (source = text)}
            oncompile={compile}
          />
          <div class="editor-footer">
            <span
              >Python syntax · Tab to indent · Esc, then Tab to leave editor</span
            ><span>Ctrl / ⌘ + Enter to compile</span>
          </div>
        </div>
        {#if job}<section class="compiler-console" aria-label="Compiler output">
            <div class="console-heading">
              <h2>{statusLabel[job.status]}</h2>
              <span
                >{job.id.slice(0, 8)} · {job.finished
                  ? `${(job.finished - job.started).toFixed(1)}s`
                  : `limit ${info.timeout_seconds}s`}</span
              >
            </div>
            {#if job.error}<p class="notice" role="alert">{job.error}</p>{/if}
            <pre>{job.log || "Waiting for compiler output…"}</pre>
            {#if job.capture_available}<div class="console-actions">
                <button onclick={() => inspectRun(job!)}
                  >{job.status === "succeeded"
                    ? "Inspect capture"
                    : "Inspect partial capture"}</button
                ><a href={`/api/compilations/${job.id}/capture`} download
                  >Download .cuteviz.json</a
                ><small>{job.object_count} observations · {job.outcome}</small>
              </div>
              <p class="saved-path">Saved to {job.capture_path}</p>{/if}
          </section>{/if}
      </div>
      <aside class="editor-guide">
        <section class="guide-card">
          <p class="eyebrow">START EXPLORING</p>
          <h2>Bring a kernel. Add a probe.</h2>
          <p>
            Write a <code>@cute.jit</code> entry that launches your
            <code>@cute.kernel</code>. Probes observe objects during
            compilation.
          </p>
          <pre>cuteviz.inspect("tile", tile)
cuteviz.inspect_copy("copy", copy,
    src=gA, dst=sA)
cuteviz.inspect_mma("mma", mma,
    a=a, b=b, c=acc)</pre>
          <label
            >Example<select aria-label="Example" bind:value={example}
              ><option value="copy">Copy + tensor layouts</option><option
                value="mma">Ampere / Hopper / Blackwell MMA</option
              ><option value="tiles">CTA tiles + backing tensors</option
              ></select
            ></label
          ><button onclick={loadExample}>Load example</button>
        </section>
        <section class="guide-card">
          <p class="eyebrow">CONNECT YOUR ARGUMENTS</p>
          <p>
            <code>make_args()</code> returns the arguments for
            <code>cute.compile(entry, *args)</code>. Use fake tensors or pointer
            placeholders when you only need compilation.
          </p>
          <pre>def make_args():
    return (a, b, c)</pre>
          <p>
            For keyword arguments or compiler options, return <code
              >{'{"args": (a, b), "kwargs": {...}}'}</code
            >. Omit the factory for an entry without arguments.
          </p>
          <p>Local project imports resolve from the working directory below.</p>
        </section>
        <section class="guide-card environment">
          <p class="eyebrow">LOCAL PYTHON ENVIRONMENT</p>
          <p>CuTeDSL {info.compiler ?? "not installed"}</p>
          <code>{info.python}</code>
          <p>Working directory</p>
          <code>{info.workdir}</code>
          <p>
            Python code runs with your local user permissions. The app compiles
            the entry; it does not launch the compiled kernel. Top-level code
            and the argument factory execute normally.
          </p>
          {#if !info.can_compile}<p class="notice">{info.reason}</p>{/if}
        </section>
        {#if history.length}<section class="guide-card run-history">
            <p class="eyebrow">SESSION RUNS</p>
            {#each history as run}<div>
                <button
                  disabled={!run.capture_available ||
                    ["queued", "running"].includes(run.status)}
                  onclick={() => inspectRun(run)}
                  ><strong>{run.entry}</strong><span
                    >{statusLabel[run.status]} · {run.object_count} objects</span
                  ></button
                ><small
                  >{new Date(run.started * 1000).toLocaleTimeString()} · {run.target ||
                    "auto"}</small
                >
              </div>{/each}
          </section>{/if}
      </aside>
    </div>
  </div>
  <div
    id="inspect-panel"
    role="tabpanel"
    aria-labelledby="inspect-tab"
    hidden={tab !== "inspect"}
  >
    {#if capture}<div class="capture-toolbar">
        <span>{captureName}</span>{#if stale}<span class="draft-difference"
            >Draft differs from this capture. Compile to refresh.</span
          >{/if}{#if captureId}<a
            href={`/api/compilations/${captureId}/capture`}
            download>Download .cuteviz.json</a
          >{/if}<button onclick={() => (tab = "code")}>Back to editor</button>
      </div>
      {#key captureId}<Viewer {capture} {captureId} />{/key}{/if}
  </div>
{/if}
