<script lang="ts">
  import { onMount } from "svelte";
  import { basicSetup } from "codemirror";
  import { EditorView, keymap } from "@codemirror/view";
  import { EditorState } from "@codemirror/state";
  import { indentWithTab } from "@codemirror/commands";
  import { indentUnit } from "@codemirror/language";
  import { python } from "@codemirror/lang-python";

  let {
    value,
    onchange,
    oncompile,
  }: {
    value: string;
    onchange: (text: string) => void;
    oncompile: () => void;
  } = $props();
  let container: HTMLDivElement;
  let editor: EditorView | undefined;
  onMount(() => {
    editor = new EditorView({
      parent: container,
      state: EditorState.create({
        doc: value,
        extensions: [
          python(),
          indentUnit.of("    "),
          keymap.of([
            {
              key: "Mod-Enter",
              run: () => {
                oncompile();
                return true;
              },
            },
            indentWithTab,
          ]),
          basicSetup,
          EditorView.contentAttributes.of({
            "aria-label": "Python kernel editor",
            spellcheck: "false",
          }),
          EditorView.updateListener.of((update) => {
            if (update.docChanged) onchange(update.state.doc.toString());
          }),
          EditorView.theme({
            "&": { height: "100%", fontSize: "13px" },
            ".cm-scroller": {
              overflow: "auto",
              fontFamily: "'SFMono-Regular', Consolas, monospace",
              lineHeight: "1.65",
            },
            ".cm-content": { padding: "14px 0" },
            ".cm-gutters": {
              backgroundColor: "#f4f7f6",
              borderRight: "1px solid #e0e8e5",
              color: "#8a9999",
            },
            ".cm-activeLine": { backgroundColor: "#e8f2ef80" },
          }),
        ],
      }),
    });
    return () => editor?.destroy();
  });
  $effect(() => {
    const next = value;
    if (editor && editor.state.doc.toString() !== next)
      editor.dispatch({
        changes: { from: 0, to: editor.state.doc.length, insert: next },
      });
  });
</script>

<div class="python-editor" bind:this={container}></div>
