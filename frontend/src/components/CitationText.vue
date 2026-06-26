<template>
  <span class="citation-text whitespace-pre-wrap">
    <span v-for="(part, index) in parts" :key="index">
      <button
        v-if="part.type === 'citation'"
        type="button"
        class="citation-link text-blue-600 hover:text-blue-800 underline font-mono text-sm mx-0.5"
        @click="emit('citeClick', part.docId, part.chunkIndex, part.refId)"
      >
        [{{ part.refId }}]
      </button>
      <span v-else>{{ part.text }}</span>
    </span>
  </span>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const CITATION_RE = /\[([0-9a-fA-F-]{36}#(\d+))\]/g

type TextPart = { type: 'text'; text: string }
type CitationPart = { type: 'citation'; refId: string; docId: string; chunkIndex: number }
type Part = TextPart | CitationPart

const props = defineProps<{
  content: string
}>()

const emit = defineEmits<{
  citeClick: [docId: string, chunkIndex: number, refId: string]
}>()

const parts = computed<Part[]>(() => {
  const text = props.content
  if (!text) return []

  const result: Part[] = []
  let lastIndex = 0
  let match: RegExpExecArray | null

  const re = new RegExp(CITATION_RE.source, 'g')
  while ((match = re.exec(text)) !== null) {
    if (match.index > lastIndex) {
      result.push({ type: 'text', text: text.slice(lastIndex, match.index) })
    }
    const refId = match[1]
    const docId = refId.slice(0, 36)
    const chunkIndex = Number(match[2])
    result.push({ type: 'citation', refId, docId, chunkIndex })
    lastIndex = match.index + match[0].length
  }

  if (lastIndex < text.length) {
    result.push({ type: 'text', text: text.slice(lastIndex) })
  }

  return result.length ? result : [{ type: 'text', text }]
})
</script>
