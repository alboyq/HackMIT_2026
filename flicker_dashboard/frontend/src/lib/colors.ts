export const OBJECT_COLORS = ['#5eead4', '#c4b5fd', '#7dd3fc', '#fb7185']

export function colorForIndex(i: number): string {
  return OBJECT_COLORS[i % OBJECT_COLORS.length]
}
