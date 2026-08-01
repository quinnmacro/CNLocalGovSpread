/**
 * Minimal theme hook — this project is dark-first and fixed.
 * Returns the current theme class.
 */
export function useTheme() {
  // Project is dark-first, fixed theme
  return { theme: "dark" as const, isDark: true };
}
